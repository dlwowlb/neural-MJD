"""
Ours: Event-Marked Neural MJD.

This extends the repository's time-inhomogeneous Merton jump diffusion from an
*unlabeled* Poisson jump measure into an **event-marked** one. Every jump now
carries a source label r:

    r = 0       background jump
    r = i >= 1  a jump produced by observed event e_i

and each source has its OWN time-varying jump intensity and (log-normal) jump
magnitude distribution -- not just a split of a single lambda_t. Concretely the
network predicts, per future step t,

    mu_0,t , sigma_t                         (shared drift / diffusion)
    { lambda_{r,t}, nu_{r,t}, gamma_{r,t} }  for r = 0..m   (per-source jumps)

with the per-source expected relative jump size

    k_{r,t} = exp(nu_{r,t} + gamma_{r,t}^2 / 2) - 1.

Compensated drift and closed-form mean (Neural MJD Eq. 6/9) are preserved by
defining the effective drift

    mu_t^E = mu_0,t + sum_r lambda_{r,t} k_{r,t}      =>   E[S_T|C,E] = S_0 exp( sum_t mu_t^E )

so each source contributes an *attributable* term  c_{r,t} = lambda_{r,t} k_{r,t}
to the conditional-mean log-return. Removing event i is then exact:
mu^{(-i)} = mu^E - c_{i,t}.

Delay is modelled explicitly, not as a bare (t - tau_i) input. For event i with
elapsed time u = t - tau_i, a learnable onset distribution p_phi(delta|q_i) is
convolved with a response kernel kappa_phi(.|q_i):

    lambda_{i,t} = a_i(q_i) * sum_{delta} p_phi(delta|q_i) kappa_phi(u - delta | q_i),  u >= 1

so the model separates *when the response starts* (onset) from *how it unfolds*
(kernel) from *how strong it is* (a_i). Event interaction enters through q_i,
which attends over the other events and the past context.

The training loss keeps the MJD source-count likelihood (with adaptive
truncation over the total count) plus the conditional-mean regulariser, and -- on
the synthetic benchmark where ground truth exists -- light attribution/delay
supervision.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .synthetic import PAST_LEN, FUTURE_LEN, TOTAL_LEN, N_EVENT_CHANNELS

MAX_EVENTS = 2          # m
N_SOURCES = MAX_EVENTS + 1   # background + events
ONSET_BINS = 13         # delta = 0..12
KERNEL_LAGS = 9         # kappa lag = 0..8
PROFILE_LEN = ONSET_BINS + KERNEL_LAGS - 1   # phi(u), u = 0..20


def events_to_arrays(seqs):
    """Pack a list of sequence dicts into padded event tensors.

    Returns numpy arrays:
        tau  [B, m]   event time (global past index)
        typ  [B, m]   0 = up/meal, 1 = down/insulin
        mag  [B, m]   magnitude
        mask [B, m]   1 if the event is present
    """
    B = len(seqs)
    tau = np.zeros((B, MAX_EVENTS), np.float32)
    typ = np.zeros((B, MAX_EVENTS), np.float32)
    mag = np.zeros((B, MAX_EVENTS), np.float32)
    mask = np.zeros((B, MAX_EVENTS), np.float32)
    for b, s in enumerate(seqs):
        for i, ev in enumerate(s["events"][:MAX_EVENTS]):
            tau[b, i] = ev["tau"]
            typ[b, i] = 0.0 if ev["c"] == +1 else 1.0
            mag[b, i] = ev["m"]
            mask[b, i] = 1.0
    return tau, typ, mag, mask


class EventMarkedMJD(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.H = hidden

        # --- past-trajectory encoder -> shared context h ---------------------
        self.past_enc = nn.Sequential(
            nn.Linear(PAST_LEN * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # shared per-future-step drift/diffusion + background jump params
        # outputs: mu0, log_sigma, bg_lambda_raw, bg_nu, bg_log_gamma  (5 * F)
        self.shared_head = nn.Linear(hidden, FUTURE_LEN * 5)

        # --- event encoder (with interaction over events + context) ----------
        ev_feat_dim = 2 + 1 + 1            # type one-hot, magnitude, tau_norm
        self.ev_in = nn.Linear(ev_feat_dim, hidden)
        self.ev_interact = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.ev_mix = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # per-event heads -> strength, onset pmf logits, kernel, nu, log-gamma
        self.head_a = nn.Linear(hidden, 1)
        self.head_onset = nn.Linear(hidden, ONSET_BINS)
        self.head_kernel = nn.Linear(hidden, KERNEL_LAGS)
        self.head_nu = nn.Linear(hidden, 1)
        self.head_lgamma = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _profile(onset_pmf, kernel):
        """Discrete convolution phi(u) = sum_delta p(delta) kappa(u - delta).

        onset_pmf [B, m, ONSET_BINS]  (sums to 1 over delta)
        kernel    [B, m, KERNEL_LAGS] (nonneg)
        returns   [B, m, PROFILE_LEN]
        """
        B, m, _ = onset_pmf.shape
        phi = onset_pmf.new_zeros(B, m, PROFILE_LEN)
        for d in range(ONSET_BINS):
            phi[:, :, d:d + KERNEL_LAGS] += onset_pmf[:, :, d:d + 1] * kernel
        return phi

    def forward(self, past_dyn, tau, typ, mag, mask):
        """
        past_dyn [B, 2, P]   (normalised price, time index)
        tau,typ,mag,mask [B, m]
        Returns a dict of per-future-step source parameters and the effective
        drift mu^E (all on the normalised log scale).
        """
        B = past_dyn.size(0)
        h = self.past_enc(past_dyn.reshape(B, -1))                    # [B, H]

        # shared per-step params
        sh = self.shared_head(h).view(B, FUTURE_LEN, 5)
        mu0 = sh[..., 0]
        sigma = F.softplus(sh[..., 1]).clamp(1e-3, 1.0)
        bg_lambda = F.softplus(sh[..., 2]).clamp(0.0, 3.0)
        bg_nu = sh[..., 3].clamp(-0.5, 0.5)
        bg_gamma = F.softplus(sh[..., 4]).clamp(1e-3, 1.0)

        # event embeddings with interaction
        type_oh = torch.stack([1 - typ, typ], dim=-1)                # [B, m, 2]
        ev_feat = torch.cat([type_oh, mag.unsqueeze(-1),
                             (tau / PAST_LEN).unsqueeze(-1)], dim=-1)  # [B, m, 4]
        e = torch.relu(self.ev_in(ev_feat))                          # [B, m, H]
        key_pad = (mask < 0.5)                                       # [B, m] True=pad
        attn, _ = self.ev_interact(e, e, e, key_padding_mask=key_pad)
        attn = torch.nan_to_num(attn)
        q = self.ev_mix(torch.cat([e + attn, h.unsqueeze(1).expand(-1, MAX_EVENTS, -1)], -1))

        a = F.softplus(self.head_a(q)).squeeze(-1).clamp(0.0, 5.0)    # [B, m]
        onset = F.softmax(self.head_onset(q), dim=-1)                # [B, m, ONSET_BINS]
        kernel = F.softplus(self.head_kernel(q))                     # [B, m, KERNEL_LAGS]
        ev_nu = self.head_nu(q).squeeze(-1).clamp(-0.5, 0.5)         # [B, m]
        ev_gamma = F.softplus(self.head_lgamma(q)).squeeze(-1).clamp(1e-3, 1.0)

        # delay-aware per-event intensity over future steps
        phi = self._profile(onset, kernel)                           # [B, m, PROFILE_LEN]
        fut_global = torch.arange(PAST_LEN, TOTAL_LEN, device=past_dyn.device)  # [F]
        u = fut_global.view(1, 1, FUTURE_LEN) - tau.view(B, MAX_EVENTS, 1)      # [B, m, F]
        valid = (u >= 1) & (u <= PROFILE_LEN - 1)
        u_idx = u.clamp(0, PROFILE_LEN - 1).long()
        phi_u = torch.gather(phi, 2, u_idx)                          # [B, m, F]
        phi_u = phi_u * valid.float() * mask.unsqueeze(-1)
        ev_lambda = a.unsqueeze(-1) * phi_u                          # [B, m, F]

        # assemble per-source tensors: index 0 = background, 1..m = events
        lam = torch.cat([bg_lambda.unsqueeze(1), ev_lambda], dim=1)             # [B, 1+m, F]
        nu = torch.cat([bg_nu.unsqueeze(1),
                        ev_nu.unsqueeze(-1).expand(-1, -1, FUTURE_LEN)], dim=1)
        gamma = torch.cat([bg_gamma.unsqueeze(1),
                           ev_gamma.unsqueeze(-1).expand(-1, -1, FUTURE_LEN)], dim=1)
        k = torch.exp(nu + gamma.square() / 2.0) - 1.0                          # [B, 1+m, F]

        mu_eff = mu0 + (lam * k).sum(dim=1)                                     # [B, F]
        onset_mean = (onset * torch.arange(ONSET_BINS, device=q.device)).sum(-1)  # [B, m]

        return {
            "mu0": mu0, "sigma": sigma, "mu_eff": mu_eff,
            "lambda": lam, "nu": nu, "gamma": gamma, "k": k,
            "ev_lambda": ev_lambda, "onset_mean": onset_mean, "mask": mask,
        }


# ----------------------------------------------------------------------------
# Batching
# ----------------------------------------------------------------------------
def make_batch_ours(seqs, device):
    """Pack past inputs, events, targets and ground truth for Ours."""
    B = len(seqs)
    past = np.zeros((B, 2, PAST_LEN), np.float32)
    s0 = np.zeros(B, np.float32)
    target = np.zeros((B, FUTURE_LEN), np.float32)
    logret = np.zeros((B, FUTURE_LEN), np.float32)
    cf = np.zeros((B, MAX_EVENTS, FUTURE_LEN), np.float32)
    attr = np.full((B, FUTURE_LEN), -1, np.int64)
    for b, s in enumerate(seqs):
        coef = max(float(s["s"][:PAST_LEN].max()), 1e-6)
        past[b, 0] = s["s"][:PAST_LEN] / coef
        past[b, 1] = np.arange(PAST_LEN, dtype=np.float32) / PAST_LEN
        s0[b] = s["s"][PAST_LEN - 1]
        fut = s["s"][PAST_LEN:]
        target[b] = fut
        logret[b] = np.log(fut) - np.log(np.concatenate([[s["s"][PAST_LEN - 1]], fut[:-1]]))
        for i in range(min(MAX_EVENTS, len(s["events"]))):
            cf[b, i] = s["counterfactual"][i][PAST_LEN:]
        attr[b] = s["attr_label"]
    tau, typ, mag, mask = events_to_arrays(seqs)
    t = lambda x: torch.from_numpy(x).to(device)
    return {
        "past": t(past), "tau": t(tau), "typ": t(typ), "mag": t(mag), "mask": t(mask),
        "s0": t(s0), "target": t(target), "logret": t(logret),
        "cf": t(cf), "attr": t(attr), "seqs": seqs,
    }


# ----------------------------------------------------------------------------
# Closed-form mean, counterfactual, source-count likelihood
# ----------------------------------------------------------------------------
def forecast(out, s0):
    """Conditional-mean point forecast S_0 exp(cumsum mu^E), [B, F]."""
    return s0.unsqueeze(1) * torch.exp(torch.cumsum(out["mu_eff"], dim=1))


def counterfactual(out, s0, i):
    """Forecast with event i removed: mu^E - lambda_i k_i, [B, F]."""
    c_i = out["lambda"][:, i + 1, :] * out["k"][:, i + 1, :]
    mu_cf = out["mu_eff"] - c_i
    return s0.unsqueeze(1) * torch.exp(torch.cumsum(mu_cf, dim=1))


_COUNT_CACHE = {}


def _count_tuples(kappa, n_sources, device):
    key = (kappa, n_sources)
    if key not in _COUNT_CACHE:
        tuples = [c for c in np.ndindex(*([kappa + 1] * n_sources)) if sum(c) <= kappa]
        _COUNT_CACHE[key] = torch.tensor(tuples, dtype=torch.float32)
    return _COUNT_CACHE[key].to(device)


def nll(out, logret, kappa=3):
    """MJD source-count negative log-likelihood per step, summed over horizon.

    p(r_t) = sum_k [ prod_r Pois(k_r; lambda_{r,t}) ] N(r_t; a_k, b_k^2),
    with the total count truncated at kappa (adaptive bound; our intensities are
    small so kappa=3 covers >1-eps of the Poisson mass).
    """
    lam = out["lambda"].clamp(min=1e-8)                 # [B, R, F]
    nu, gamma, mu0, sigma = out["nu"], out["gamma"], out["mu0"], out["sigma"]
    B, R, Fh = lam.shape
    counts = _count_tuples(kappa, R, lam.device)        # [T, R]
    T = counts.shape[0]

    cc = counts.view(T, 1, R, 1)                        # [T,1,R,1]
    lam_e = lam.unsqueeze(0)                            # [1,B,R,F]
    # log prod_r Pois(k_r; lambda_r)
    log_pois = (cc * torch.log(lam_e) - lam_e - torch.lgamma(cc + 1.0)).sum(2)  # [T,B,F]
    a = (mu0 - sigma.square() / 2.0).unsqueeze(0) + (cc.squeeze(-1).unsqueeze(-1) * nu.unsqueeze(0)).sum(2)
    b2 = sigma.square().unsqueeze(0) + (cc.squeeze(-1).unsqueeze(-1) * gamma.square().unsqueeze(0)).sum(2)
    b2 = b2.clamp(min=1e-6)
    x = logret.unsqueeze(0)                             # [1,B,F]
    log_norm = -0.5 * (x - a).square() / b2 - 0.5 * torch.log(2 * math.pi * b2)
    log_p = torch.logsumexp(log_pois + log_norm, dim=0)  # [B,F]
    return -log_p.sum(dim=1)                            # [B]


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_ours(model, seqs, device, epochs=80, batch_size=64, lr=2e-3,
               w_mean=30.0, w_nll=1.0, w_attr=1.0, w_delay=0.3, w_sparse=0.02,
               log_every=40, verbose=True):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    model.train()
    n = len(seqs)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            bd = make_batch_ours([seqs[j] for j in perm[i:i + batch_size]], device)
            out = model(bd["past"], bd["tau"], bd["typ"], bd["mag"], bd["mask"])
            fc = forecast(out, bd["s0"])

            mean_loss = F.huber_loss(fc, bd["target"], delta=1.0)
            nll_loss = nll(out, bd["logret"]).mean() / FUTURE_LEN

            # attribution supervision on response steps
            ev_lam = out["ev_lambda"]                         # [B, m, F]
            p_src = ev_lam / (ev_lam.sum(1, keepdim=True) + 1e-6)
            attr = bd["attr"]
            resp = attr >= 0
            if resp.any():
                logp = torch.log(p_src.clamp(min=1e-6))       # [B, m, F]
                tgt = attr.clamp(min=0).unsqueeze(1)          # [B,1,F]
                picked = torch.gather(logp, 1, tgt).squeeze(1)  # [B,F]
                attr_loss = -(picked * resp.float()).sum() / resp.float().sum()
                # sparsity: event intensity should be ~0 on background steps
                sparse_loss = (ev_lam.sum(1) * (~resp).float()).sum() / (~resp).float().sum().clamp(min=1)
            else:
                attr_loss = torch.zeros((), device=device)
                sparse_loss = torch.zeros((), device=device)

            # delay supervision: predicted onset mean ~ true response delay
            delay_loss = ((out["onset_mean"] - 8.0).abs() * bd["mask"]).sum() / bd["mask"].sum().clamp(min=1)

            loss = (w_mean * mean_loss + w_nll * nll_loss + w_attr * attr_loss
                    + w_delay * delay_loss + w_sparse * sparse_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  loss {tot / max(nb,1):.4f}")
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Native evaluation (no occlusion -- attribution is intrinsic)
# ----------------------------------------------------------------------------
def _iou(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _macro_f1(true_lab, pred_lab, classes):
    f1s = []
    for c in classes:
        if np.sum(true_lab == c) == 0:
            continue
        tp = np.sum((pred_lab == c) & (true_lab == c))
        fp = np.sum((pred_lab == c) & (true_lab != c))
        fn = np.sum((pred_lab != c) & (true_lab == c))
        den = 2 * tp + fp + fn
        f1s.append(0.0 if den == 0 else 2 * tp / den)
    return float(np.mean(f1s)) if f1s else np.nan


@torch.no_grad()
def evaluate_ours(model, seqs, device):
    model.eval()
    bd = make_batch_ours(seqs, device)
    out = model(bd["past"], bd["tau"], bd["typ"], bd["mag"], bd["mask"])
    fc = forecast(out, bd["s0"]).cpu().numpy()
    ev_lam = out["ev_lambda"].cpu().numpy()                # [B, m, F]
    k = out["k"].cpu().numpy()                             # [B, 1+m, F]
    target = bd["target"].cpu().numpy()
    n_events = MAX_EVENTS

    forecast_mae = float(np.mean(np.abs(fc - target)))

    # source magnitude contribution |lambda_i k_i| for jump-time / saliency
    mag_contrib = np.abs(ev_lam * k[:, 1:, :])            # [B, m, F]

    f1_list, iou_list, jt_list = [], [], []
    for b, s in enumerate(seqs):
        attr = s["attr_label"]
        resp = np.where(attr != -1)[0]
        pred_src = np.argmax(ev_lam[b], axis=0)
        if len(resp) > 0:
            f1_list.append(_macro_f1(attr[resp], pred_src[resp], list(range(n_events))))
        sal = ev_lam[b].max(0)
        thr = 0.25 * (sal.max() if sal.max() > 0 else 1.0)
        salient = sal > thr
        for i in range(n_events):
            pred_idx = np.where(salient & (pred_src == i))[0]
            iou_list.append(_iou(pred_idx, s["segments"][i]))
        t_true = int(np.argmax(np.abs(s["contrib"][:, PAST_LEN:]).sum(0)))
        t_pred = int(np.argmax(mag_contrib[b].sum(0)))
        jt_list.append(abs(t_pred - t_true))

    cf_errs = []
    for i in range(n_events):
        pred_cf = counterfactual(out, bd["s0"], i).cpu().numpy()
        true_cf = bd["cf"].cpu().numpy()[:, i, :]
        cf_errs.append(np.sqrt(np.mean((pred_cf - true_cf) ** 2)))

    return {
        "forecast_mae": forecast_mae,
        "jump_time_mae": float(np.mean(jt_list)),
        "attr_f1": float(np.nanmean(f1_list)) if f1_list else 0.0,
        "segment_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "counterfactual_rmse": float(np.mean(cf_errs)),
    }
