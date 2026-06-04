"""Event-Field Marked Merton Jump Diffusion (EF-MJD).

A faithful, self-contained PyTorch implementation of the equation flow described
in the task spec. Section/equation numbers in comments refer to that spec.

The model maps a *dense* sensor log-trajectory ``X`` plus *irregular* events
``(tau_i, x_i)`` to a per-interval, mark-collapsed marked-jump-diffusion
likelihood, and recovers per-event attribution as a posterior quantity.

Design notes / approximations (kept explicit so the math stays auditable):

* History ``h_{t_j}`` is produced by a **causal** GRU over the trajectory and
  per-interval event aggregates, so it never sees ``X_{t_{j+1}}`` when scoring
  interval ``j`` (eq. 68 -- no information leakage).
* The response field ``Z`` (eqs. 8-10) is integrated on the sensor grid with an
  Euler step for the continuous part ``F_theta`` and additive event jumps
  ``U_theta``. ``Z_{t_j}`` denotes the field at the *start* of interval ``j``
  (predictable context ``C_j``, eq. 68).
* Intensities/magnitudes are treated as piecewise-constant on each interval
  (eq. 37, 51), so integrated intensities are ``Lambda = lambda * Delta``
  (eqs. 33-34, 42-43).
* The likelihood is the **collapsed, truncated** mixture over background/response
  counts (eqs. 53-59); per-event identity is restored afterwards via the
  posterior multinomial split (eqs. 46-48, 60-66).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.SiLU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


def poisson_logpmf(k, lam):
    """log Pois(k; lam), correct in the lam -> 0 limit.

    For lam == 0: returns 0 when k == 0 and -inf otherwise.
    ``k`` is a float tensor of non-negative integers.
    """
    lam_safe = lam.clamp_min(1e-12)
    out = k * torch.log(lam_safe) - lam - torch.lgamma(k + 1.0)
    zero_lam = lam <= 0
    if zero_lam.any():
        neg_inf = torch.full_like(out, float("-inf"))
        out = torch.where(zero_lam, torch.where(k == 0, torch.zeros_like(out), neg_inf), out)
    return out


def gaussian_logpdf(x, mean, var):
    var = var.clamp_min(1e-8)
    return -0.5 * (math.log(2 * math.pi) + torch.log(var) + (x - mean) ** 2 / var)


class EventFieldMJD(nn.Module):
    def __init__(
        self,
        x_feat_dim=1,
        d_h=48,            # history hidden size
        d_z=24,            # response-field size
        d_evt=24,          # event-trace size
        kappa_trunc=4,     # jump-count truncation (eq. 57)
        W=8.0,             # attribution window
        dt=1.0,
        # parameter bounds (mirrors Neural-MJD clamping for stability)
        bound_mu=2.0,
        bound_sigma=0.5,
        bound_nu=0.6,
        bound_gamma=0.5,
        bound_lambda=4.0,
        w_mean=1.0,        # omega in eq. 69
    ):
        super().__init__()
        self.d_h, self.d_z, self.d_evt = d_h, d_z, d_evt
        self.kappa_trunc = kappa_trunc
        self.W = W
        self.dt = dt
        self.bound_mu = bound_mu
        self.bound_sigma = bound_sigma
        self.bound_nu = bound_nu
        self.bound_gamma = bound_gamma
        self.bound_lambda = bound_lambda
        self.w_mean = w_mean

        # --- Encoder for history h_t (eq. 7), causal GRU over [x_value, event_agg]
        self.enc_in = 1 + x_feat_dim          # current log-state + aggregated event feature
        self.encoder = nn.GRU(self.enc_in, d_h, batch_first=True)

        # --- Response field Z (eqs. 8-10)
        self.F_theta = mlp([d_z + d_h + 1, 2 * d_z, d_z])          # continuous evolution dZ/dt
        self.U_theta = mlp([d_z + x_feat_dim + d_h, 2 * d_z, d_z])  # event update jump
        self.z0 = nn.Parameter(torch.zeros(d_z))

        # --- Intensities (eqs. 11-12)
        self.g_bg = mlp([d_h, d_h, 1])                 # background intensity logit
        self.g_resp = mlp([d_z + d_h, d_h, 1])         # shared response intensity logit

        # --- Event trace + attribution score (eqs. 14-15)
        self.psi = mlp([x_feat_dim + 1 + d_z + d_h, d_evt, d_evt])     # zeta_i(t)
        self.score = mlp([d_evt + d_z + d_h, d_evt, 1])                # s_theta
        # event-specific signed response magnitude rho_{i,j} from the trace.
        # Used in the auxiliary mean head (eq. 70) so that the attribution
        # softmax pi (eqs. 15/47) and trace psi (eq. 14) receive a gradient:
        # the *collapsed* NLL (eqs. 53-58) marginalises event identity and gives
        # pi no signal on its own. See README "identifiability" note.
        self.rho_head = mlp([d_evt, d_evt, 1])

        # --- Jump magnitude heads (eqs. 23-26): each outputs (nu, gamma_raw)
        self.m_bg = mlp([d_h, d_h, 2])
        self.m_resp = mlp([d_z + d_h, d_h, 2])

        # --- Drift / diffusion (eq. 28): outputs (mu, sigma_raw)
        self.p_theta = mlp([d_h + d_z, d_h, 2])

        # Precompute truncated count grid (eq. 57): {(k_bg,k_resp): sum<=kappa}
        pairs = [(a, b) for a in range(kappa_trunc + 1)
                 for b in range(kappa_trunc + 1) if a + b <= kappa_trunc]
        kb = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
        kr = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
        self.register_buffer("k_bg_grid", kb)        # [P]
        self.register_buffer("k_resp_grid", kr)      # [P]

    # ------------------------------------------------------------------ #
    #  Encoders / latent fields
    # ------------------------------------------------------------------ #
    def encode_history(self, X, tau, x_feat, evt_mask, grid):
        """h_{t_j} for j=0..T (causal). Returns [B, T+1, d_h].

        IMPORTANT design choice: this trajectory history is **event-free**. It
        feeds the smooth drift/diffusion (eq. 28) and the background channel
        (eqs. 12/23-24). Event information enters the model *only* through the
        response field ``Z`` (eqs. 8-10) and the attribution heads. If the drift
        could see events it would simply absorb every event-driven increment,
        leaving the response/attribution machinery with no work to do (and no
        gradient). Separating the two is exactly the endogenous-vs-exogenous
        split the formulation is built around.
        """
        # zero out the event aggregate -> GRU sees trajectory only (the
        # encoder still has the input slot so the architecture is unchanged).
        evt_agg = torch.zeros(X.shape[0], X.shape[1], self.enc_in - 1, device=X.device)
        x_in = (X - X[:, :1]).unsqueeze(-1)                                  # [B, T+1, 1]
        enc_input = torch.cat([x_in, evt_agg], dim=-1)
        h, _ = self.encoder(enc_input)                                      # [B, T+1, d_h]
        return h

    def rollout_field(self, h_grid, tau, x_feat, evt_mask, grid):
        """Integrate Z over the grid. Returns Z_grid [B, T+1, d_z] where
        Z_grid[:, j] == Z_{t_j} (field at the start of interval j)."""
        B, Tp1, _ = h_grid.shape
        T = Tp1 - 1
        floor = torch.floor(tau / self.dt + 1e-6).long().clamp(0, T - 1)    # interval containing event
        Z = self.z0.unsqueeze(0).expand(B, -1).contiguous()
        Z_list = []
        for j in range(Tp1):
            Z_list.append(Z)
            if j == T:
                break
            h_j = h_grid[:, j]
            t_j = grid[j].expand(B, 1)
            # continuous evolution F_theta (eq. 8), Euler step over dt
            dZ = self.F_theta(torch.cat([Z, h_j, t_j], dim=-1))
            Z = Z + dZ * self.dt
            # event jumps U_theta (eq. 9) for events whose interval == j
            in_j = ((floor == j) & (evt_mask > 0)).float()                  # [B, M]
            if in_j.any():
                # U depends on Z at jump time; approximate with post-evolution Z.
                Zr = Z.unsqueeze(1).expand(-1, tau.shape[1], -1)
                u_in = torch.cat([Zr, x_feat, h_j.unsqueeze(1).expand(-1, tau.shape[1], -1)], dim=-1)
                U = self.U_theta(u_in) * in_j.unsqueeze(-1)                 # [B, M, d_z]
                Z = Z + U.sum(dim=1)
        return torch.stack(Z_list, dim=1)                                   # [B, T+1, d_z]

    # ------------------------------------------------------------------ #
    #  Per-interval parameters
    # ------------------------------------------------------------------ #
    def interval_params(self, h_grid, Z_grid):
        """Compute the piecewise-constant interval parameters for j=0..T-1.

        Returns a dict of [B, T] tensors (and intensities)."""
        h = h_grid[:, :-1]      # context at interval start (predictable)
        Z = Z_grid[:, :-1]
        hz = torch.cat([h, Z], dim=-1)

        mu_raw, sig_raw = self.p_theta(hz).chunk(2, dim=-1)                 # eq. 28
        mu = self.bound_mu * torch.tanh(mu_raw.squeeze(-1))
        sigma = self.bound_sigma * torch.sigmoid(sig_raw.squeeze(-1)) + 1e-3

        nu_bg, gam_bg_raw = self.m_bg(h).chunk(2, dim=-1)                   # eqs. 23-24
        nu_bg = self.bound_nu * torch.tanh(nu_bg.squeeze(-1))
        gam_bg = self.bound_gamma * torch.sigmoid(gam_bg_raw.squeeze(-1)) + 1e-3

        nu_rp, gam_rp_raw = self.m_resp(torch.cat([Z, h], dim=-1)).chunk(2, dim=-1)  # eqs. 25-26
        nu_rp = self.bound_nu * torch.tanh(nu_rp.squeeze(-1))
        gam_rp = self.bound_gamma * torch.sigmoid(gam_rp_raw.squeeze(-1)) + 1e-3

        lam_bg = F.softplus(self.g_bg(h).squeeze(-1)).clamp(max=self.bound_lambda)    # eq. 12
        lam_rp = F.softplus(self.g_resp(hz).squeeze(-1)).clamp(max=self.bound_lambda) # eq. 11

        return {
            "mu": mu, "sigma": sigma,
            "nu_bg": nu_bg, "gam_bg": gam_bg,
            "nu_rp": nu_rp, "gam_rp": gam_rp,
            "lam_bg": lam_bg, "lam_rp": lam_rp,
        }

    def attribution_shares(self, h_grid, Z_grid, tau, x_feat, evt_mask, grid):
        """Interval-level normalised attribution shares ``pi_bar[b,j,m]`` (eq. 47)
        and the per-interval active mask (eqs. 13/49).

        Returns: pi_bar [B, T, M], active [B, T, M] (float), has_active [B, T],
                 rho [B, T, M] (event-specific signed log-response magnitude).
        """
        B, Tp1, _ = h_grid.shape
        T = Tp1 - 1
        M = tau.shape[1]

        # event-time context Z_{tau^-}, h_{tau^-} (interval-start of the event's interval)
        floor = torch.floor(tau / self.dt + 1e-6).long().clamp(0, T - 1)    # [B, M]
        Z_evt = torch.gather(Z_grid, 1, floor.unsqueeze(-1).expand(-1, -1, self.d_z))  # [B,M,d_z]
        h_evt = torch.gather(h_grid, 1, floor.unsqueeze(-1).expand(-1, -1, self.d_h))  # [B,M,d_h]

        t_start = grid[:-1]                       # [T] interval start times t_j
        t_end = grid[1:]                          # [T] interval end times t_{j+1}
        age = t_start.view(1, T, 1) - tau.view(B, 1, M)        # [B,T,M] = t_j - tau_i

        # active set A_j (eq. 49): window (tau, tau+W] overlaps interval (t_j, t_{j+1}]
        overlap = (tau.view(B, 1, M) < t_end.view(1, T, 1)) & \
                  ((t_start.view(1, T, 1) - tau.view(B, 1, M)) < self.W)
        active = (overlap & (evt_mask.view(B, 1, M) > 0)).float()           # [B,T,M]

        # event trace zeta_i(t) (eq. 14)
        x_e = x_feat.view(B, 1, M, -1).expand(-1, T, -1, -1)
        Z_e = Z_evt.view(B, 1, M, -1).expand(-1, T, -1, -1)
        h_e = h_evt.view(B, 1, M, -1).expand(-1, T, -1, -1)
        zeta = self.psi(torch.cat([x_e, age.unsqueeze(-1), Z_e, h_e], dim=-1))   # [B,T,M,d_evt]

        # score s_theta(zeta, Z_t, h_t) (eq. 15)
        Z_t = Z_grid[:, :-1].view(B, T, 1, self.d_z).expand(-1, -1, M, -1)
        h_t = h_grid[:, :-1].view(B, T, 1, self.d_h).expand(-1, -1, M, -1)
        s = self.score(torch.cat([zeta, Z_t, h_t], dim=-1)).squeeze(-1)     # [B,T,M]

        # masked softmax over active events (eqs. 15-16, 47)
        neg_inf = torch.finfo(s.dtype).min
        s_masked = torch.where(active > 0, s, torch.full_like(s, neg_inf))
        pi_bar = torch.softmax(s_masked, dim=-1)
        pi_bar = pi_bar * active                          # zero-out inactive
        has_active = active.sum(dim=-1) > 0               # [B,T]
        # if no active events, softmax over all -inf is uniform; force to 0
        pi_bar = torch.where(has_active.unsqueeze(-1), pi_bar, torch.zeros_like(pi_bar))

        # event-specific signed response magnitude rho_{i,j} from the trace
        rho = self.bound_nu * torch.tanh(self.rho_head(zeta).squeeze(-1))    # [B,T,M]
        return pi_bar, active, has_active.float(), rho

    # ------------------------------------------------------------------ #
    #  Likelihood (collapsed, truncated) + auxiliary mean
    # ------------------------------------------------------------------ #
    def _collapsed_terms(self, X, params, has_active):
        """Build per-(interval, count-pair) Gaussian params and Poisson weights.

        Returns:
          log_w   [B, T, P]  log mixture weight (eq. 60, sans normalisation)
          a       [B, T, P]  conditional mean (eq. 54)
          b2      [B, T, P]  conditional variance (eq. 55)
          x_next  [B, T]     target X_{t_{j+1}}
        """
        B = X.shape[0]
        T = params["mu"].shape[1]
        P = self.k_bg_grid.shape[0]
        dt = self.dt

        mu, sigma = params["mu"], params["sigma"]
        nu_bg, gam_bg = params["nu_bg"], params["gam_bg"]
        nu_rp, gam_rp = params["nu_rp"], params["gam_rp"]
        lam_bg, lam_rp = params["lam_bg"], params["lam_rp"]

        # response intensity only counts when active events exist (eq. 43)
        lam_rp = lam_rp * has_active
        Lam_bg = lam_bg * dt                              # eq. 42
        Lam_rp = lam_rp * dt                              # eq. 43

        # compensator kappa_t (eq. 29)
        kappa = lam_bg * (torch.exp(nu_bg + 0.5 * gam_bg ** 2) - 1.0) \
            + lam_rp * (torch.exp(nu_rp + 0.5 * gam_rp ** 2) - 1.0)

        drift = (mu - kappa - 0.5 * sigma ** 2) * dt      # smooth part of eq. 54

        kb = self.k_bg_grid.view(1, 1, P)
        kr = self.k_resp_grid.view(1, 1, P)

        x_cur = X[:, :-1]                                 # [B, T]
        x_next = X[:, 1:]                                 # [B, T]

        a = x_cur.unsqueeze(-1) + drift.unsqueeze(-1) \
            + kb * nu_bg.unsqueeze(-1) + kr * nu_rp.unsqueeze(-1)             # eq. 54
        b2 = (sigma ** 2 * dt).unsqueeze(-1) \
            + kb * (gam_bg ** 2).unsqueeze(-1) + kr * (gam_rp ** 2).unsqueeze(-1)  # eq. 55

        log_pois = poisson_logpmf(kb.expand(B, T, P), Lam_bg.unsqueeze(-1)) \
            + poisson_logpmf(kr.expand(B, T, P), Lam_rp.unsqueeze(-1))        # eqs. 56/60
        log_gauss = gaussian_logpdf(x_next.unsqueeze(-1), a, b2)
        log_w = log_pois + log_gauss                                         # eq. 60

        return log_w, a, b2, x_next, {"Lam_bg": Lam_bg, "Lam_rp": Lam_rp,
                                      "kappa": kappa, "drift": drift,
                                      "nu_rp": nu_rp}

    def forward(self, batch):
        """Compute NLL, auxiliary mean loss, and cache quantities for attribution."""
        X = batch["X"]
        tau, x_feat, evt_mask, grid = batch["tau"], batch["x_feat"], batch["evt_mask"], batch["grid"]

        h_grid = self.encode_history(X, tau, x_feat, evt_mask, grid)
        Z_grid = self.rollout_field(h_grid, tau, x_feat, evt_mask, grid)
        params = self.interval_params(h_grid, Z_grid)
        pi_bar, active, has_active, rho = self.attribution_shares(
            h_grid, Z_grid, tau, x_feat, evt_mask, grid)

        log_w, a, b2, x_next, extra = self._collapsed_terms(X, params, has_active)

        # truncated collapsed log-likelihood (eq. 58)
        log_p = torch.logsumexp(log_w, dim=-1)            # [B, T]
        nll = -log_p

        # auxiliary mean prediction (eqs. 69-70): expected one-step log-increment.
        # The response term is the sum of *attributed* expected responses, each
        # carrying the event-specific magnitude rho_{i,j}:
        #   resp_incr_j = sum_i [Lambda_resp,j * pi_bar_{i,j}] * rho_{i,j}   (eqs. 46/66)
        # Routing the response through pi_bar and rho gives the attribution
        # softmax and event trace a real gradient (the collapsed NLL alone does
        # not -- it marginalises event identity).
        x_cur = X[:, :-1]
        Lam_attr = extra["Lam_rp"].unsqueeze(-1) * pi_bar          # [B, T, M] (eq. 46)
        resp_incr = (Lam_attr * rho).sum(dim=-1)                   # [B, T]
        bg_incr = extra["Lam_bg"] * params["nu_bg"]
        mean_pred = x_cur + extra["drift"] + bg_incr + resp_incr
        mean_loss = (x_next - mean_pred) ** 2

        loss = nll.mean() + self.w_mean * mean_loss.mean()

        cache = {
            "log_w": log_w, "a": a, "b2": b2, "x_next": x_next,
            "params": params, "pi_bar": pi_bar, "active": active,
            "has_active": has_active, "mean_pred": mean_pred, "rho": rho,
            "nll": nll, "mean_loss": mean_loss, "extra": extra,
            "h_grid": h_grid, "Z_grid": Z_grid,
        }
        return loss, cache

    # ------------------------------------------------------------------ #
    #  Posterior event attribution (eqs. 60-66)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def attribute(self, batch):
        """Return per-event-per-interval posterior attribution.

        R_hat   [B, M, T]  signed log-response contribution R_{i,j} (eq. 66)
        A_hat   [B, M, T]  attribution score A_{i,j} (eq. 64)
        P_resp  [B, M, T]  prob event contributed >=1 response jump (eq. 65)
        K_resp  [B, T]     posterior expected response count (eq. 63)
        K_bg    [B, T]     posterior expected background count (eq. 62)
        """
        _, cache = self.forward(batch)
        log_w, pi_bar = cache["log_w"], cache["pi_bar"]
        rho = cache["rho"]                                # [B, T, M] event-specific magnitude

        # posterior weights q (eq. 61)
        q = torch.softmax(log_w, dim=-1)                  # [B, T, P]
        kb = self.k_bg_grid.view(1, 1, -1)
        kr = self.k_resp_grid.view(1, 1, -1)
        K_bg = (q * kb).sum(-1)                            # eq. 62
        K_resp = (q * kr).sum(-1)                          # eq. 63

        # event-level attribution (eq. 64): A_{i,j} = K_resp_hat * pi_bar_{i,j}
        A_hat = K_resp.unsqueeze(1) * pi_bar.transpose(1, 2)   # [B, M, T]
        # signed log-response (eq. 66): R_{i,j} = A_{i,j} * rho_{i,j}
        # (event-specific magnitude rho replaces the shared nu_resp so that the
        #  recovered sign/scale is per-event).
        R_hat = A_hat * rho.transpose(1, 2)                    # [B, M, T]

        # prob event contributed at least one response jump (eq. 65)
        # P = 1 - sum_pairs q * (1 - pi_bar)^{k_resp}
        pi_bm = pi_bar.transpose(1, 2).unsqueeze(-1)          # [B, M, T, 1]
        kr_e = kr.view(1, 1, 1, -1)                            # [1,1,1,P]
        q_e = q.unsqueeze(1)                                   # [B, 1, T, P]
        no_hit = (q_e * (1.0 - pi_bm).clamp(min=0) ** kr_e).sum(-1)   # [B, M, T]
        P_resp = 1.0 - no_hit

        return {"R_hat": R_hat, "A_hat": A_hat, "P_resp": P_resp,
                "K_resp": K_resp, "K_bg": K_bg, "pi_bar": pi_bar}
