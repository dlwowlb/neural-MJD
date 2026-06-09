"""Event-conditioned marked MJD synthetic generator.

This is the *data-generating process (DGP)* that matches our model's
assumptions, in the same spirit that Neural-MJD validates itself on data drawn
from a Merton jump-diffusion. Whereas plain MJD only has "how many jumps", here
every latent jump additionally carries a **source mark** -- background (`r=0`)
or "caused by event `e_i`" (`r=i`) -- so we get ground-truth response counts and
per-event signed contributions and can score *attribution recovery*, not just
forecasting.

DGP (one line):

    observed events  ->  lambda_resp^GT, pi_i^GT, magnitude law
                     ->  marked latent jump counter N^E(dt,dy,dr)
                     ->  trajectory X_t = log S_t

Pipeline (matches the spec Steps 1-8):

1. Sample events  e_i = (tau_i, c_i, m_i): time, type, magnitude.
2. Per-event delayed response kernel  g_i(t) = alpha_{c_i} m_i K_{c_i}(t-tau_i)
   on  0 < t-tau_i < W   (K = Gamma pdf -> delayed peak).
3. True response intensity   lambda_resp^GT(t) = lambda_min + sum_{i in A_t} g_i(t).
4. True attribution share     pi_i^GT(t) = g_i(t) / sum_l g_l(t).
5. Background intensity        lambda_bg^GT(t) = lambda_bg0.
6. Sample the marked jump counter on a fine grid: for each fine step draw
   Poisson background / response counts; each response jump's source is drawn
   from pi^GT(t); record (time, mark, logY).
7. Response magnitude  log Y_i ~ N(nu_{c_i} m_i, gamma_resp^2) with type-signed
   nu (meal/stress > 0, insulin/exercise < 0); background  log Y ~ N(0,gamma_bg^2).
8. Integrate  X_{n+1} = X_n + mu^GT dt + sigma^GT sqrt(dt) eps + sum logY.

Ground truth saved per sensor interval I_j: K_bg^GT, K_resp^GT, per-event count
K_{i,j}^GT, and signed contribution R_{i,j}^GT = sum of logY of jumps marked i.
"""

import math
import numpy as np
import torch

# event types and their (intensity scale alpha, magnitude mean coef nu,
# Gamma kernel shape k, Gamma kernel scale theta)
EVENT_TYPES = ["meal", "insulin", "exercise", "stress"]
TYPE_PARAMS = {
    #            alpha   nu      k     theta
    "meal":     (1.0,   +0.16,  2.5,  1.1),   # raises glucose, delayed peak
    "insulin":  (1.0,   -0.16,  2.0,  0.9),   # lowers glucose
    "exercise": (0.9,   -0.13,  2.2,  1.4),   # lowers, slower
    "stress":   (0.8,   +0.10,  1.5,  1.0),   # raises, fast
}


def _gamma_pdf(u, k, theta):
    """Gamma(k, theta) density for u > 0, else 0. Vectorised over u."""
    out = np.zeros_like(u)
    pos = u > 0
    up = u[pos]
    out[pos] = up ** (k - 1.0) * np.exp(-up / theta) / (math.gamma(k) * theta ** k)
    return out


def generate_dataset(
    num_samples=768,
    T=48,                  # number of sensor intervals (T+1 grid points)
    dt=1.0,                # sensor spacing Delta_y
    fine_per_interval=20,  # latent jumps simulated on a finer grid
    n_events_range=(3, 7),
    W=8.0,                 # response window (shared across types)
    lambda_min=0.04,       # baseline response intensity floor
    lambda_bg0=0.05,       # background jump intensity
    gamma_resp=0.05,       # response log-jump magnitude std
    gamma_bg=0.10,         # background log-jump magnitude std
    mr_rate=0.10,          # mean-reversion rate of the drift
    sigma_diff=0.01,       # diffusion sigma (log-space, per sqrt(time))
    mag_scale=1.0,         # global multiplier on response magnitude mean
    max_delay=2.0,         # per-event response onset delay ~ U(0, max_delay)
    personal_scale_sigma=0.2,  # per-subject latent effect scaling (unseen by model)
    event_span_frac=0.80,  # OVERLAP KNOB: events packed into [0.05, 0.05+span]*horizon;
                           # smaller -> events bunch up -> more co-active events / interval
    # --- Synthetic-III misspecification toggles (all default off) ---
    mag_skew=0.0,          # >0: response log-magnitude is right-skewed, not Gaussian
    label_noise_p=0.0,     # prob the model-visible event TYPE is randomised (true effect intact)
    label_missing_p=0.0,   # prob an event is hidden from the model (effect still in trajectory)
    smooth_response=False, # response added as a continuous drift instead of discrete jumps
    baseline_log_range=(math.log(80.0), math.log(120.0)),
    seed=0,
):
    rng = np.random.default_rng(seed)
    n_types = len(EVENT_TYPES)
    M_max = n_events_range[1]
    n_fine = T * fine_per_interval
    dt_fine = dt / fine_per_interval
    grid = np.arange(T + 1) * dt                          # [T+1] sensor times
    t_fine = np.arange(n_fine + 1) * dt_fine              # [n_fine+1]

    X = np.zeros((num_samples, T + 1), dtype=np.float32)
    tau = np.zeros((num_samples, M_max), dtype=np.float32)
    x_feat = np.zeros((num_samples, M_max, n_types + 1), dtype=np.float32)  # onehot(type)+m
    evt_mask = np.zeros((num_samples, M_max), dtype=np.float32)

    gt_R = np.zeros((num_samples, M_max, T), dtype=np.float32)      # signed contribution R_{i,j}
    gt_K_evt = np.zeros((num_samples, M_max, T), dtype=np.float32)  # per-event response count
    gt_K_resp = np.zeros((num_samples, T), dtype=np.float32)        # total response count
    gt_K_bg = np.zeros((num_samples, T), dtype=np.float32)          # background count
    gt_bg_R = np.zeros((num_samples, T), dtype=np.float32)          # signed background contribution
    gt_delay = np.zeros((num_samples, M_max), dtype=np.float32)     # per-event onset delay
    gt_amp = np.zeros((num_samples, M_max), dtype=np.float32)       # per-event signed effect mean (nu_i)
    gt_event_type = -np.ones((num_samples, M_max), dtype=np.int64)  # per-event type index
    gt_baseline = np.zeros((num_samples,), dtype=np.float32)        # per-subject log baseline

    horizon = T * dt
    for b in range(num_samples):
        baseline = rng.uniform(*baseline_log_range)
        gt_baseline[b] = baseline
        # per-subject latent gain that the model never observes -> the event
        # feature (type, magnitude) no longer fully reveals the response.
        personal = float(rng.lognormal(0.0, personal_scale_sigma))
        n_evt = int(rng.integers(n_events_range[0], n_events_range[1] + 1))

        # ---- Step 1: events (time, type, magnitude), off the sensor grid.
        # event_span_frac controls temporal packing (overlap knob).
        span = max(event_span_frac, 1e-3) * horizon
        taus = np.sort(rng.uniform(0.05 * horizon, 0.05 * horizon + span, size=n_evt))
        taus = np.clip(taus + 0.5 * dt * rng.uniform(-0.4, 0.4, size=n_evt), 0.0, horizon - 1e-3)
        type_idx = rng.integers(0, n_types, size=n_evt)
        mags = rng.uniform(0.5, 1.5, size=n_evt)
        delays = rng.uniform(0.0, max_delay, size=n_evt)       # per-event onset delay

        tau[b, :n_evt] = taus
        gt_delay[b, :n_evt] = delays
        gt_event_type[b, :n_evt] = type_idx
        # --- model-visible features, with optional label noise / missingness ---
        for i in range(n_evt):
            if rng.random() < label_missing_p:
                continue                              # event hidden: x_feat=0, evt_mask=0
            vis_type = type_idx[i]
            if rng.random() < label_noise_p:
                vis_type = int(rng.integers(0, n_types))   # corrupt the observed type only
            x_feat[b, i, vis_type] = 1.0              # one-hot type (possibly corrupted)
            x_feat[b, i, n_types] = mags[i]           # magnitude (observed)
            evt_mask[b, i] = 1.0

        # ---- Step 2: per-event response kernels g_i on the fine grid.
        # Response starts only after an onset delay and is hard-windowed to
        # 0 < (t - tau_i - delay_i) < W (explicit attribution window).
        g = np.zeros((n_evt, n_fine + 1), dtype=np.float64)     # [n_evt, n_fine+1]
        nu_i = np.zeros(n_evt)                                  # per-event signed magnitude mean
        for i in range(n_evt):
            ctype = EVENT_TYPES[type_idx[i]]
            alpha, nu, k, theta = TYPE_PARAMS[ctype]
            u = t_fine - taus[i] - delays[i]
            kern = _gamma_pdf(u, k, theta)
            kern[(u <= 0) | (u >= W)] = 0.0                     # window 0<u<W
            g[i] = alpha * mags[i] * kern
            # true signed effect mean: type sign x magnitude x latent personal gain
            nu_i[i] = mag_scale * nu * mags[i] * personal
        gt_amp[b, :n_evt] = nu_i

        # ---- Steps 3-4: response intensity and attribution share on fine grid
        g_sum = g.sum(axis=0)                                   # [n_fine+1]
        lam_resp = lambda_min + g_sum                           # lambda_resp^GT(t)
        # only "on" where at least one event active; floor still allows rare jumps
        active_any = g_sum > 1e-8
        pi_gt = np.zeros_like(g)
        pi_gt[:, active_any] = g[:, active_any] / g_sum[active_any]      # sums to 1 exactly

        # ---- Steps 6-8: simulate the marked jump counter + trajectory
        x = baseline
        X[b, 0] = x
        # accumulators indexed by sensor interval
        for n in range(n_fine):
            t_n = t_fine[n]
            j = min(int(t_n // dt), T - 1)                      # sensor interval index

            # smooth drift (mean reversion) + diffusion
            drift = mr_rate * (baseline - x) * dt_fine
            diff = sigma_diff * math.sqrt(dt_fine) * rng.normal()
            x = x + drift + diff

            # background jumps  r=0
            k_bg = rng.poisson(lambda_bg0 * dt_fine)
            for _ in range(k_bg):
                logY = rng.normal(0.0, gamma_bg)
                x += logY
                gt_K_bg[b, j] += 1
                gt_bg_R[b, j] += logY

            # response: discrete marked jumps (default) or a continuous drift
            # (smooth_response misspecification -- model assumes jumps).
            lam_r = lam_resp[n]
            if smooth_response:
                # add E[response] as a deterministic increment; record continuous
                # contributions / expected counts (real-valued) per event.
                if active_any[n]:
                    for i in range(n_evt):
                        if g[i, n] <= 0:
                            continue
                        contrib = g[i, n] * nu_i[i] * dt_fine
                        x += contrib
                        gt_R[b, i, j] += contrib
                        gt_K_evt[b, i, j] += g[i, n] * dt_fine
                        gt_K_resp[b, j] += g[i, n] * dt_fine
            elif active_any[n] and lam_r > 0:
                k_resp = rng.poisson(lam_r * dt_fine)
                p_src = pi_gt[:, n] / pi_gt[:, n].sum()         # guard fp drift
                for _ in range(k_resp):
                    src = rng.choice(n_evt, p=p_src)            # source ~ pi^GT(t)
                    if mag_skew > 0:
                        e = rng.exponential(1.0) - 1.0          # mean 0, right-skewed
                        logY = nu_i[src] + gamma_resp * e
                    else:
                        logY = rng.normal(nu_i[src], gamma_resp)
                    x += logY
                    gt_K_resp[b, j] += 1
                    gt_K_evt[b, src, j] += 1
                    gt_R[b, src, j] += logY

            # record sensor observation at interval boundaries
            if (n + 1) % fine_per_interval == 0:
                X[b, (n + 1) // fine_per_interval] = x

    out = {
        "X": torch.from_numpy(X),                              # [B, T+1] log-states
        "S": torch.from_numpy(np.exp(X)),                      # [B, T+1] raw states
        "tau": torch.from_numpy(tau),                          # [B, M]
        "x_feat": torch.from_numpy(x_feat),                    # [B, M, n_types+1]
        "evt_mask": torch.from_numpy(evt_mask),                # [B, M]
        "grid": torch.from_numpy(grid.astype(np.float32)),     # [T+1]
        "gt_R": torch.from_numpy(gt_R),                        # [B, M, T] signed contribution
        "gt_K_evt": torch.from_numpy(gt_K_evt),                # [B, M, T] per-event count
        "gt_K_resp": torch.from_numpy(gt_K_resp),              # [B, T] response count
        "gt_K_bg": torch.from_numpy(gt_K_bg),                  # [B, T] background count
        "gt_bg_R": torch.from_numpy(gt_bg_R),                  # [B, T] signed background contribution
        "gt_delay": torch.from_numpy(gt_delay),                # [B, M] per-event onset delay
        "gt_amp": torch.from_numpy(gt_amp),                    # [B, M] per-event signed effect mean
        "gt_event_type": torch.from_numpy(gt_event_type),      # [B, M] type index (-1 = pad)
        "gt_baseline": torch.from_numpy(gt_baseline),          # [B] log baseline
        "meta": {"T": T, "dt": dt, "W": W, "M_max": M_max,
                 "n_types": n_types, "x_feat_dim": n_types + 1,
                 "event_types": EVENT_TYPES, "max_delay": max_delay,
                 "personal_scale_sigma": personal_scale_sigma},
    }
    return out


def train_test_split(data, frac=0.8, seed=0):
    B = data["X"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(B, generator=g)
    n_train = int(frac * B)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]

    def _subset(idx):
        sub = {}
        for k, v in data.items():
            if k in ("meta", "grid"):
                sub[k] = v
            else:
                sub[k] = v[idx]
        return sub

    return _subset(tr_idx), _subset(te_idx)
