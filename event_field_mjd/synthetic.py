"""Synthetic data generator for the Event-Field Marked MJD model.

This mirrors the spirit of the Neural-MJD synthetic demo (``demo_notebook.ipynb``):
we fabricate data with a *known* generative process so that we can check whether
the model recovers the structure it is supposed to learn.

The generative story (CGM-glucose flavoured, but generic):

* A dense sensor grid ``t_j = j * dt`` carries a log-state ``X_t = log S_t`` that
  mean-reverts to a per-sequence baseline and is perturbed by small diffusion
  noise (eq. 30: ``dX_t = (mu - kappa - 0.5 sigma^2) dt + sigma dW`` without the
  jump term, plus the additive contributions below).
* Irregular events ``e_i = (tau_i, x_i)`` arrive at times that are *not* aligned
  with the sensor grid (eq. 1-4). Each event injects an additive **log-response**
  that rises to a persistent signed level over the attribution window ``W`` (this
  plays the role of the shared response field driving response jumps, eqs. 8-10,
  22, 25-26); the trajectory later relaxes to baseline through mean reversion,
  not through the event -- matching a jump-diffusion's inductive bias.
* A handful of rare **background jumps** unexplained by any event are injected
  (eq. 21, 23) so the model has to separate "event-attributed" from "background".

Crucially we record, per sensor interval ``I_j`` and per event ``i``, the exact
log-space contribution ``gt_R[i, j]`` that event ``i`` made to that interval.
This is the ground-truth analogue of the model's signed log-response
``R_{i,j} = A_{i,j} * nu_resp`` (eq. 66) and lets us measure attribution recovery.
"""

import math
import numpy as np
import torch


def _bump(s, tau_kernel):
    """Causal, monotonic, *persistent* response kernel.

    g(s) = 1 - exp(-s / tau) for s > 0, else 0  (saturates at 1).

    The cumulative log-response of an event rises from 0 to its full amplitude
    over a few intervals and then *stays* there; the trajectory returns to
    baseline through the global mean-reversion drift, not through the event.
    This matches a jump-diffusion's inductive bias (the event = a signed jump,
    the relaxation = endogenous drift) and -- unlike a rise-then-decay bump --
    makes each event's net signed contribution well-defined and non-degenerate.
    """
    out = np.zeros_like(s)
    pos = s > 0
    out[pos] = 1.0 - np.exp(-s[pos] / tau_kernel)
    return out


def generate_dataset(
    num_samples=512,
    T=48,                # number of sensor intervals (T+1 grid points)
    dt=1.0,              # sensor spacing Delta_y
    n_events_range=(3, 7),
    W=8.0,               # attribution window
    tau_kernel=1.5,      # response rise time-constant (time to ~63% of amplitude)
    resp_scale=0.25,     # max |log-response| amplitude per unit feature
    mean_revert=0.15,    # mean-reversion strength (relaxes the level back to baseline)
    diffusion=0.02,      # diffusion sigma (log-space, per sqrt(dt))
    bg_jump_prob=0.01,   # per-interval probability of a background jump
    bg_jump_scale=0.15,  # background jump log-magnitude std
    baseline_log_range=(math.log(80.0), math.log(120.0)),
    seed=0,
):
    """Return a dict of padded tensors describing ``num_samples`` sequences.

    Event features ``x_i`` are scalar signed magnitudes in [-1, 1]; positive
    means an up-response (e.g. a meal raising glucose), negative a down-response
    (e.g. exercise lowering it). This lets us check that the model recovers both
    the magnitude *and the sign* of each event's effect.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(T + 1) * dt                       # [T+1]
    M_max = n_events_range[1]

    X = np.zeros((num_samples, T + 1), dtype=np.float32)
    tau = np.zeros((num_samples, M_max), dtype=np.float32)
    x_feat = np.zeros((num_samples, M_max, 1), dtype=np.float32)
    evt_mask = np.zeros((num_samples, M_max), dtype=np.float32)
    gt_R = np.zeros((num_samples, M_max, T), dtype=np.float32)   # per-event per-interval log-contribution
    gt_bg = np.zeros((num_samples, T), dtype=np.float32)         # background log-contribution per interval

    total_horizon = T * dt
    for b in range(num_samples):
        baseline = rng.uniform(*baseline_log_range)
        n_evt = rng.integers(n_events_range[0], n_events_range[1] + 1)

        # Irregular event times deliberately off-grid (eq. 4: tau_i not in T).
        taus = np.sort(rng.uniform(0.05 * total_horizon, 0.9 * total_horizon, size=n_evt))
        taus = taus + 0.5 * dt * rng.uniform(-0.4, 0.4, size=n_evt)   # jitter off grid
        amps = rng.uniform(-1.0, 1.0, size=n_evt) * resp_scale
        # mark feature is the *signed magnitude* (what the model observes as x_i)
        feats = (amps / resp_scale).astype(np.float32)

        tau[b, :n_evt] = taus
        x_feat[b, :n_evt, 0] = feats
        evt_mask[b, :n_evt] = 1.0

        # Per-event cumulative log-response evaluated on the grid.
        # response_i(t) = amp_i * bump(t - tau_i); contribution to interval j is the
        # difference of the cumulative response across the interval endpoints.
        resp_cum = np.zeros((n_evt, T + 1), dtype=np.float32)
        for i in range(n_evt):
            resp_cum[i] = amps[i] * _bump(grid - taus[i], tau_kernel)
        gt_R[b, :n_evt, :] = resp_cum[:, 1:] - resp_cum[:, :-1]      # [n_evt, T]

        # Background jumps: rare, event-independent.
        bg = np.zeros(T, dtype=np.float32)
        fire = rng.random(T) < bg_jump_prob
        bg[fire] = rng.normal(0.0, bg_jump_scale, size=fire.sum()).astype(np.float32)
        gt_bg[b] = bg

        # Roll out the log-trajectory.
        x = baseline
        X[b, 0] = x
        for j in range(T):
            drift = mean_revert * (baseline - x) * dt
            diff = diffusion * math.sqrt(dt) * rng.normal()
            resp = float(gt_R[b, :n_evt, j].sum())
            x = x + drift + diff + resp + bg[j]
            X[b, j + 1] = x

    out = {
        "X": torch.from_numpy(X),                       # [B, T+1] log-states
        "S": torch.from_numpy(np.exp(X)),               # [B, T+1] raw states
        "tau": torch.from_numpy(tau),                   # [B, M]
        "x_feat": torch.from_numpy(x_feat),             # [B, M, 1]
        "evt_mask": torch.from_numpy(evt_mask),         # [B, M]
        "grid": torch.from_numpy(grid.astype(np.float32)),  # [T+1]
        "gt_R": torch.from_numpy(gt_R),                 # [B, M, T] ground-truth signed log-contribution
        "gt_bg": torch.from_numpy(gt_bg),               # [B, T]
        "meta": {"T": T, "dt": dt, "W": W, "M_max": M_max,
                 "tau_kernel": tau_kernel, "resp_scale": resp_scale},
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
            if k == "meta":
                sub[k] = v
            elif k == "grid":
                sub[k] = v
            else:
                sub[k] = v[idx]
        return sub

    return _subset(tr_idx), _subset(te_idx)
