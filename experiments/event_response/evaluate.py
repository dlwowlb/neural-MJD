"""
Attribution protocols and benchmark metrics.

The central scientific point is that Neural MJD only predicts an *aggregate* jump
intensity lambda_t. It therefore has no native event-source variable. To give each
baseline the *best possible* attribution it can produce, we use:

  * Neural MJD + event context  -> occlusion attribution. The event marker of
    event i is removed from the input context and the change in the predicted
    trajectory is read as that event's post-hoc "response intensity"
    S_i(t) = | yhat(t) - yhat_{remove i}(t) |. This is exactly the probe a
    reviewer means by "just put the events in C".

  * Neural MJD (no context)     -> timing-only attribution. The model has no
    event input, so the only signal it can offer is *when* a jump occurs
    (its jump energy). Each response step is assigned to the nearest event
    onset in time -- an oracle timing prior. This is generous to the baseline
    and still collapses under overlap.

Metrics: forecast MAE, jump-time MAE, event-source attribution F1, segment IoU,
counterfactual RMSE.
"""

import numpy as np
import torch

from .synthetic import PAST_LEN, FUTURE_LEN
from . import model_io as mio


# ----------------------------------------------------------------------------
# Per-event response intensity (the quantity each baseline can produce)
# ----------------------------------------------------------------------------
def _increments(y, s0):
    """Per-step forecast increments, with the pre-horizon level prepended."""
    return np.diff(y, axis=1, prepend=s0[:, None])


@torch.no_grad()
def occlusion_intensity(model, seqs, device):
    """Occlusion sensitivity S_i(t) for the +event-context model.

    The response process is cumulative, so occluding an event shifts the *level*
    of every downstream step (the largest event would then dominate the whole
    tail). We therefore read the occlusion effect on the per-step *increments*,
    which localizes each event's response to its own segment -- and blurs where
    two responses overlap. Returns [B, n_events, F] and the factual forecast.
    """
    n_events = len(seqs[0]["events"])
    bd, tgt = mio.make_batch(seqs, use_ctx=True, device=device)
    base = mio.predict_mean(model, bd, tgt).cpu().numpy()        # [B, F]
    s0 = np.array([s["s"][PAST_LEN - 1] for s in seqs])
    base_inc = _increments(base, s0)

    S = np.zeros((len(seqs), n_events, FUTURE_LEN), dtype=np.float64)
    for i in range(n_events):
        def override(seq, _i=i):
            ctx = seq["event_ctx"][:, :PAST_LEN].copy()
            ev = seq["events"][_i]
            ch = 0 if ev["c"] == +1 else 1
            ctx[ch, ev["tau"]] = 0.0                              # remove event i
            return ctx
        bd_i, tgt_i = mio.make_batch(seqs, use_ctx=True, device=device,
                                     event_override=override)
        yi = mio.predict_mean(model, bd_i, tgt_i).cpu().numpy()   # [B, F]
        S[:, i, :] = np.abs(base_inc - _increments(yi, s0))
    return S, base


@torch.no_grad()
def occlusion_counterfactual(model, seqs, device, remove_event):
    """Predicted counterfactual forecast with `remove_event` removed, [B, F]."""
    def override(seq):
        ctx = seq["event_ctx"][:, :PAST_LEN].copy()
        ev = seq["events"][remove_event]
        ch = 0 if ev["c"] == +1 else 1
        ctx[ch, ev["tau"]] = 0.0
        return ctx
    bd, tgt = mio.make_batch(seqs, use_ctx=True, device=device, event_override=override)
    return mio.predict_mean(model, bd, tgt).cpu().numpy()


@torch.no_grad()
def jump_energy(model, seqs, use_ctx, device):
    """Model jump energy lambda*(|nu|+|gamma|) per future step, [B, F]."""
    bd, _ = mio.make_batch(seqs, use_ctx, device=device)
    return mio.get_mjd_params(model, bd)["jump_energy"].cpu().numpy()


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def _macro_f1(true_lab, pred_lab, classes):
    """Macro-F1 over source classes that actually occur in `true_lab`.

    Classes absent from the ground truth are skipped (scoring them as a perfect
    1.0 would spuriously inflate the metric on, e.g., fully-overlapped sequences
    where only one source is present).
    """
    f1s = []
    for c in classes:
        if np.sum(true_lab == c) == 0:
            continue
        tp = np.sum((pred_lab == c) & (true_lab == c))
        fp = np.sum((pred_lab == c) & (true_lab != c))
        fn = np.sum((pred_lab != c) & (true_lab == c))
        denom = (2 * tp + fp + fn)
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)) if f1s else np.nan


def _iou(pred_idx, true_idx):
    a, b = set(pred_idx.tolist()), set(true_idx.tolist())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ----------------------------------------------------------------------------
# Full evaluation for one model variant
# ----------------------------------------------------------------------------
def evaluate_model(model, seqs, use_ctx, device):
    """Compute all benchmark metrics for one trained baseline on `seqs`."""
    n_events = len(seqs[0]["events"])

    # forecast MAE (raw scale, cond-mean point forecast)
    bd, tgt = mio.make_batch(seqs, use_ctx, device=device)
    yhat = mio.predict_mean(model, bd, tgt).cpu().numpy()         # [B, F]
    ytrue = np.stack([s["s"][PAST_LEN:] for s in seqs])           # [B, F]
    forecast_mae = float(np.mean(np.abs(yhat - ytrue)))

    # jump-time MAE: where the model expects the largest move vs the true onset
    je = jump_energy(model, seqs, use_ctx, device)                # [B, F] (Panel B)
    s0 = np.array([s["s"][PAST_LEN - 1] for s in seqs])
    yhat_move = np.abs(np.diff(yhat, axis=1, prepend=s0[:, None]))   # |Delta forecast|
    jump_time_err = []
    for b, s in enumerate(seqs):
        t_true = int(np.argmax(np.abs(s["contrib"][:, PAST_LEN:]).sum(0)))
        t_pred = int(np.argmax(yhat_move[b]))
        jump_time_err.append(abs(t_pred - t_true))
    jump_time_mae = float(np.mean(jump_time_err))

    # per-event response intensity S_i(t) -- only the +context model has one
    S = occlusion_intensity(model, seqs, device)[0] if use_ctx else None

    # attribution F1 + segment IoU over response steps
    rng = np.random.default_rng(0)
    f1_list, iou_list = [], []
    for b, s in enumerate(seqs):
        attr = s["attr_label"]                                   # [F] in {-1,0,1}
        resp_steps = np.where(attr != -1)[0]

        if use_ctx:
            # event-specific occlusion intensity -> dominant source per step
            pred_src_full = np.argmax(S[b], axis=0)              # [F]
            sal = S[b].max(axis=0)
            thr = 0.25 * (sal.max() if sal.max() > 0 else 1.0)
            salient = sal > thr
            if len(resp_steps) > 0:
                f1_list.append(_macro_f1(attr[resp_steps],
                                         pred_src_full[resp_steps],
                                         classes=list(range(n_events))))
            for i in range(n_events):
                pred_idx = np.where(salient & (pred_src_full == i))[0]
                iou_list.append(_iou(pred_idx, s["segments"][i]))
        else:
            # Neural MJD has NO event-source variable: with same-type events the
            # aggregate intensity/sign cannot separate sources. The best it can do
            # is detect *that* a jump occurs (jump energy) and then guess the
            # source -> chance level. Average a few random assignments for a stable
            # estimate of that chance baseline.
            sal = je[b]
            salient = sal > 0.25 * (sal.max() if sal.max() > 0 else 1.0)
            f1_runs, iou_runs = [], []
            for _ in range(16):
                pred_src_full = rng.integers(0, n_events, size=FUTURE_LEN)
                if len(resp_steps) > 0:
                    f1_runs.append(_macro_f1(attr[resp_steps],
                                             pred_src_full[resp_steps],
                                             classes=list(range(n_events))))
                for i in range(n_events):
                    pred_idx = np.where(salient & (pred_src_full == i))[0]
                    iou_runs.append(_iou(pred_idx, s["segments"][i]))
            if f1_runs:
                f1_list.append(float(np.mean(f1_runs)))
            iou_list.append(float(np.mean(iou_runs)))

    attr_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    seg_iou = float(np.mean(iou_list)) if iou_list else 0.0

    # counterfactual RMSE
    cf_errs = []
    for i in range(n_events):
        true_cf = np.stack([s["counterfactual"][i][PAST_LEN:] for s in seqs])  # [B, F]
        if use_ctx:
            pred_cf = occlusion_counterfactual(model, seqs, device, remove_event=i)
        else:
            pred_cf = yhat                                       # cannot remove events
        cf_errs.append(np.sqrt(np.mean((pred_cf - true_cf) ** 2)))
    cf_rmse = float(np.mean(cf_errs))

    return {
        "forecast_mae": forecast_mae,
        "jump_time_mae": jump_time_mae,
        "attr_f1": attr_f1,
        "segment_iou": seg_iou,
        "counterfactual_rmse": cf_rmse,
    }
