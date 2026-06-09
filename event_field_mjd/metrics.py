"""Attribution / count / forecast metrics for the EF-MJD synthetic validation.

The headline attribution metric is **interval-level argmax-match**: for every
sensor interval that truly contained a response, does the model assign the
largest expected response count to the *same* event the ground-truth process
did? This is invariant to magnitude scale (unlike a global Pearson correlation
on per-event totals) and directly measures attribution *resolution* under
overlap. We report it alongside top-2 accuracy, an overlap-only slice
(>=2 co-active events), per-event count recovery, count Poisson-NLL, and simple
non-learned attribution baselines for context.
"""

import torch


def pearson(x, y):
    x = x.reshape(-1).float(); y = y.reshape(-1).float()
    x = x - x.mean(); y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-12)
    return float((x * y).sum() / denom)


def poisson_nll(k, lam):
    """Mean -log Pois(k; lam) over all entries (k real >= 0, lam >= 0)."""
    lam = lam.clamp_min(1e-8)
    return float((lam - k * torch.log(lam) + torch.lgamma(k + 1.0)).mean())


def _argmatch(scores, gt_K_evt, cand, min_cand=1):
    """Interval-level argmax-match for a per-event score tensor.

    scores, gt_K_evt, cand : [B, M, T]   (cand = candidate/active & visible mask)
    Returns (n_correct_top1, n_correct_top2, n_sel, n_correct_top1_overlap, n_sel_overlap).
    Only intervals with a well-defined GT responsible event (max gt count > 0)
    and at least ``min_cand`` candidates are scored.
    """
    neg = torch.finfo(scores.dtype).min
    gt_m = torch.where(cand, gt_K_evt, torch.full_like(gt_K_evt, -1.0))
    sc_m = torch.where(cand, scores, torch.full_like(scores, neg))

    gt_arg = gt_m.argmax(dim=1)                 # [B, T]
    gt_max = gt_m.max(dim=1).values             # [B, T]
    n_cand = cand.sum(dim=1)                     # [B, T]

    mdl_arg = sc_m.argmax(dim=1)                  # [B, T]
    top2 = sc_m.topk(min(2, scores.shape[1]), dim=1).indices   # [B, k, T]
    top2_hit = (top2 == gt_arg.unsqueeze(1)).any(dim=1)        # [B, T]

    sel = (gt_max > 0) & (n_cand >= min_cand)
    sel_ov = sel & (n_cand >= 2)
    top1 = (mdl_arg == gt_arg)
    return (int((top1 & sel).sum()), int((top2_hit & sel).sum()), int(sel.sum()),
            int((top1 & sel_ov).sum()), int(sel_ov.sum()))


def attribution_report(A_hat, gt_K_evt, evt_mask, active, tau, mag):
    """Compute model + baseline argmax-match and per-event count recovery.

    All tensors on CPU. A_hat/gt_K_evt/active: [B,M,T]; evt_mask/tau/mag: [B,M].
    Returns a dict of scalars (counts and rates).
    """
    B, M, T = A_hat.shape
    vis = (evt_mask > 0).unsqueeze(-1)                       # [B,M,1]
    cand = vis & (active > 0)                                # [B,M,T] candidates per interval

    # candidate-aware baseline score tensors [B,M,T]
    recent = tau.unsqueeze(-1).expand(-1, -1, T)             # most-recent event ~ largest tau
    magsc = mag.unsqueeze(-1).expand(-1, -1, T)              # largest observed magnitude
    out = {}
    for name, sc in [("model", A_hat), ("recent", recent), ("magnitude", magsc)]:
        c1, c2, n, c1o, no = _argmatch(sc, gt_K_evt, cand)
        out[f"{name}_top1"] = c1 / max(n, 1)
        out[f"{name}_top2"] = c2 / max(n, 1)
        out[f"{name}_top1_overlap"] = c1o / max(no, 1)
    # random baseline: expected top1 = mean 1/n_cand over scored intervals
    gt_m = torch.where(cand, gt_K_evt, torch.full_like(gt_K_evt, -1.0))
    n_cand = cand.sum(dim=1).float()
    sel = (gt_m.max(dim=1).values > 0) & (n_cand >= 1)
    out["random_top1"] = float((1.0 / n_cand.clamp_min(1))[sel].mean()) if int(sel.sum()) else 0.0
    out["n_scored"] = int(sel.sum())
    out["frac_overlap"] = float((n_cand[sel] >= 2).float().mean()) if int(sel.sum()) else 0.0

    # per-event count recovery (over visible events, all intervals)
    m = vis.expand_as(A_hat)
    out["count_corr"] = pearson(A_hat[m], gt_K_evt[m])
    out["count_mae"] = float((A_hat[m] - gt_K_evt[m]).abs().mean())
    return out
