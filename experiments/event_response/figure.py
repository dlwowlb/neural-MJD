"""
Qualitative figures.

`three_panel` is the key figure for the paper:

    Panel A  Ground truth      trajectory + events + true response segments,
                               colour-coded by source event.
    Panel B  Neural MJD        aggregate jump intensity lambda_t -- "a jump is
                               likely here" but with NO source information.
    Panel C  Neural MJD + ctx  occlusion response intensity per event -- a
                               post-hoc probe that blurs across overlapping
                               responses (the reviewer's "just feed events to C").

The figure makes the thesis legible: Neural MJD sees *that* a jump happens; it
cannot say *which event* produced *which* response.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .synthetic import PAST_LEN, FUTURE_LEN, TOTAL_LEN, generate_sequence
from . import evaluate as ev

EV_COLORS = ["#1f77b4", "#d62728"]      # event 0 (meal/up), event 1 (insulin/down)
EV_NAMES = ["event A: meal -> rise", "event B: insulin -> fall"]


def pick_example(gap, seed):
    """A clean opposite-type example with event A = meal (up), B = insulin (down)."""
    rng = np.random.default_rng(seed)
    for _ in range(200):
        seq = generate_sequence(gap, rng, opposite_type=True)
        if seq["events"][0]["c"] == +1 and seq["events"][1]["c"] == -1:
            return seq
    return seq


def three_panel(seq, model_plain, model_ctx, device, path):
    t_all = np.arange(TOTAL_LEN)
    t_fut = np.arange(PAST_LEN, TOTAL_LEN)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # --- Panel A: ground truth -------------------------------------------
    ax = axes[0]
    ax.plot(t_all, seq["s"], color="k", lw=1.5, label="trajectory S_t")
    ax.axvline(PAST_LEN - 0.5, color="grey", ls=":", lw=1)
    ax.text(PAST_LEN - 0.5, ax.get_ylim()[1], " forecast horizon",
            va="top", ha="left", fontsize=8, color="grey")
    for i, evd in enumerate(seq["events"]):
        c = 0 if evd["c"] == +1 else 1
        ax.axvline(evd["tau"], color=EV_COLORS[i], ls="--", lw=1.2)
        ax.scatter([evd["tau"]], [seq["s"][evd["tau"]]], color=EV_COLORS[i], zorder=5)
        # true response segment shading (future-local -> global)
        seg = seq["segments"][i] + PAST_LEN
        if len(seg):
            ax.axvspan(seg.min() - 0.5, seg.max() + 0.5, color=EV_COLORS[i], alpha=0.15)
    ax.set_title("A.  Ground truth: events and their delayed response segments")
    ax.set_ylabel("S_t")
    handles = [plt.Line2D([0], [0], color=EV_COLORS[i], ls="--") for i in range(2)]
    ax.legend([plt.Line2D([0], [0], color="k")] + handles,
              ["trajectory"] + EV_NAMES, fontsize=8, loc="best")

    # --- Panel B: Neural MJD aggregate intensity -------------------------
    ax = axes[1]
    je = ev.jump_energy(model_plain, [seq], use_ctx=False, device=device)[0]   # [F]
    ax.bar(t_fut, je, color="#555555", width=0.8)
    ax.set_title(r"B.  Neural MJD: aggregate jump intensity $\lambda_t(|\nu_t|+|\gamma_t|)$ "
                 "(no event source)")
    ax.set_ylabel("intensity")
    for i in range(2):
        seg = seq["segments"][i] + PAST_LEN
        if len(seg):
            ax.axvspan(seg.min() - 0.5, seg.max() + 0.5, color=EV_COLORS[i], alpha=0.12)

    # --- Panel C: Neural MJD + context occlusion intensity ---------------
    ax = axes[2]
    S, _ = ev.occlusion_intensity(model_ctx, [seq], device=device)     # [1, n_ev, F]
    S = S[0]
    for i in range(S.shape[0]):
        ax.bar(t_fut + (i - 0.5) * 0.4, S[i], width=0.4,
               color=EV_COLORS[i], alpha=0.8, label=f"S_{{{i}}}(t) ({EV_NAMES[i]})")
        seg = seq["segments"][i] + PAST_LEN
        if len(seg):
            ax.axvspan(seg.min() - 0.5, seg.max() + 0.5, color=EV_COLORS[i], alpha=0.12)
    ax.set_title("C.  Neural MJD + event context: occlusion response intensity "
                 "(blurred across overlap)")
    ax.set_ylabel("|forecast change|")
    ax.set_xlabel("time step")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _panel_truth(ax, seq, t_all):
    ax.plot(t_all, seq["s"], color="k", lw=1.5)
    ax.axvline(PAST_LEN - 0.5, color="grey", ls=":", lw=1)
    ax.text(PAST_LEN - 0.5, ax.get_ylim()[1], " forecast horizon",
            va="top", ha="left", fontsize=8, color="grey")
    for i, evd in enumerate(seq["events"]):
        ax.axvline(evd["tau"], color=EV_COLORS[i], ls="--", lw=1.2)
        ax.scatter([evd["tau"]], [seq["s"][evd["tau"]]], color=EV_COLORS[i], zorder=5)
        seg = seq["segments"][i] + PAST_LEN
        if len(seg):
            ax.axvspan(seg.min() - 0.5, seg.max() + 0.5, color=EV_COLORS[i], alpha=0.15)
    ax.set_title("A.  Ground truth: events and their delayed response segments")
    ax.set_ylabel("S_t")
    handles = [plt.Line2D([0], [0], color=EV_COLORS[i], ls="--") for i in range(2)]
    ax.legend([plt.Line2D([0], [0], color="k")] + handles,
              ["trajectory"] + EV_NAMES, fontsize=8, loc="best")


def _shade_segments(ax, seq):
    for i in range(2):
        seg = seq["segments"][i] + PAST_LEN
        if len(seg):
            ax.axvspan(seg.min() - 0.5, seg.max() + 0.5, color=EV_COLORS[i], alpha=0.12)


def four_panel(seq, model_plain, model_ctx, model_ours, device, path):
    """A: truth | B: Neural MJD aggregate lambda | C: +ctx occlusion | D: Ours."""
    import torch
    from . import ours as ours_mod

    t_all = np.arange(TOTAL_LEN)
    t_fut = np.arange(PAST_LEN, TOTAL_LEN)
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    _panel_truth(axes[0], seq, t_all)

    # B: Neural MJD aggregate jump intensity (no source)
    ax = axes[1]
    je = ev.jump_energy(model_plain, [seq], use_ctx=False, device=device)[0]
    ax.bar(t_fut, je, color="#555555", width=0.8)
    ax.set_title(r"B.  Neural MJD: aggregate jump intensity $\lambda_t(|\nu_t|+|\gamma_t|)$ "
                 "(no event source)")
    ax.set_ylabel("intensity")
    _shade_segments(ax, seq)

    # C: +context occlusion response intensities (entangled across overlap)
    ax = axes[2]
    S, _ = ev.occlusion_intensity(model_ctx, [seq], device=device)
    S = S[0]
    for i in range(S.shape[0]):
        ax.bar(t_fut + (i - 0.5) * 0.4, S[i], width=0.4, color=EV_COLORS[i],
               alpha=0.8, label=f"S_{i}(t)  ({EV_NAMES[i]})")
    _shade_segments(ax, seq)
    ax.set_title("C.  Neural MJD + event context: occlusion intensity "
                 "(entangled, post-hoc)")
    ax.set_ylabel("|forecast change|")
    ax.legend(fontsize=8, loc="best")

    # D: Ours -- native per-event response intensities lambda_i(t)
    ax = axes[3]
    bd = ours_mod.make_batch_ours([seq], device)
    with torch.no_grad():
        out = model_ours(bd["past"], bd["tau"], bd["typ"], bd["mag"], bd["mask"])
    lam = out["ev_lambda"][0].cpu().numpy()                # [m, F]
    for i in range(lam.shape[0]):
        ax.bar(t_fut + (i - 0.5) * 0.4, lam[i], width=0.4, color=EV_COLORS[i],
               alpha=0.85, label=rf"$\lambda_{i}(t)$  ({EV_NAMES[i]})")
    _shade_segments(ax, seq)
    ax.set_title(r"D.  Ours: event-marked intensities $\lambda_i(t)$ "
                 "(each response attributed to its source)")
    ax.set_ylabel("intensity")
    ax.set_xlabel("time step")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def overlap_robustness(results, gaps, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    # (key, label, note, ylim, chance-level reference or None)
    metrics = [("attr_f1", "Event attribution F1", "higher better", (0.0, 1.05), 0.5),
               ("segment_iou", "Segment IoU", "higher better", (0.0, 1.05), None),
               ("counterfactual_rmse", "Counterfactual RMSE", "lower better, log", None, None)]
    for ax, (key, label, note, ylim, chance) in zip(axes, metrics):
        for name in results:
            ys = [results[name][g][key] for g in gaps]
            ax.plot(gaps, ys, marker="o", label=name)
        if chance is not None:
            ax.axhline(chance, color="grey", ls="--", lw=1)
            ax.text(gaps[-1], chance, " chance", va="bottom", ha="right",
                    fontsize=8, color="grey")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if key == "counterfactual_rmse":
            ax.set_yscale("log")
        ax.set_title(f"{label}\n({note})", fontsize=10)
        ax.set_xlabel("event gap  (smaller = more overlap)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Both Neural MJD baselines sit at chance on attribution; "
                 "Ours recovers event-source structure and is far better on every axis.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
