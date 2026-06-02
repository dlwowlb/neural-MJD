"""
Regenerate the qualitative figures from saved baseline checkpoints, without
retraining. Run `run.py` first to produce results/neural_mjd*.pth.

    python -m experiments.event_response.make_figures
"""

import os
import json
import torch

from . import model_io as mio
from . import ours as ours_mod
from . import figure as fig
from .run import GAPS


def _load_baseline(out_dir, name):
    ckpt = torch.load(os.path.join(out_dir, f"{name}.pth"), map_location="cpu")
    model = mio.build_model(in_seq_dim=ckpt["in_dim"], feature_dims=ckpt["feat"],
                            num_layers=2, num_heads=4)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _load_ours(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ours.pth"), map_location="cpu")
    model = ours_mod.EventMarkedMJD(hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    device = torch.device("cpu")
    plain = _load_baseline(out_dir, "neural_mjd")
    ctx = _load_baseline(out_dir, "neural_mjd_ctx")
    ours_model = _load_ours(out_dir)

    example = fig.pick_example(gap=3, seed=7)
    fig.four_panel(example, plain, ctx, ours_model, device,
                   os.path.join(out_dir, "four_panel.png"))

    with open(os.path.join(out_dir, "metrics.json")) as f:
        results = json.load(f)
    results = {k: {int(g): v for g, v in d.items()} for k, d in results.items()}
    fig.overlap_robustness(results, GAPS, os.path.join(out_dir, "overlap_robustness.png"))
    print(f"Figures regenerated in {out_dir}/")


if __name__ == "__main__":
    main()
