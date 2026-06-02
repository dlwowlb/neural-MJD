"""
Run the overlapping event-response baseline experiment.

Trains two baselines on the synthetic benchmark and reports forecasting + event-
response attribution metrics across a sweep of event-overlap levels:

    1. neural_mjd       Neural MJD with no event context (latent jump only)
    2. neural_mjd_ctx   Neural MJD with the observed events fed into context C

("Ours" -- event-specific response intensity decomposition -- is intentionally
left out here; this script establishes the baseline behaviour.)

Usage:
    python -m experiments.event_response.run            # full run
    python -m experiments.event_response.run --quick    # fast smoke run
"""

import os
import json
import argparse
import numpy as np
import torch

from .synthetic import make_dataset, make_train_dataset
from . import model_io as mio
from . import evaluate as ev
from . import figure as fig


# gap=0 is excluded: there the two events share a time and collapse into a single
# source (attribution is then trivially one-class). gap in {1,2,4,6} spans heavy
# overlap (g=1) to fully separated responses (g=6, since RESPONSE_DUR=6).
GAPS = [1, 2, 4, 6]
BASELINES = [("neural_mjd", False), ("neural_mjd_ctx", True)]


def run(quick=False, seed=0, out_dir=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if quick:
        n_train, n_test, epochs, feat = 384, 96, 20, 32
    else:
        n_train, n_test, epochs, feat = 1536, 256, 80, 64

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    # mixed single/double-event training set; fixed-overlap two-event test sets
    train_seqs = make_train_dataset(n=n_train, seed=seed + 1)
    test_sets = {g: make_dataset(n=n_test, gap=g, seed=seed + 100 + g, opposite_type=True)
                 for g in GAPS}

    results = {}          # results[name][gap] = metrics dict
    models = {}
    for name, use_ctx in BASELINES:
        print(f"\n=== Training baseline: {name} (event context = {use_ctx}) ===")
        in_dim = 2 + (2 if use_ctx else 0)
        model = mio.build_model(in_seq_dim=in_dim, feature_dims=feat,
                                num_layers=2, num_heads=4, seed=seed,
                                w_cond_mean_loss=30.0)
        model.to(device)
        mio.train(model, train_seqs, use_ctx, device, epochs=epochs, lr=2e-3)
        models[(name, use_ctx)] = model

        torch.save({"state_dict": model.state_dict(), "in_dim": in_dim,
                    "feat": feat}, os.path.join(out_dir, f"{name}.pth"))

        results[name] = {}
        for g in GAPS:
            m = ev.evaluate_model(model, test_sets[g], use_ctx, device)
            results[name][g] = m
            print(f"  gap={g}: forecastMAE={m['forecast_mae']:.3f} "
                  f"jumpMAE={m['jump_time_mae']:.2f} F1={m['attr_f1']:.3f} "
                  f"IoU={m['segment_iou']:.3f} cfRMSE={m['counterfactual_rmse']:.3f}")

    _print_summary(results)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_csv(results, os.path.join(out_dir, "metrics.csv"))

    # qualitative 3-panel figure on one representative overlapping example
    # (meal->rise + insulin->fall, matching the opposite-type benchmark)
    example = fig.pick_example(gap=3, seed=7)
    fig.three_panel(example, models[("neural_mjd", False)],
                    models[("neural_mjd_ctx", True)], device,
                    os.path.join(out_dir, "three_panel.png"))
    fig.overlap_robustness(results, GAPS, os.path.join(out_dir, "overlap_robustness.png"))
    print(f"\nArtifacts written to {out_dir}/")
    return results


def _avg_over_gaps(results, name, key):
    return float(np.mean([results[name][g][key] for g in GAPS]))


def _print_summary(results):
    keys = [("forecast_mae", "Forecast MAE (raw)", "lower"),
            ("jump_time_mae", "Jump-time MAE (steps)", "lower"),
            ("attr_f1", "Event attribution F1", "higher"),
            ("segment_iou", "Segment IoU", "higher"),
            ("counterfactual_rmse", "Counterfactual RMSE", "lower")]
    names = [n for n, _ in BASELINES]
    print("\n" + "=" * 72)
    print("SUMMARY (averaged over overlap gaps {})".format(GAPS))
    print("=" * 72)
    print(f"{'Metric':<28}" + "".join(f"{n:>22}" for n in names) + "   better")
    for key, label, better in keys:
        row = f"{label:<28}"
        for n in names:
            row += f"{_avg_over_gaps(results, n, key):>22.3f}"
        row += f"   {better}"
        print(row)
    print("=" * 72)
    print("Overlap robustness (event attribution F1 as gap -> 0):")
    for n in names:
        vals = "  ".join(f"g{g}:{results[n][g]['attr_f1']:.2f}" for g in GAPS)
        print(f"  {n:<18} {vals}")


def _write_csv(results, path):
    keys = ["forecast_mae", "jump_time_mae", "attr_f1", "segment_iou", "counterfactual_rmse"]
    with open(path, "w") as f:
        f.write("baseline,gap," + ",".join(keys) + "\n")
        for name in results:
            for g in results[name]:
                row = [name, str(g)] + [f"{results[name][g][k]:.4f}" for k in keys]
                f.write(",".join(row) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(quick=args.quick, seed=args.seed)
