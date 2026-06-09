"""End-to-end synthetic validation for the Event-Field Marked MJD model.

Mirrors the Neural-MJD demo flow (generate synthetic data -> build model ->
train on NLL + auxiliary mean loss -> evaluate), and additionally validates the
*attribution* machinery that is the whole point of this formulation:

  1. Optimisation sanity     : NLL decreases on train.
  2. Forecast quality        : one-step mean MAE (raw S space) on held-out data.
  3. Attribution recovery    : correlation between the model's posterior signed
                               log-response R_{i,j} (eq. 66) and the *ground-truth*
                               per-event-per-interval log-contribution.
  4. Sign accuracy           : does the model get the direction (up/down) of each
                               event's dominant effect right?
  5. Background separation   : does posterior expected background count K_bg track
                               where ground-truth background jumps actually fired?

A trivial "shuffle" baseline is reported alongside (3) so the correlation is
interpretable rather than absolute.
"""

import os
import math
import argparse
import numpy as np
import torch

from synthetic import generate_dataset, train_test_split
from model import EventFieldMJD


def move(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def iterate_batches(data, batch_size, shuffle, device, generator=None):
    B = data["X"].shape[0]
    idx = torch.randperm(B, generator=generator) if shuffle else torch.arange(B)
    for s in range(0, B, batch_size):
        sel = idx[s:s + batch_size]
        batch = {
            "X": data["X"][sel],
            "tau": data["tau"][sel],
            "x_feat": data["x_feat"][sel],
            "evt_mask": data["evt_mask"][sel],
            "gt_R": data["gt_R"][sel],
            "gt_K_resp": data["gt_K_resp"][sel],
            "gt_K_bg": data["gt_K_bg"][sel],
            "grid": data["grid"],
        }
        yield move(batch, device)


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-12)
    return float((x * y).sum() / denom)


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    nll_sum, mean_se_sum, n_interval = 0.0, 0.0, 0
    mae_sum, mae_count = 0.0, 0

    R_model, R_gt = [], []          # per active event, summed over its window
    sign_correct, sign_total = 0, 0
    kbg_model, kbg_gt = [], []
    kresp_model, resp_gt = [], []   # response localization per interval

    for batch in iterate_batches(data, batch_size, shuffle=False, device=device):
        loss, cache = model.forward(batch)
        nll_sum += float(cache["nll"].sum())
        mean_se_sum += float(cache["mean_loss"].sum())
        n_interval += cache["nll"].numel()

        # one-step forecast MAE in raw S space (exp of predicted next log-state)
        s_pred = torch.exp(cache["mean_pred"])
        s_true = torch.exp(batch["X"][:, 1:])
        mae_sum += float((s_pred - s_true).abs().sum())
        mae_count += s_true.numel()

        attr = model.attribute(batch)
        R_hat = attr["R_hat"]                       # [B, M, T]
        gt_R = batch["gt_R"]                        # [B, M, T]
        evt_mask = batch["evt_mask"]                # [B, M]

        # aggregate each event's contribution over its window (total signed effect)
        R_hat_tot = R_hat.sum(dim=-1)               # [B, M]
        gt_R_tot = gt_R.sum(dim=-1)                 # [B, M]
        valid = evt_mask > 0
        R_model.append(R_hat_tot[valid].cpu())
        R_gt.append(gt_R_tot[valid].cpu())

        # sign accuracy on events whose true effect is non-trivial
        nontrivial = valid & (gt_R_tot.abs() > 1e-3)
        sign_correct += int((torch.sign(R_hat_tot[nontrivial]) ==
                             torch.sign(gt_R_tot[nontrivial])).sum())
        sign_total += int(nontrivial.sum())

        # background separation: posterior K_bg vs GT background jump count per interval
        kbg_model.append(attr["K_bg"].reshape(-1).cpu())
        kbg_gt.append(batch["gt_K_bg"].reshape(-1).cpu())

        # response localization: posterior K_resp vs GT response jump count per interval
        kresp_model.append(attr["K_resp"].reshape(-1).cpu())
        resp_gt.append(batch["gt_K_resp"].reshape(-1).cpu())

    R_model = torch.cat(R_model)
    R_gt = torch.cat(R_gt)
    kbg_model = torch.cat(kbg_model)
    kbg_gt = torch.cat(kbg_gt)
    kresp_model = torch.cat(kresp_model)
    resp_gt = torch.cat(resp_gt)

    # shuffle baseline for the attribution correlation
    perm = torch.randperm(R_gt.numel())
    corr = pearson(R_model, R_gt)
    corr_shuffle = pearson(R_model, R_gt[perm])

    return {
        "nll": nll_sum / n_interval,
        "mean_mse": mean_se_sum / n_interval,
        "mae_raw": mae_sum / mae_count,
        "attr_corr": corr,
        "attr_corr_shuffle": corr_shuffle,
        "sign_acc": sign_correct / max(sign_total, 1),
        "bg_corr": pearson(kbg_model, kbg_gt),
        "resp_loc_corr": pearson(kresp_model, resp_gt),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa_trunc", type=int, default=5)
    ap.add_argument("--w_mean", type=float, default=1.0)
    ap.add_argument("--w_rho", type=float, default=1e-3)
    ap.add_argument("--w_ent", type=float, default=1e-3)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- data
    data = generate_dataset(num_samples=args.num_samples, seed=args.seed)
    train_data, test_data = train_test_split(data, frac=0.8, seed=args.seed)
    W = data["meta"]["W"]
    dt = data["meta"]["dt"]
    print(f"Train sequences: {train_data['X'].shape[0]}, Test: {test_data['X'].shape[0]}, "
          f"T={data['meta']['T']}, W={W}")

    # ----- model
    x_feat_dim = data["meta"]["x_feat_dim"]
    model = EventFieldMJD(x_feat_dim=x_feat_dim, kappa_trunc=args.kappa_trunc, W=W, dt=dt,
                          w_mean=args.w_mean, w_rho=args.w_rho, w_ent=args.w_ent).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    gen = torch.Generator().manual_seed(args.seed)

    nll_curve = []
    print("\nTraining...")
    for epoch in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in iterate_batches(train_data, args.batch_size, True, device, gen):
            loss, _ = model.forward(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        sched.step()
        avg = ep_loss / nb
        nll_curve.append(avg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tr = evaluate(model, train_data, device, args.batch_size)
            print(f"Epoch {epoch+1:3d}/{args.epochs} | loss {avg:8.4f} | "
                  f"train NLL {tr['nll']:7.4f} | attr_corr {tr['attr_corr']:.3f} | "
                  f"sign_acc {tr['sign_acc']:.3f}")

    # ----- final evaluation
    print("\n=== Final evaluation ===")
    for name, ds in [("train", train_data), ("test", test_data)]:
        m = evaluate(model, ds, device, args.batch_size)
        print(f"\n[{name}]")
        print(f"  NLL (per interval)        : {m['nll']:.4f}")
        print(f"  mean-loss MSE (log space) : {m['mean_mse']:.5f}")
        print(f"  one-step MAE (raw S)      : {m['mae_raw']:.4f}")
        print(f"  attribution corr (R vs GT): {m['attr_corr']:.3f}   "
              f"(shuffle baseline {m['attr_corr_shuffle']:.3f})")
        print(f"  event sign accuracy       : {m['sign_acc']:.3f}")
        print(f"  response loc. corr (Kresp): {m['resp_loc_corr']:.3f}")
        print(f"  background corr (Kbg vs GT): {m['bg_corr']:.3f}")

    # ----- plots
    if not args.no_plot:
        try:
            make_plots(model, train_data, test_data, nll_curve, device, args.batch_size)
        except Exception as e:
            print(f"(plotting skipped: {e})")


@torch.no_grad()
def make_plots(model, train_data, test_data, nll_curve, device, batch_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(outdir, exist_ok=True)
    model.eval()

    # (1) NLL training curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nll_curve)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (NLL + w*mean)")
    ax.set_title("Optimisation curve")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "01_loss_curve.png"), dpi=120)
    plt.close(fig)

    # gather one test batch
    batch = next(iterate_batches(test_data, batch_size, False, device))
    _, cache = model.forward(batch)
    attr = model.attribute(batch)

    # (2) example trajectory + events + predicted mean
    b = 0
    grid = batch["grid"].cpu().numpy()
    X = batch["X"][b].cpu().numpy()
    mean_pred = cache["mean_pred"][b].cpu().numpy()       # predicts X[1:]
    taus = batch["tau"][b].cpu().numpy()
    gt_sign = batch["gt_R"][b].sum(-1).cpu().numpy()     # signed true effect per event
    emask = batch["evt_mask"][b].cpu().numpy() > 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grid, np.exp(X), "k.-", label="true S", ms=3)
    ax.plot(grid[1:], np.exp(mean_pred), "C0--", label="pred mean S")
    for i in np.where(emask)[0]:
        c = "C2" if gt_sign[i] >= 0 else "C3"
        ax.axvline(taus[i], color=c, alpha=0.4, lw=1.5)
    ax.set_xlabel("time"); ax.set_ylabel("S (raw)")
    ax.set_title("Trajectory, events (green=up, red=down), one-step mean")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "02_trajectory.png"), dpi=120)
    plt.close(fig)

    # (3) scatter: model R vs ground-truth R (per event, summed over window)
    R_hat = attr["R_hat"].sum(-1)[batch["evt_mask"] > 0].cpu().numpy()
    R_gt = batch["gt_R"].sum(-1)[batch["evt_mask"] > 0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(R_gt, R_hat, s=14, alpha=0.6)
    lim = max(np.abs(R_gt).max(), np.abs(R_hat).max()) * 1.1 + 1e-6
    ax.plot([-lim, lim], [-lim, lim], "k:", lw=1)
    ax.set_xlabel("ground-truth signed log-response")
    ax.set_ylabel("model R_{i} (posterior)")
    ax.set_title("Attribution recovery")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "03_attribution_scatter.png"), dpi=120)
    plt.close(fig)

    print(f"\nPlots written to {outdir}/")


if __name__ == "__main__":
    main()
