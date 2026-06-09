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
            "gt_K_evt": data["gt_K_evt"][sel],
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
def multi_horizon_mae(forecaster, data, device, batch_size, horizons, start_frac=0.4):
    """Autoregressive multi-step forecast MAE (raw S) at the given horizons.

    ``forecaster`` exposes ``rollout_forecast(batch, start_j, horizon)``.
    """
    Tp1 = data["X"].shape[1]
    start_j = int(start_frac * (Tp1 - 1))
    Hmax = min(max(horizons), Tp1 - 1 - start_j)
    err = {h: 0.0 for h in horizons}
    cnt = 0
    for batch in iterate_batches(data, batch_size, shuffle=False, device=device):
        preds = forecaster.rollout_forecast(batch, start_j, Hmax)          # [B, Hmax]
        true = batch["X"][:, start_j + 1: start_j + 1 + Hmax]
        ae = (torch.exp(preds) - torch.exp(true)).abs()                    # [B, Hmax]
        cnt += ae.shape[0]
        for h in horizons:
            if h <= Hmax:
                err[h] += float(ae[:, h - 1].sum())
    return {h: err[h] / max(cnt, 1) for h in horizons if h <= Hmax}


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    import metrics as M
    model.eval()
    nll_sum, mean_se_sum, n_interval = 0.0, 0.0, 0
    mae_sum, mae_count = 0.0, 0
    sign_correct, sign_total = 0, 0

    R_model, R_gt = [], []
    kbg_model, kbg_gt, kresp_model, resp_gt = [], [], [], []
    A_all, gtKevt_all, active_all, mask_all, tau_all, mag_all = [], [], [], [], [], []
    pnll_resp_sum, pnll_bg_sum, pnll_n = 0.0, 0.0, 0
    R_mae_sum, R_mae_n = 0.0, 0

    n_types = data["meta"]["n_types"]
    for batch in iterate_batches(data, batch_size, shuffle=False, device=device):
        _, cache = model.forward(batch)
        nll_sum += float(cache["nll"].sum())
        mean_se_sum += float(cache["mean_loss"].sum())
        n_interval += cache["nll"].numel()

        s_pred = torch.exp(cache["mean_pred"])
        s_true = torch.exp(batch["X"][:, 1:])
        mae_sum += float((s_pred - s_true).abs().sum())
        mae_count += s_true.numel()

        attr = model.attribute(batch)
        R_hat, gt_R, evt_mask = attr["R_hat"], batch["gt_R"], batch["evt_mask"]

        R_hat_tot, gt_R_tot = R_hat.sum(-1), gt_R.sum(-1)
        valid = evt_mask > 0
        R_model.append(R_hat_tot[valid].cpu()); R_gt.append(gt_R_tot[valid].cpu())
        nontrivial = valid & (gt_R_tot.abs() > 1e-3)
        sign_correct += int((torch.sign(R_hat_tot[nontrivial]) ==
                             torch.sign(gt_R_tot[nontrivial])).sum())
        sign_total += int(nontrivial.sum())
        # signed-contribution MAE on non-trivial events
        R_mae_sum += float((R_hat_tot[nontrivial] - gt_R_tot[nontrivial]).abs().sum())
        R_mae_n += int(nontrivial.sum())

        kbg_model.append(attr["K_bg"].reshape(-1).cpu()); kbg_gt.append(batch["gt_K_bg"].reshape(-1).cpu())
        kresp_model.append(attr["K_resp"].reshape(-1).cpu()); resp_gt.append(batch["gt_K_resp"].reshape(-1).cpu())

        # count Poisson-NLL of GT counts under the model's integrated intensities
        pnll_resp_sum += M.poisson_nll(batch["gt_K_resp"].cpu(), attr["Lam_resp"].cpu()) * batch["gt_K_resp"].numel()
        pnll_bg_sum += M.poisson_nll(batch["gt_K_bg"].cpu(), attr["Lam_bg"].cpu()) * batch["gt_K_bg"].numel()
        pnll_n += batch["gt_K_resp"].numel()

        # gather for interval-level argmax-match
        A_all.append(attr["A_hat"].cpu()); gtKevt_all.append(batch["gt_K_evt"].cpu())
        active_all.append(attr["active"].cpu()); mask_all.append(evt_mask.cpu())
        tau_all.append(batch["tau"].cpu()); mag_all.append(batch["x_feat"][..., n_types].cpu())

    R_model, R_gt = torch.cat(R_model), torch.cat(R_gt)
    perm = torch.randperm(R_gt.numel())

    att = M.attribution_report(torch.cat(A_all), torch.cat(gtKevt_all), torch.cat(mask_all),
                               torch.cat(active_all), torch.cat(tau_all), torch.cat(mag_all))

    out = {
        "nll": nll_sum / n_interval,
        "mae_raw": mae_sum / mae_count,
        "attr_corr": pearson(R_model, R_gt),
        "attr_corr_shuffle": pearson(R_model, R_gt[perm]),
        "sign_acc": sign_correct / max(sign_total, 1),
        "R_mae": R_mae_sum / max(R_mae_n, 1),
        "bg_corr": pearson(torch.cat(kbg_model), torch.cat(kbg_gt)),
        "resp_loc_corr": pearson(torch.cat(kresp_model), torch.cat(resp_gt)),
        "pnll_resp": pnll_resp_sum / max(pnll_n, 1),
        "pnll_bg": pnll_bg_sum / max(pnll_n, 1),
    }
    out.update(att)
    return out


def train_efmjd(args, train_data, W, dt, x_feat_dim, device, epochs, log=False):
    """Train an EventFieldMJD on ``train_data``; returns (model, loss_curve)."""
    model = EventFieldMJD(x_feat_dim=x_feat_dim, kappa_trunc=args.kappa_trunc, W=W, dt=dt,
                          w_mean=args.w_mean, w_rho=args.w_rho, w_ent=args.w_ent,
                          couple_attr=args.couple_attr,
                          endo_baseline=args.endo_baseline).to(device)
    if log:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\nTraining...")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    gen = torch.Generator().manual_seed(args.seed)
    curve = []
    for epoch in range(epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in iterate_batches(train_data, args.batch_size, True, device, gen):
            loss, _ = model.forward(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1
        sched.step(); curve.append(ep_loss / nb)
        if log and ((epoch + 1) % 20 == 0 or epoch == 0):
            tr = evaluate(model, train_data, device, args.batch_size)
            print(f"Epoch {epoch+1:3d}/{epochs} | loss {curve[-1]:8.4f} | "
                  f"argmatch top1 {tr['model_top1']:.3f} | sign {tr['sign_acc']:.3f}")
    return model, curve


def run_overlap_sweep(args, gen_data, device, horizons):
    """Synthetic-II: vary event packing and track interval argmax-match.

    The headline robustness plot: does attribution resolution hold as more
    events become co-active per interval (and how do baselines fall off)?
    """
    spans = [0.80, 0.55, 0.35, 0.20]
    rows = []
    for span in spans:
        data = gen_data(args.seed, span=span)
        tr, te = train_test_split(data, frac=0.8, seed=args.seed)
        W, dt = data["meta"]["W"], data["meta"]["dt"]
        xfd = data["meta"]["x_feat_dim"]
        model, _ = train_efmjd(args, tr, W, dt, xfd, device, args.sweep_epochs, log=False)
        m = evaluate(model, te, device, args.batch_size)
        rows.append((span, m["frac_overlap"], m["model_top1"], m["model_top1_overlap"],
                     m["random_top1"], m["recent_top1"], m["magnitude_top1"], m["attr_corr"]))
        print(f"span={span:.2f} overlap_frac={m['frac_overlap']:.2f} | "
              f"top1={m['model_top1']:.3f} top1_ov={m['model_top1_overlap']:.3f} | "
              f"rand={m['random_top1']:.3f} recent={m['recent_top1']:.3f} mag={m['magnitude_top1']:.3f} | "
              f"corr={m['attr_corr']:.3f}")
    if not args.no_plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            outdir = os.path.join(os.path.dirname(__file__), "outputs"); os.makedirs(outdir, exist_ok=True)
            ov = [r[1] for r in rows]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(ov, [r[2] for r in rows], "o-", label="EF-MJD top1")
            ax.plot(ov, [r[3] for r in rows], "s--", label="EF-MJD top1 (overlap only)")
            ax.plot(ov, [r[4] for r in rows], ":", label="random")
            ax.plot(ov, [r[5] for r in rows], ":", label="most-recent")
            ax.plot(ov, [r[6] for r in rows], ":", label="largest-magnitude")
            ax.set_xlabel("fraction of scored intervals with >=2 co-active events")
            ax.set_ylabel("interval argmax-match accuracy")
            ax.set_title("Attribution resolution vs overlap (Synthetic-II)")
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(outdir, "04_overlap_sweep.png"), dpi=120); plt.close(fig)
            print(f"\nSweep figure written to {outdir}/04_overlap_sweep.png")
        except Exception as e:
            print(f"(sweep plot skipped: {e})")


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
    ap.add_argument("--couple_attr", action="store_true",
                    help="couple attribution (pi,rho) into the NLL response magnitude "
                         "(identifiable but deviates from spec eq.26/52); default off")
    ap.add_argument("--endo_baseline", action="store_true",
                    help="factorise X=B+R and condition the smooth drift on the "
                         "event-free baseline B (stops drift from absorbing events); "
                         "default off")
    # --- DGP knobs (overlap + Synthetic-III misspecification) ---
    ap.add_argument("--event_span_frac", type=float, default=0.80)  # smaller = more overlap
    ap.add_argument("--max_delay", type=float, default=2.0)
    ap.add_argument("--personal_scale_sigma", type=float, default=0.2)
    ap.add_argument("--mag_skew", type=float, default=0.0)
    ap.add_argument("--label_noise_p", type=float, default=0.0)
    ap.add_argument("--label_missing_p", type=float, default=0.0)
    ap.add_argument("--smooth_response", action="store_true")
    ap.add_argument("--misspec", action="store_true",
                    help="preset: skewed magnitude + label noise + missing labels")
    # --- evaluation extras ---
    ap.add_argument("--horizons", type=str, default="1,4,8,12")
    ap.add_argument("--no_baseline", action="store_true")
    ap.add_argument("--no_multi_horizon", action="store_true")
    ap.add_argument("--overlap_sweep", action="store_true",
                    help="train+eval across event_span_frac levels and plot argmax-match")
    ap.add_argument("--sweep_epochs", type=int, default=120)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    if args.misspec:
        args.mag_skew = max(args.mag_skew, 1.0)
        args.label_noise_p = max(args.label_noise_p, 0.15)
        args.label_missing_p = max(args.label_missing_p, 0.15)

    horizons = [int(h) for h in args.horizons.split(",") if h]

    def gen_data(seed, span=None):
        return generate_dataset(
            num_samples=args.num_samples, seed=seed,
            event_span_frac=span if span is not None else args.event_span_frac,
            max_delay=args.max_delay, personal_scale_sigma=args.personal_scale_sigma,
            mag_skew=args.mag_skew, label_noise_p=args.label_noise_p,
            label_missing_p=args.label_missing_p, smooth_response=args.smooth_response)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- overlap sweep mode (Synthetic-II): train+eval across packing levels
    if args.overlap_sweep:
        run_overlap_sweep(args, gen_data, device, horizons)
        return

    # ----- data
    data = gen_data(args.seed)
    train_data, test_data = train_test_split(data, frac=0.8, seed=args.seed)
    W, dt = data["meta"]["W"], data["meta"]["dt"]
    x_feat_dim = data["meta"]["x_feat_dim"]
    print(f"Train sequences: {train_data['X'].shape[0]}, Test: {test_data['X'].shape[0]}, "
          f"T={data['meta']['T']}, W={W}  | misspec(skew={args.mag_skew},"
          f" noise={args.label_noise_p}, miss={args.label_missing_p}, smooth={args.smooth_response})")

    model, nll_curve = train_efmjd(args, train_data, W, dt, x_feat_dim, device, args.epochs, log=True)

    # ----- final evaluation
    print("\n=== Final evaluation ===")
    for name, ds in [("train", train_data), ("test", test_data)]:
        m = evaluate(model, ds, device, args.batch_size)
        print(f"\n[{name}]")
        print(f"  NLL (per interval)         : {m['nll']:.4f}")
        print(f"  one-step MAE (raw S)       : {m['mae_raw']:.4f}")
        print(f"  -- attribution (headline: interval argmax-match) --")
        print(f"  argmax-match top1          : {m['model_top1']:.3f}  "
              f"(random {m['random_top1']:.3f}, recent {m['recent_top1']:.3f}, mag {m['magnitude_top1']:.3f})")
        print(f"  argmax-match top2          : {m['model_top2']:.3f}")
        print(f"  argmax-match top1 (overlap): {m['model_top1_overlap']:.3f}  "
              f"[{m['frac_overlap']*100:.0f}% of {m['n_scored']} scored intervals]")
        print(f"  per-event count corr / MAE : {m['count_corr']:.3f} / {m['count_mae']:.3f}")
        print(f"  signed-R: corr {m['attr_corr']:.3f} (shuffle {m['attr_corr_shuffle']:.3f}),"
              f" sign-acc {m['sign_acc']:.3f}, MAE {m['R_mae']:.4f}")
        print(f"  count Poisson-NLL resp/bg  : {m['pnll_resp']:.3f} / {m['pnll_bg']:.3f}")
        print(f"  resp/bg count corr         : {m['resp_loc_corr']:.3f} / {m['bg_corr']:.3f}")

    # ----- GRU forecasting baseline + multi-horizon comparison
    if not args.no_multi_horizon or not args.no_baseline:
        print("\n=== Forecasting (multi-horizon MAE, raw S) ===")
        mh_model = multi_horizon_mae(model, test_data, device, args.batch_size, horizons)
        gru = None
        if not args.no_baseline:
            from baselines import GRUForecaster, train_gru
            gru = GRUForecaster(x_feat_dim=x_feat_dim, dt=dt).to(device)
            train_gru(gru, train_data, device, epochs=max(60, args.epochs // 4),
                      lr=args.lr, batch_size=args.batch_size, seed=args.seed)
            mh_gru = multi_horizon_mae(gru, test_data, device, args.batch_size, horizons)
        hdr = "  horizon:      " + "  ".join(f"{h:>7d}" for h in mh_model)
        print(hdr)
        print("  EF-MJD :      " + "  ".join(f"{mh_model[h]:7.3f}" for h in mh_model))
        if gru is not None:
            print("  GRU    :      " + "  ".join(f"{mh_gru[h]:7.3f}" for h in mh_gru))

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
