"""
Glue between the synthetic benchmark and the existing Neural MJD model.

We reuse the *unmodified* `MJDTransformer` backbone and `NeuralMJD` head from the
repository. Because the benchmark is a single univariate trajectory, every sample
is a trivial one-node graph (N = 1), which makes the graph-transformer plumbing
collapse to a standard sequence model while keeping the exact MJD parameterisation
(mu, sigma, lambda, nu, gamma) and sampler that the paper uses.

Two baselines are constructed with the same code path, differing only in the input
channels of `node_past_dyn_data`:

    * "neural_mjd"        : channel 0 = normalised price, channel 1 = time index
    * "neural_mjd_ctx"    : the above + 2 observed event-context channels
"""

import math
import numpy as np
import torch

from model.transformer import MJDTransformer
from model.mjd.neural_mjd import NeuralMJD
from .synthetic import PAST_LEN, FUTURE_LEN, N_EVENT_CHANNELS

EVENT_CTX_SCALE = 4.0    # amplification of the sparse event-context markers


def build_model(in_seq_dim, feature_dims=64, num_layers=2, num_heads=4,
                steps_per_unit_time=5, dropout=0.0, seed=0, w_cond_mean_loss=5.0):
    """Construct a small Neural MJD (transformer backbone + MJD head)."""
    torch.manual_seed(seed)
    network = MJDTransformer(
        in_seq_length=PAST_LEN,
        in_seq_dim=in_seq_dim,
        out_seq_length=FUTURE_LEN,
        out_seq_dim=5,                       # MJD: mu, sigma, lambda, nu, gamma
        num_static_features=1,
        num_encoder_layers=num_layers,
        embedding_dim=feature_dims,
        ffn_embedding_dim=feature_dims,
        num_attention_heads=num_heads,
        pre_layernorm=False,
        activation_fn="relu",
        dropout=dropout,
        light_mode=True,                     # N = 1, skip inter-node fusion
    )
    model = NeuralMJD(
        model=network,
        w_cond_mean_loss=w_cond_mean_loss,
        n_runs=16,
        steps_per_unit_time=steps_per_unit_time,
        jump_diffusion=True,
        s_0_from_avg=False,
        cond_mean_raw_scale=False,
    )
    return model


def _past_channels(seq, use_ctx):
    """Stack the past-window input channels for one sequence."""
    s = seq["s"]
    coef = float(np.maximum(s[:PAST_LEN].max(), 1e-6))   # per-seq max normalisation
    price_norm = s[:PAST_LEN] / coef                     # [P]
    time_feat = np.arange(PAST_LEN, dtype=np.float32) / PAST_LEN
    chans = [price_norm, time_feat]
    if use_ctx:
        for ch in range(N_EVENT_CHANNELS):
            chans.append(seq["event_ctx"][ch, :PAST_LEN])
    return np.stack(chans, axis=0).astype(np.float32), coef   # [C, P], scalar


def make_batch(seqs, use_ctx, device, event_override=None):
    """Assemble a `batched_data` dict and raw future target for a list of sequences.

    Args:
        seqs: list of synthetic sequence dicts.
        use_ctx: whether to include the event-context channels.
        device: torch device.
        event_override: optional callable(seq) -> [2, P] replacement event-context
            channels (used for counterfactual / occlusion probing).
    """
    B = len(seqs)
    C = 2 + (N_EVENT_CHANNELS if use_ctx else 0)
    dyn = np.zeros((B, C, PAST_LEN), dtype=np.float32)
    coefs = np.zeros((B,), dtype=np.float32)
    target = np.zeros((B, FUTURE_LEN), dtype=np.float32)

    for b, seq in enumerate(seqs):
        chans, coef = _past_channels(seq, use_ctx)
        if use_ctx and event_override is not None:
            chans = chans.copy()
            chans[2:2 + N_EVENT_CHANNELS] = event_override(seq)
        if use_ctx:
            # amplify the sparse event markers so they are not washed out next to
            # the O(1) normalised price channel when the input window is flattened.
            chans = chans.copy()
            chans[2:2 + N_EVENT_CHANNELS] *= EVENT_CTX_SCALE
        dyn[b] = chans
        coefs[b] = coef
        target[b] = seq["s"][PAST_LEN:]

    dyn_t = torch.from_numpy(dyn).to(device).unsqueeze(1)          # [B, 1, C, P]
    coef_t = torch.from_numpy(coefs).to(device).view(B, 1, 1)      # [B, N=1, 1]
    target_t = torch.from_numpy(target).to(device).unsqueeze(1)    # [B, 1, F]

    node_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
    static = torch.zeros(B, 1, 1, device=device)

    batched_data = {
        "node_type": torch.ones(B, 1, 1, dtype=torch.long, device=device),
        "in_degree": torch.zeros(B, 1, dtype=torch.long, device=device),
        "out_degree": torch.zeros(B, 1, dtype=torch.long, device=device),
        "spatial_pos": torch.zeros(B, 1, 1, dtype=torch.long, device=device),
        "edge_attr": torch.zeros(B, 1, 1, 1, device=device),
        "adj_matrix": torch.zeros(B, 1, 1, dtype=torch.long, device=device),
        "node_mask": node_mask,
        "node_past_dyn_data": dyn_t,
        "node_past_static_data": static,
        "node_norm_coef": coef_t,
        "data_norm": "max",
        "huber_delta": 1.0,
    }
    return batched_data, target_t


def get_mjd_params(model, batched_data):
    """Return the per-future-step MJD parameters as a dict of [B, F] tensors."""
    raw = model.model(batched_data)                              # [B, 1, F, 5]
    mus, sigmas, lam_logits, nus, gammas = raw.chunk(5, dim=-1)
    mus = mus.squeeze(-1).squeeze(1)
    sigmas = sigmas.squeeze(-1).squeeze(1)
    lam_logits = lam_logits.squeeze(-1).squeeze(1)
    nus = nus.squeeze(-1).squeeze(1)
    gammas = gammas.squeeze(-1).squeeze(1)

    bound_lambdas = 1.0
    lam_logits = lam_logits.clamp(max=math.log(bound_lambdas))
    lambdas = torch.exp(lam_logits).clamp(1e-6, bound_lambdas)
    nus = nus.clamp(-0.5, 0.5)
    gammas = gammas.clamp(0, 1.0)
    # expected absolute jump contribution per step (intensity * mean |jump size|)
    jump_energy = lambdas * (nus.abs() + gammas.abs())
    return {
        "mu": mus, "sigma": sigmas, "lambda": lambdas,
        "nu": nus, "gamma": gammas, "jump_energy": jump_energy,
    }


@torch.no_grad()
def predict_mean(model, batched_data, target):
    """Conditional-mean point forecast on the raw scale, [B, F].

    The deterministic MJD conditional mean exp(cumsum(mu_t)) is the model's drift
    forecast; the delayed event responses are fit through mu_t, so this is the
    clean point estimate (robust to the heavy tails of jump sample paths).
    """
    _, _, cond_mean = model(batched_data, target=target, flag_sample=False)
    return cond_mean.squeeze(1)


def train(model, train_seqs, use_ctx, device, epochs=40, batch_size=64,
          lr=1e-3, log_every=10, verbose=True):
    """Standard Neural MJD training loop (cond-mean + MJD negative log-likelihood)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    model.train()
    n = len(train_seqs)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            seqs = [train_seqs[j] for j in idx]
            bd, target = make_batch(seqs, use_ctx, device)
            cond_mean_loss, ll_loss, _ = model(bd, target=target, flag_sample=False)
            loss = (cond_mean_loss + ll_loss).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            ep_loss += float(loss.detach())
            nb += 1
        sched.step()
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  loss {ep_loss / max(nb,1):.4f}")
    model.eval()
    return model
