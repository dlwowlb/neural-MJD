"""Forecasting baseline for the EF-MJD synthetic validation.

A plain causal GRU that predicts the next-step log-increment from the past
trajectory and per-step event aggregates, trained with MSE. It models the
trajectory but has **no** notion of jumps, intensities, or event attribution --
so it is a fair forecast-only reference (one-step and multi-horizon), and a
reminder that attribution is something this class of model cannot provide.
"""

import torch
import torch.nn as nn


def event_grid_aggregate(tau, x_feat, evt_mask, grid, dt):
    """Sum event features into the grid step j = ceil(tau/dt). [B, T+1, F]."""
    B, M = tau.shape
    Tp1 = grid.shape[0]
    step = torch.ceil(tau / dt + 1e-6).long().clamp(0, Tp1 - 1)
    agg = torch.zeros(B, Tp1, x_feat.shape[-1], device=tau.device)
    idx = step.unsqueeze(-1).expand(-1, -1, x_feat.shape[-1])
    agg.scatter_add_(1, idx, x_feat * evt_mask.unsqueeze(-1))
    return agg


class GRUForecaster(nn.Module):
    def __init__(self, x_feat_dim, d_h=48, dt=1.0):
        super().__init__()
        self.dt = dt
        self.enc_in = 1 + x_feat_dim
        self.gru = nn.GRU(self.enc_in, d_h, batch_first=True)
        self.head = nn.Linear(d_h, 1)

    def _increment(self, X, tau, x_feat, evt_mask, grid):
        evt_agg = event_grid_aggregate(tau, x_feat, evt_mask, grid, self.dt)  # [B,T+1,F]
        x_in = (X - X[:, :1]).unsqueeze(-1)
        h, _ = self.gru(torch.cat([x_in, evt_agg], dim=-1))                   # [B,T+1,d_h]
        inc = self.head(h).squeeze(-1)                                        # increment from t_j
        return inc

    def forward(self, batch):
        X = batch["X"]
        inc = self._increment(X, batch["tau"], batch["x_feat"], batch["evt_mask"], batch["grid"])
        mean_pred = X[:, :-1] + inc[:, :-1]                                   # predict X_{j+1}
        loss = ((X[:, 1:] - mean_pred) ** 2).mean()
        return loss, mean_pred

    @torch.no_grad()
    def rollout_forecast(self, batch, start_j, horizon):
        X = batch["X"]
        horizon = min(horizon, X.shape[1] - 1 - start_j)
        x_pred = X.clone()
        preds = []
        for k in range(horizon):
            j = start_j + k
            inc = self._increment(x_pred, batch["tau"], batch["x_feat"],
                                  batch["evt_mask"], batch["grid"])
            x_next = x_pred[:, j] + inc[:, j]
            x_pred = x_pred.clone()
            x_pred[:, j + 1] = x_next
            preds.append(x_next)
        return torch.stack(preds, dim=1)


def train_gru(model, train_data, device, epochs=60, lr=3e-3, batch_size=64, seed=0):
    from validate import iterate_batches
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        for batch in iterate_batches(train_data, batch_size, True, device, gen):
            loss, _ = model(batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model
