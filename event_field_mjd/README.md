# Event-Field Marked MJD — synthetic validation

A self-contained implementation + synthetic validation of the **Event-Field
Marked Merton Jump Diffusion** model, built in the same spirit as the Neural-MJD
synthetic demo (`../demo_notebook.ipynb`): we generate data from a *known*
generative process and check that the model recovers the structure it is meant
to learn.

The model couples a **dense sensor log-trajectory** `X_t = log S_t` with
**irregular events** `(τ_i, x_i)`. Events accumulate into a shared
**event-response field** `Z_t`; that field drives a shared **response-jump
intensity** which is split across active events by an attribution softmax; and
the per-interval likelihood is a **mark-collapsed, truncated** marked
jump-diffusion mixture. Per-event attribution is then recovered as a
**posterior** quantity.

## Files

| File | Purpose |
|------|---------|
| `synthetic.py` | Generates trajectories with known per-event log-response (`gt_R`) and background jumps (`gt_bg`). |
| `model.py` | `EventFieldMJD`: encoder, response field, intensities, attribution, collapsed truncated likelihood, posterior attribution. |
| `validate.py` | Train on NLL + auxiliary mean loss, then report forecast quality **and** attribution recovery. Writes plots to `outputs/`. |

## Run

```bash
cd event_field_mjd
python validate.py                 # full run (~120 epochs, CPU-friendly)
python validate.py --epochs 30 --num_samples 256 --no_plot   # quick check
```

## Representative result (120 epochs, CPU, held-out test set)

| metric | value | reading |
|--------|-------|---------|
| NLL / interval | **−2.29** | likelihood converges |
| one-step MAE (raw `S`) | **1.87** | ≈1.9 % on an `S≈100` scale |
| attribution corr `R` vs GT | **0.77** (shuffle −0.03) | per-event signed response recovered |
| event sign accuracy | **0.83** | up/down direction recovered |
| response localization (`K_resp`) | **0.57** | response jumps land where events act |
| background separation (`K_bg`) | **0.79** | background jumps land on real ones |

The attribution scatter (`outputs/03_attribution_scatter.png`) is clearly
monotonic; magnitude is somewhat attenuated because `R = K_resp · π̄ · ρ` with
`E[K_resp] < 1`. Exact numbers vary a little with `--seed`.

## What is validated

1. **Optimisation** — NLL (eq. 58) decreases on train.
2. **Forecast quality** — one-step mean MAE in raw `S` space (eqs. 69–70).
3. **Response localization** — correlation between posterior expected response
   count `K_resp` (eq. 63) and where the ground-truth event responses actually
   occurred. This is the cleanly *identifiable* part of the model.
4. **Background separation** — does posterior `K_bg` (eq. 62) track where
   ground-truth background jumps fired?
5. **Attribution recovery** — Pearson correlation between the model's posterior
   signed log-response `R_{i,j}` (eq. 66) and the ground-truth per-event
   contribution, reported against a shuffled baseline.
6. **Sign accuracy** — does the model get each event's up/down direction right?

## Identifiability note (an honest finding of this validation)

The **collapsed** likelihood (eqs. 53–58) marginalises over event identity: it
only sees the *total* background/response counts `(k_bg, k_resp)`, never the
shares `π`. Likewise all response jumps share one magnitude distribution
(eqs. 25–26), so even the *source-wise* likelihood (eq. 41) is invariant to how
a response is split across simultaneously-active events. **Consequence:** the
attribution softmax `π` (eqs. 15/47) and the trace `ψ` (eq. 14) get **no
gradient from the NLL**, and per-event attribution is *not identifiable from the
marginal trajectory likelihood alone*.

A faithful run therefore learns the jump structure well (NLL ↓, forecast tight,
`K_resp`/`K_bg` localize correctly) but recovers attribution only weakly — and
only because event *timing* + the response field's localized sign carry indirect
signal. This module makes that gradient explicit by routing the response term of
the **auxiliary mean loss** (eq. 70) through `π` and an event-specific magnitude
`ρ_{i,j}`:

```
resp_incr_j = Σ_i (Λ_resp,j · π̄_{i,j}) · ρ_{i,j}      # eqs. 46 + 66
```

This stays inside the spec's `L_mean` term, is fully self-supervised (it
reconstructs `X`, never the ground truth), and makes attribution identifiable
whenever active events are reasonably separated in time — which is what lifts
`attr_corr` and `sign_acc` well above their shuffled baselines.

## Equation → code map

| Spec | Where |
|------|-------|
| (1–5) dense grid / log-space | `synthetic.generate_dataset` (`grid`, `X = log S`) |
| (7) history `h_t = Enc(...)` | `model.encode_history` (causal GRU, no leakage — eq. 68) |
| (8–10) response field `Z_t` ODE + jumps | `model.rollout_field` (`F_theta`, `U_theta`) |
| (11–12) intensities `λ_resp`, `λ_bg` | `model.interval_params` (`g_resp`, `g_bg`) |
| (13/49) active set `A_t` / `A_j` | `model.attribution_shares` (`overlap`) |
| (14) event trace `ζ_i(t)` | `model.attribution_shares` (`psi`) |
| (15–16, 47) attribution share `π_i` / `π̄_{i,j}` | `model.attribution_shares` (masked softmax) |
| (23–26) jump magnitudes | `model.interval_params` (`m_bg`, `m_resp`) |
| (28) drift/diffusion | `model.interval_params` (`p_theta`) |
| (29) compensator `κ_t` | `model._collapsed_terms` |
| (54–55) collapsed mean/variance | `model._collapsed_terms` (`a`, `b2`) |
| (57) truncated count set | `EventFieldMJD.__init__` (`k_bg_grid`, `k_resp_grid`) |
| (58) truncated likelihood | `model.forward` (`logsumexp`) |
| (60–61) posterior weights `q` | `model.attribute` |
| (62–63) posterior counts `K̂` | `model.attribute` |
| (64) attribution score `A_{i,j}` | `model.attribute` (`A_hat`) |
| (65) prob ≥1 response jump | `model.attribute` (`P_resp`) |
| (66) signed log-response `R_{i,j}` | `model.attribute` (`R_hat`, via `ρ`) |
| (46) attributed intensity `Λ_attr` | `model.forward` (`Lam_attr`) |
| (69–70) overall loss | `model.forward` (`loss`) |
| event magnitude `ρ_{i,j}` (extension) | `model.attribution_shares` (`rho_head`) |

## Modelling choices / approximations

These keep the formulation auditable while staying batchable:

- **No leakage** (eq. 68): `h_{t_j}` is the causal GRU hidden state after
  consuming inputs up to `t_j`; the response/drift/intensity heads for interval
  `j` read only `(h_{t_j}, Z_{t_j})` plus events inside the interval.
- **Piecewise-constant intervals** (eqs. 37/51): integrated intensities are
  `Λ = λ · Δ`, and magnitude/drift parameters are constant within an interval.
- **`Z` integration**: Euler step for the continuous part `F_θ`, additive event
  jumps `U_θ` for events falling in the interval; `Z_{τ_i^-}` and `h_{τ_i^-}`
  are approximated by their interval-start values.
- **Stability clamps** mirror Neural-MJD (`bound_mu`, `bound_sigma`,
  `bound_nu`, `bound_gamma`, `bound_lambda`).
