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
| `synthetic.py` | **Event-conditioned marked-MJD** simulator: samples a marked latent jump counter `N^E(dt,dy,dr)` and saves ground-truth jump counts (`gt_K_resp`, `gt_K_bg`), per-event counts (`gt_K_evt`), and signed contributions (`gt_R`). |
| `model.py` | `EventFieldMJD`: encoder, response field, intensities, attribution, collapsed truncated likelihood, posterior attribution. |
| `validate.py` | Train on NLL + routed mean loss, then report forecast quality **and** attribution recovery. Writes plots to `outputs/`. |

## Data-generating process (`synthetic.py`)

Just as Neural-MJD validates on data drawn from an MJD, we draw from the
**event-conditioned marked MJD** the model assumes — so every latent jump has a
*source mark* and we have ground-truth attribution, not just a trajectory:

```
events e_i=(τ_i, c_i, m_i)  →  g_i(t)=α_{c_i} m_i K_{c_i}(t−τ_i)·1{0<t−τ_i<W}
λ_resp^GT(t)=λ_min+Σ_i g_i(t)      π_i^GT(t)=g_i(t)/Σ_l g_l(t)
N^E(dt,dy,{i}) ~ λ_resp^GT π_i^GT f_i      N^E(dt,dy,{0}) ~ λ_bg^GT f_bg
log Y_i ~ N(ν_{c_i} m_i, γ_resp²)   (meal/stress>0, insulin/exercise<0)
dX_t = μ^GT dt + σ^GT dW_t + ∫ log y N^E(dt,dy,dr)
```

Event types `{meal, insulin, exercise, stress}` have distinct **Gamma response
kernels** (delayed peaks), a per-event **onset delay** `δ_i ~ U(0, max_delay)`
(response is hard-windowed to `0 < t−τ_i−δ_i < W`), and **signed magnitudes**.
A per-subject **latent gain** `personal ~ LogNormal(0, σ)` scales every effect
but is *never shown to the model*, so the feature `x_i = [onehot(type), m_i]`
no longer fully reveals the response (the model must infer sign/shape/scale).
Per sensor interval we store `K_bg^GT`, `K_resp^GT`, per-event `K_{i,j}^GT`, and
the signed `R_{i,j}^GT = Σ log Y`; per event we also store the true onset delay
`gt_delay`, signed effect mean `gt_amp`, and type `gt_event_type`.

## Run

```bash
cd event_field_mjd
python validate.py                 # full run (~250 epochs) + GRU baseline + multi-horizon
python validate.py --epochs 30 --num_samples 256 --no_plot   # quick check
python validate.py --overlap_sweep                           # Synthetic-II: argmax-match vs overlap
python validate.py --misspec                                 # Synthetic-III: skew + label noise/missing
# knobs: --w_mean --w_rho --w_ent  (loss)  | --event_span_frac (overlap; smaller=more)
#        --max_delay --personal_scale_sigma | --mag_skew --label_noise_p --label_missing_p --smooth_response
#        --horizons 1,4,8,12 | --no_baseline --no_multi_horizon
```

### Metrics reported
- **Forecast:** one-step MAE (raw `S`) and **multi-horizon** MAE (1/4/8/12-step, autoregressive) vs a **GRU** baseline.
- **Attribution headline — interval argmax-match:** for each interval that truly
  had a response, does the model give the largest expected count to the *same*
  event the GT did? Reported as top-1, top-2, and an **overlap-only** slice
  (≥2 co-active events), against **random / most-recent / largest-magnitude** baselines.
- **Counts:** per-event count corr & MAE (`A_{i,j}` vs `K^GT_{i,j}`), response/background
  count corr, and count **Poisson-NLL**.
- **Signed response (secondary):** Pearson corr, sign-accuracy, MAE of `R_{i,j}` vs GT.

## Training objective

```
L = L_NLL  +  w_mean · L_route  +  w_rho · L_rho  +  w_ent · L_ent
```

- `L_NLL  = −Σ_j log p_κ(X_{t_{j+1}} | X_{t_j}, E, C_j)`  — the collapsed truncated
  likelihood (eqs. 56–58). Trains drift/diffusion, both intensities, both jump
  magnitudes, and the field `Z`.
- `L_route = Σ_j ‖ X_{t_{j+1}} − (X_{t_j} + ΔX^Y_j + Σ_i Λ^attr_{i,j} ρ_{i,j}) ‖²`
  — self-supervised reconstruction. `ΔX^Y` is a **dedicated endogenous head**
  `b_θ^Y(X_{t_j}, h_t^Y, Δ_j)`; the response term is routed through `π̄` and the
  event magnitude `ρ`, which is the **only** gradient the attribution softmax /
  trace receive (the collapsed `L_NLL` marginalises event identity). No
  background term enters this reconstruction.
- `L_rho = mean ρ²` — keeps event magnitudes small.
- `L_ent = Σ π̄ log π̄` — entropy term (minimised → higher entropy) that
  discourages degenerate attribution collapse.

## Results (250 epochs, CPU, held-out test; seeds 0–3)

| metric | result | reading |
|--------|--------|---------|
| NLL / interval | **≈ −2.5** | likelihood converges |
| one-step MAE (raw `S`) | **≈ 2.7** | forecast is good |
| count Poisson-NLL (resp/bg) | converges | intensities calibrated; bg-count corr ≈ 0.4 |
| **interval argmax-match (top-1)** | **≈ random** | ⚠ attribution **not** resolved |
| signed-`R` sign-accuracy | **≈ 0.2–0.3** | ⚠ systematically *anti*-correlated |

> ### ⚠ Finding: attribution is not yet reliably identified
> The stronger metrics added here (interval argmax-match + multi-seed) reveal
> that, under the spec-faithful objective, **the model forecasts well but does
> not reliably recover per-event attribution**: across seeds 0–3 the argmax-match
> is at chance and the signed-`R` sign-accuracy sits at ~0.2 (anti-correlated).
> Earlier single-run numbers (corr 0.81 / sign 0.99) were a **lucky optimisation
> basin, not reproducible** — exactly the failure mode the headline change to
> argmax-match was meant to catch (a magnitude-weighted global correlation can
> look good while interval-level attribution is wrong).
>
> **Root cause.** The NLL drift `μ = p_θ(h_t^Y)` is conditioned on the causal
> trajectory encoding `h`, which already reflects the ongoing event response, so
> `μ` can predict the response-driven increment with **zero response jumps**. The
> likelihood is then indifferent to `λ_resp`, `π`, `ρ`, leaving attribution to a
> weak self-supervised signal (`L_route`) whose optimisation is **bimodal**
> (correct- vs flipped-sign basins). Restricting the *endogenous* head `b_θ`
> alone does **not** fix it (tested), because the absorption happens in `μ`.
>
> **Proposed fix (not yet applied — deviates from the spec's eq. 26/52).** Make
> the NLL response magnitude *attribution-derived*, `ν̄_j = Σ_i π̄_{i,j} ρ_{i,j}`,
> so the likelihood itself depends on `π,ρ` and identifies them (and their sign)
> directly. This couples attribution into the collapsed mixture rather than
> leaving it to `L_route`. Awaiting decision since it changes the likelihood.

> **Validation scope.** This is the **Level-1, model-matched** check: the DGP
> obeys the model's own assumptions, so strong recovery is the *expected* sanity
> result, not evidence of robustness. Natural follow-ups are Level-2
> (semi-misspecified: smooth responses, skewed magnitudes, subject-varying delay,
> noisy/missing event labels) and Level-3 (real-data-informed observation gaps,
> event rates, baselines) — those stress-test attribution rather than confirm it.

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
| (28) drift/diffusion `μ,σ = p_θ^Y(h_t^Y)` | `model.interval_params` (`p_theta`, trajectory-only) |
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
| endogenous head `ΔX^Y = b_θ^Y(X,h,Δ)` | `model.forward` (`b_theta`, `dXY`) |
| routed reconstruction `L_route` | `model.forward` (`mean_loss`) |
| magnitude reg `L_ρ`, entropy reg `L_ent` | `model.forward` (`L_rho`, `L_ent`) |
| overall loss `L = L_NLL + ω L_route + …` | `model.forward` (`loss`) |
| event magnitude `ρ_{i,j} = ρ_max tanh r_θ(ζ,Z,h)` | `model.attribution_shares` (`rho_head`) |

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
