# Event–Response Attribution Benchmark + Event-Marked Neural MJD (Ours)

This experiment makes one point precise and then fixes it:

> Neural MJD models *when* a trajectory jumps via a single aggregate intensity
> `λ_t`, but it has **no event-source variable**. Even when it predicts a jump at
> the right time — and even when the observed events are fed into the context `C`
> — it cannot say **which event** caused or explains a given response.
>
> **Ours (Event-Marked Neural MJD)** extends the model so that every jump carries
> a source label, giving each observed event its own time-varying response
> intensity and jump-magnitude distribution. This recovers event→response
> attribution while preserving Neural MJD's compensated form and closed-form mean.

Three methods are compared on the same synthetic benchmark:

| method | what it predicts |
|---|---|
| `neural_mjd` | `μ_t,σ_t,λ_t,ν_t,γ_t` from past only (no events) |
| `neural_mjd_ctx` | the same, with observed events fed into context `C` |
| **`ours`** | per-source `{λ_{r,t},ν_{r,t},γ_{r,t}}` via an event-marked Poisson measure |

## Why the baselines cannot attribute (structural)

Neural MJD's jump SDE `dS_t = S_t[(μ_t−λ_t k_t)dt + σ_t dW_t + dQ_t]` exposes a
**total** intensity `λ_t`. There is no `P(i | response at t)` in
`(μ_t,σ_t,λ_t,ν_t,γ_t)`. Feeding events into `C` does not create one: the model
learns an aggregate response (for two same-type overlapping events, occluding
either event changes the forecast *identically*, `S₀≡S₁`; for opposite-type
events the occlusion signal is no better than chance once responses overlap).

## Ours: Event-Marked Neural MJD (`ours.py`)

We extend the unlabeled Poisson jump measure `N(dt,dy)` into an **event-marked**
measure `N^E(dt,dy,dr)` with source `r ∈ {0,…,m}` (0 = background, `i` = event
`e_i`). The network predicts, per future step,

```
μ_0,t, σ_t                              (shared)
{ λ_{r,t}, ν_{r,t}, γ_{r,t} }_{r=0..m}  (per source)      k_{r,t}=exp(ν+γ²/2)−1
```

* **Compensated form & closed-form mean preserved.** Define the effective drift
  `μ_t^E = μ_0,t + Σ_r λ_{r,t} k_{r,t}`, so `E[S_T|C,E] = S_0 exp(Σ_t μ_t^E)` and
  each source contributes an *attributable* term `c_{r,t}=λ_{r,t}k_{r,t}` to the
  conditional-mean log-return. Removing event `i` is exact: `μ^{(−i)} = μ^E − c_{i,t}`.
* **Delay is modelled, not just `t−τ_i`.** For event `i` with elapsed time
  `u=t−τ_i`, a learnable onset distribution is convolved with a response kernel:
  `λ_{i,t} = a_i(q_i) · Σ_δ p_φ(δ|q_i) κ_φ(u−δ|q_i)`, separating *when the
  response starts* from *how it unfolds* from *how strong it is*. Event
  interaction enters through `q_i`, which attends over the other events and the
  past context.
* **Source-count likelihood.** The MJD likelihood is kept but over a per-source
  count vector `k=(k_0,…,k_m)` with adaptive truncation on the total count;
  `a_k = μ_0,t − σ²/2 + Σ_r k_r ν_{r,t}`, `b²_k = σ² + Σ_r k_r γ_{r,t}²`.
* **Magnitude-aware attribution (native).** `P(i|t) = λ_{i,t}/Σ_j λ_{j,t}`, or
  with magnitude `P(i|t,y) ∝ λ_{i,t} f_i(t,y)`.

Training keeps the source-count NLL + conditional-mean regulariser; on the
synthetic benchmark (ground truth available) light attribution/delay supervision
is added. **Ablation:** with that supervision *removed*, Ours still attributes
near-perfectly — the event-marked *architecture*, not the labels, is what enables
attribution.

## Synthetic benchmark (`synthetic.py`)

A glucose-like trajectory driven by observed events: `meal` (→ delayed rise) and
`insulin` (→ delayed fall), each starting `Δ=8` steps after the event for `L=6`
steps. Every ground-truth quantity is known (event/response times, per-step
source label, segments `R_i`, counterfactuals `Y^{(−i)}`). The overlap stress
test sweeps the event gap `g=τ_B−τ_A`: as `g→0` the two responses overlap.

## Metrics (`evaluate.py`, `ours.py`)

Forecast MAE · Jump-time MAE · Event-attribution F1 · Segment IoU · Counterfactual
RMSE. Baselines are probed by occlusion (`+ctx`) or chance (no context); Ours
reads attribution natively from `λ_{i,t}`.

## Running

```bash
python -m experiments.event_response.run            # full run (CPU, ~12 min)
python -m experiments.event_response.run --quick    # fast smoke run
python -m experiments.event_response.make_figures   # rebuild figures from saved checkpoints
```

Artifacts → `results/`: `metrics.{json,csv}`, `four_panel.png`, `overlap_robustness.png`.

## Results

Averaged over overlap gaps `g ∈ {1,2,4,6}` (full run, CPU):

| Metric | `neural_mjd` | `neural_mjd_ctx` | **`ours`** | better |
|---|---:|---:|---:|---|
| Forecast MAE (raw) | 16.08 | 10.57 | **0.50** | lower |
| Jump-time MAE (steps) | 15.11 | 3.27 | **0.60** | lower |
| Event attribution F1 | 0.48 | 0.44 | **0.98** | higher |
| Segment IoU | 0.08 | 0.08 | **0.84** | higher |
| Counterfactual RMSE | 27.86 | 16.78 | **0.67** | lower |

Overlap robustness — event attribution F1 as the responses overlap (`g→1`):

| method | g=6 | g=4 | g=2 | g=1 |
|---|---:|---:|---:|---:|
| `neural_mjd` | 0.49 | 0.48 | 0.47 | 0.46 |
| `neural_mjd_ctx` | 0.39 | 0.43 | 0.47 | 0.47 |
| **`ours`** | **1.00** | **1.00** | **1.00** | **0.90** |


The pattern (see `results/`):

1. **Forecasting / jump-timing / counterfactuals improve with event information**
   (`neural_mjd_ctx` > `neural_mjd`), and **Ours is best on all of them**.
2. **Both baselines fail attribution** — event-source F1 and segment IoU sit at
   chance regardless of overlap.
3. **Ours recovers attribution**: high F1 and segment IoU that stay high as the
   responses overlap, and near-exact counterfactuals — because each event has its
   own delay-aware response intensity.

In one line: *forecasting (even with events in the context) is not event-response
attribution; an event-marked jump measure is.*

## The four-panel figure (`figure.py`)

* **A. Ground truth** — meal→rise and insulin→fall response segments.
* **B. Neural MJD** — one aggregate intensity `λ_t`: *"a jump is likely here"*,
  no source.
* **C. Neural MJD + context** — occlusion intensities `S_i(t)`, entangled across
  the overlapping responses.
* **D. Ours** — native per-event intensities `λ_i(t)`: the meal response and the
  insulin response are each attributed to their own source.
