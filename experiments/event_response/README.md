# Event–Response Attribution Benchmark (Neural MJD baselines)

This experiment establishes the **baseline behaviour** of Neural MJD on an
event-response *attribution* task. The point is **not** that Neural MJD fails to
forecast — it forecasts jumps fine. The point is sharper:

> Neural MJD models *when* a trajectory jumps via a single aggregate intensity
> `λ_t`, but it has **no event-source variable**. Even when it predicts a jump at
> the right time, it cannot say **which observed event** caused or explains that
> jump — and feeding the events into the context `C` does not fix this.

("Ours" — an event-specific response-intensity decomposition `λ_t = λ₀(t) +
Σᵢ λᵢ(t)` — is intentionally **left out here**; this directory only characterises
the baselines that such a method must beat.)

## Why Neural MJD cannot attribute (structural)

Neural MJD's jump SDE is

```
dS_t = S_t [ (μ_t − λ_t k_t) dt + σ_t dW_t + dQ_t ]
```

where `λ_t` is the **total** jump intensity. The model can answer *"a jump is
likely at t"* but not *"the jump at t came from event i"* — there is simply no
`P(i | response at t)` in the parameterisation `(μ_t, σ_t, λ_t, ν_t, γ_t)`. A
reviewer's natural objection is *"just put the events in the context `C`"*, so we
include that baseline explicitly and probe it for attribution.

## Synthetic benchmark (`synthetic.py`)

A single trajectory `S_t` (glucose-like, length `P+F = 24+24`) is driven by
discrete **observed** events logged in the past window:

* `meal`   (c = +1) → a delayed **rise** segment, and
* `insulin` (c = −1) → a delayed **fall** segment,

each starting `Δ = 8` steps after the event and lasting `L = 6` steps. Because the
responses are additive in log-returns we know, for free, every ground-truth
quantity: event times `τᵢ`, response onsets `ρᵢ = τᵢ+Δ`, the event→response link,
the per-step source label, the response segments `Rᵢ`, and the counterfactual
trajectory `Y^(−i)` with event `i` removed.

The **overlap stress test** sweeps the event gap `g = τ_B − τ_A`. As `g → 0` the
two response segments overlap and source attribution becomes ambiguous for any
model that only exposes an aggregate intensity. Training uses a mix of
single-event and well-separated two-event sequences so the model *can* learn
localized responses; the test then asks whether it can still attribute
*overlapping* ones.

## Baselines (`model_io.py`)

Both reuse the repository's unmodified `MJDTransformer` backbone and `NeuralMJD`
head (each sample is a trivial one-node graph, `N = 1`):

| baseline          | input context                                   |
|-------------------|-------------------------------------------------|
| `neural_mjd`      | past price + time only (latent jump, no events) |
| `neural_mjd_ctx`  | the above **+ observed event-context channels** |

## Attribution protocols (`evaluate.py`)

Since neither baseline has a native `P(i|t)`, each is given the *best* probe it
admits:

* **`neural_mjd_ctx`** → **occlusion attribution**: the event-`i` marker is removed
  from the context and the resulting change in the forecast *increments*
  `Sᵢ(t) = |Δŷ(t) − Δŷ_{−i}(t)|` is read as that event's post-hoc response
  intensity. This is exactly the reviewer's "events in `C`" probe.
* **`neural_mjd`** → **chance**: with no event input (and, for same-type events, no
  source information in the aggregate intensity/sign) the model can only detect
  *that* a jump occurs and then guess the source — the chance baseline.

## Metrics

* **Forecast MAE** – raw-scale conditional-mean point forecast.
* **Jump-time MAE** – `|t̂* − t*|` between the largest predicted move and the true
  response onset.
* **Event attribution F1** – macro-F1 of the predicted source over response steps.
* **Segment IoU** – overlap of predicted vs. true response segments `Rᵢ`.
* **Counterfactual RMSE** – error of the model's event-removal counterfactual
  vs. the true `Y^(−i)` (the no-context model cannot remove an event at all).

## Running

```bash
python -m experiments.event_response.run            # full run (CPU ~15-20 min)
python -m experiments.event_response.run --quick    # fast smoke run
```

Artifacts are written to `experiments/event_response/results/`:
`metrics.json`, `metrics.csv`, `three_panel.png`, `overlap_robustness.png`.

## Results

Averaged over overlap gaps `g ∈ {1,2,4,6}` (full run, CPU):

| Metric | `neural_mjd` | `neural_mjd_ctx` | better |
|---|---:|---:|---|
| Forecast MAE (raw) | 16.08 | **10.57** | lower |
| Jump-time MAE (steps) | 15.11 | **3.27** | lower |
| Event attribution F1 | 0.48 | 0.44 | higher (chance ≈ 0.5) |
| Segment IoU | 0.08 | 0.08 | higher |
| Counterfactual RMSE | 27.86 | **16.78** | lower |

The robust, reproducible takeaways are:

1. **Forecasting works** and **context helps it**: `neural_mjd_ctx` forecasts the
   delayed responses with low MAE and low jump-time error, far better than the
   context-free `neural_mjd` (which never saw the events).
2. **Counterfactuals need the events**: `neural_mjd_ctx` approximates single-event
   removal far better than `neural_mjd`, which structurally cannot remove an event.
3. **But neither baseline can attribute**: event-source F1 and segment IoU stay at
   / near **chance** for both. Occluding the context of `neural_mjd_ctx` is
   unstable — for two *same-type* overlapping events it is provably degenerate
   (removing either event changes the forecast identically, `S₀ ≡ S₁`), and for
   *opposite-type* events it is no better than chance once the responses overlap.

In one line: **forecasting and even counterfactual prediction are *not*
event-response attribution.** Neural MJD (with or without event context) is
sufficient for unlabeled jump forecasting but **insufficient for event-response
explanation** — the gap that an event-specific response-intensity decomposition
("ours") is meant to fill.

## The three-panel figure (`figure.py`)

* **A. Ground truth** — trajectory with events and their (colour-coded) response
  segments (meal→rise, insulin→fall).
* **B. Neural MJD** — the aggregate jump intensity `λ_t(|ν_t|+|γ_t|)`: *"a jump is
  likely here"*, with **no source**.
* **C. Neural MJD + context** — occlusion response intensities `Sᵢ(t)`: a post-hoc
  probe that blurs across overlapping responses.
