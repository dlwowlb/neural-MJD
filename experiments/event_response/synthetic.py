"""
Synthetic overlapping event-response benchmark.

The goal of this benchmark is NOT to make forecasting hard. It is to expose a
structural limitation of Neural MJD: the model has a single aggregate jump
intensity lambda_t and therefore cannot say *which observed event* produced a
given response, especially when two delayed responses overlap in time.

Generative model (glucose / meal-insulin analogy)
--------------------------------------------------
A single trajectory S_t (e.g. glucose) is observed over a window of length
T = P + F (past P, future F). Discrete, *observed* events e_i = (tau_i, c_i, m_i)
are logged in the past window:

    c_i in {+1, -1}    event type  (+1 = "meal"   -> delayed rise,
                                    -1 = "insulin" -> delayed fall)
    m_i                event magnitude
    tau_i              event time (past index)

Each event produces a delayed response that lands in the future window:

    rho_i = tau_i + Delta              response onset
    R_i   = [rho_i, rho_i + L)         response segment

The log-returns of the trajectory are

    r_t = mu0 + sigma0 * eps_t + sum_i c_i * m_i * kernel(t - rho_i)

so the response of event i is *additive and known in closed form*. This gives
us, for free, every ground-truth quantity an explanation method would need:

    * event times tau_i
    * response onsets rho_i
    * the event -> response link
    * the per-step attribution label (which event dominates step t)
    * the response segment R_i
    * the counterfactual trajectory Y^{(-i)} with event i removed.

The overlap stress test sweeps the event gap g = tau_B - tau_A. As g -> 0 the
two response segments overlap more and source attribution becomes ambiguous for
any model that only predicts an aggregate intensity.
"""

import numpy as np


# ----------------------------------------------------------------------------
# Fixed benchmark geometry
# ----------------------------------------------------------------------------
PAST_LEN = 24            # P
FUTURE_LEN = 24          # F
TOTAL_LEN = PAST_LEN + FUTURE_LEN

RESPONSE_DELAY = 8       # Delta: steps between an event and its response onset
RESPONSE_DUR = 6         # L: length of a response segment
BASE_LEVEL = 100.0       # nominal trajectory level (glucose-like)
MU0 = 0.0                # baseline drift of log-returns
SIGMA0 = 0.0015          # baseline diffusion of log-returns

N_EVENT_CHANNELS = 2     # channel 0 = "meal" (c=+1), channel 1 = "insulin" (c=-1)


def _response_kernel(length):
    """Raised-cosine bump that integrates to 1 over `length` steps."""
    t = np.arange(length)
    k = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t + 0.5) / length))
    return k / k.sum()


_KERNEL = _response_kernel(RESPONSE_DUR)


def _build(events, rng):
    """Assemble a full sequence dict from an explicit list of event dicts."""
    n_ev = len(events)

    # contrib[i, t] = signed log-return contribution of event i at step t
    contrib = np.zeros((n_ev, TOTAL_LEN), dtype=np.float64)
    for i, ev in enumerate(events):
        rho = ev["tau"] + RESPONSE_DELAY
        seg_len = min(RESPONSE_DUR, TOTAL_LEN - rho)
        contrib[i, rho:rho + seg_len] = ev["c"] * ev["m"] * _KERNEL[:seg_len]

    eps = rng.normal(size=TOTAL_LEN)
    base_returns = MU0 + SIGMA0 * eps
    returns = base_returns + contrib.sum(axis=0)
    s = np.exp(np.log(BASE_LEVEL) + np.cumsum(returns))

    # counterfactuals: remove one event's response (same noise)
    counterfactual = np.zeros((n_ev, TOTAL_LEN), dtype=np.float64)
    for i in range(n_ev):
        cf_returns = returns - contrib[i]
        counterfactual[i] = np.exp(np.log(BASE_LEVEL) + np.cumsum(cf_returns))

    # observed event-context channels (channel 0 = up events, 1 = down events)
    event_ctx = np.zeros((N_EVENT_CHANNELS, TOTAL_LEN), dtype=np.float64)
    for ev in events:
        ch = 0 if ev["c"] == +1 else 1
        event_ctx[ch, ev["tau"]] += ev["m"]

    # ground-truth per-future-step source: dominant event by |contribution|
    fut = slice(PAST_LEN, TOTAL_LEN)
    contrib_fut = np.abs(contrib[:, fut])                      # [n_ev, F]
    total_fut = contrib_fut.sum(axis=0)                        # [F]
    resp_thresh = 0.15 * (np.abs(contrib).sum(axis=0).max())
    dominant = np.argmax(contrib_fut, axis=0)
    attr_label = np.where(total_fut > resp_thresh, dominant, -1)
    segments = [(contrib_fut[i] > resp_thresh).nonzero()[0] for i in range(n_ev)]

    return {
        "s": s.astype(np.float32),
        "returns": returns.astype(np.float32),
        "event_ctx": event_ctx.astype(np.float32),
        "events": events,
        "contrib": contrib.astype(np.float32),
        "counterfactual": counterfactual.astype(np.float32),
        "attr_label": attr_label.astype(np.int64),
        "segments": segments,
        "rho": np.array([ev["tau"] + RESPONSE_DELAY for ev in events], dtype=np.int64),
    }


def generate_sequence(gap, rng, opposite_type=False):
    """Two same-type (default) events separated by `gap`, for the stress test.

    With two SAME-type events the aggregate jump intensity/sign carries no source
    information, so attribution genuinely requires an event-specific variable --
    the regime where Neural MJD's single lambda_t is insufficient. As `gap` -> 0
    the two response segments overlap and attribution becomes ambiguous.
    """
    tau_lo, tau_hi = PAST_LEN - RESPONSE_DELAY, PAST_LEN - 1   # [16, 23]
    tau_a = int(rng.integers(tau_lo, max(tau_lo, tau_hi - gap) + 1))
    tau_b = min(tau_a + gap, tau_hi)
    base_sign = +1 if rng.random() < 0.5 else -1
    c_a, c_b = (base_sign, -base_sign) if opposite_type else (base_sign, base_sign)
    # near-equal magnitudes so temporal overlap (not magnitude) drives difficulty
    m_a = rng.uniform(0.18, 0.24)
    m_b = rng.uniform(0.18, 0.24)
    events = [
        {"tau": int(tau_a), "c": int(c_a), "m": float(m_a)},
        {"tau": int(tau_b), "c": int(c_b), "m": float(m_b)},
    ]
    seq = _build(events, rng)
    seq["gap"] = int(gap)
    return seq


def generate_train_sequence(rng):
    """A single training trajectory with 1 OR 2 events at varied times/types.

    Including single-event and well-separated two-event examples forces the model
    to learn *localized* event->response mappings (rather than collapsing events
    into one aggregate response). This is what makes the overlap stress test
    meaningful: the model CAN localize separated responses, and we then show it
    still cannot attribute overlapping ones.
    """
    tau_lo, tau_hi = PAST_LEN - RESPONSE_DELAY, PAST_LEN - 1   # [16, 23]
    n_ev = 1 if rng.random() < 0.5 else 2
    events = []
    if n_ev == 1:
        tau = int(rng.integers(tau_lo, tau_hi + 1))
        c = +1 if rng.random() < 0.5 else -1
        events.append({"tau": tau, "c": c, "m": float(rng.uniform(0.18, 0.24))})
    else:
        gap = int(rng.integers(1, tau_hi - tau_lo + 1))        # 1..7
        tau_a = int(rng.integers(tau_lo, tau_hi - gap + 1))
        tau_b = tau_a + gap
        base_sign = +1 if rng.random() < 0.5 else -1
        # mostly opposite-type (meal/insulin regime) but occasionally same-type
        c_a, c_b = (base_sign, -base_sign) if rng.random() < 0.75 else (base_sign, base_sign)
        events.append({"tau": tau_a, "c": c_a, "m": float(rng.uniform(0.18, 0.24))})
        events.append({"tau": tau_b, "c": c_b, "m": float(rng.uniform(0.18, 0.24))})
    return _build(events, rng)


def make_dataset(n, gap, seed, gap_jitter=0, opposite_type=False):
    """Build `n` two-event test sequences with a given (jittered) gap."""
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n):
        g = gap
        if gap_jitter:
            g = max(0, gap + int(rng.integers(-gap_jitter, gap_jitter + 1)))
        data.append(generate_sequence(g, rng, opposite_type=opposite_type))
    return data


def make_train_dataset(n, seed):
    """Build `n` mixed single/double-event training sequences."""
    rng = np.random.default_rng(seed)
    return [generate_train_sequence(rng) for _ in range(n)]
