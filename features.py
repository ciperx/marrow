"""Features for Marrow: what the operating system already sees, turned into numbers.

Everything here is computed from a Marrow Trace (mtf.Trace) alone, so it is
device-agnostic by construction: positions are screen fractions, text events are
labelled by how they arrived, and no feature needs to know the machine, browser or app.

Three families:
  pointer       kinematics of movement and clicking
  text          rhythm and correction behaviour of typed / on-screen / dictated input
  dimensionless ratios between the two, which survive a bad day far better than raw
  speeds do: a tired hand is slower, but its *shakiness per unit of speed* and its
  *correction rhythm relative to its own typing rhythm* are much more stable.

Rates are expressed per event or per sample, never per second, so a session that happens
to be longer or shorter does not move them. That mistake was visible in the first run:
click_rate and dir_change_rate produced 70-90 sigma outliers between sessions of the same
person, which is noise, not identity.
"""
from __future__ import annotations

import math
import statistics as st

FEATURES = (
    # pointer
    "speed_mean", "speed_cv", "jerk_mean", "pause_ratio",
    "path_efficiency", "reversal_per_100", "click_per_100ev",
    # text
    "text_per_100ev", "iki_cv", "correction_rate", "correction_latency",
    "dictation_ratio", "burst_len_mean",
    # dimensionless, state-robust
    "tremor_ratio", "latency_ratio",
)

PAUSE_MS = 250.0        # a gap longer than this counts as a pause, not movement

# Which features are allowed to decide. Everything here is either a rate per event or a
# ratio between two of the person's own measurements, so it does not move when the same
# person is simply slower or more tired today. The absolute magnitudes (speed_mean,
# jerk_mean, correction_latency, pause_ratio) are still computed and printed, but they are
# NOT scored: a tired hand halves them, and scoring on them locks the real user out exactly
# as readily as it keeps the impostor out. That was measured, not assumed.
SCORED = (
    "speed_cv", "path_efficiency", "reversal_per_100", "click_per_100ev",
    "text_per_100ev", "iki_cv", "correction_rate", "burst_len_mean",
    "dictation_ratio", "tremor_ratio", "latency_ratio",
)


def _quantile(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


def _pointer(trace, n_events):
    moves = [e for e in trace.events if e["type"] == "move"]
    clicks = [e for e in trace.events if e["type"] == "click"]

    speeds, jerks, travelled, reversals = [], [], 0.0, 0
    last_dir = None
    for a, b in zip(moves, moves[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        step = math.hypot(dx, dy)
        v = step / dt
        travelled += step
        if speeds:
            jerks.append(abs(v - speeds[-1]) / dt)
        speeds.append(v)
        if step > 1e-5:
            heading = math.atan2(dy, dx)
            if last_dir is not None:
                d = abs(math.atan2(math.sin(heading - last_dir), math.cos(heading - last_dir)))
                if d > math.pi / 2:
                    reversals += 1
            last_dir = heading

    span = 0.0
    if len(moves) >= 2:
        span = math.hypot(moves[-1]["x"] - moves[0]["x"], moves[-1]["y"] - moves[0]["y"])

    gaps = [b["t"] - a["t"] for a, b in zip(trace.events, trace.events[1:])]
    pause_ms = sum(g for g in gaps if g > PAUSE_MS)

    speed_mean = st.fmean(speeds) if speeds else 0.0
    speed_cv = (st.pstdev(speeds) / speed_mean) if len(speeds) > 1 and speed_mean > 0 else 0.0
    jerk_mean = st.fmean(jerks) if jerks else 0.0
    n = max(len(moves), 1)
    return {
        "speed_mean": speed_mean,
        "speed_cv": speed_cv,
        "jerk_mean": jerk_mean,
        "pause_ratio": min(1.0, pause_ms / max(trace.duration_ms, 1.0)),
        "path_efficiency": min(1.0, span / travelled) if travelled > 1e-9 else 0.0,
        "reversal_per_100": 100.0 * reversals / n,
        "click_per_100ev": 100.0 * len(clicks) / max(n_events, 1),
        "tremor_ratio": jerk_mean / speed_mean if speed_mean > 1e-12 else 0.0,
    }


def _text(trace, n_events):
    texts = [e for e in trace.events if e["type"] == "text"]
    real = [e for e in texts if not e.get("correction")]
    corr = [e for e in texts if e.get("correction")]

    ikis = [b["t"] - a["t"] for a, b in zip(real, real[1:])]
    iki_mean = st.fmean(ikis) if ikis else 0.0
    iki_cv = (st.pstdev(ikis) / iki_mean) if len(ikis) > 1 and iki_mean > 0 else 0.0

    lat, last = [], None
    for e in texts:
        if e.get("correction") and last is not None:
            lat.append(e["t"] - last)
        elif not e.get("correction"):
            last = e["t"]
    corr_lat = _quantile(lat, 0.5)

    labels = [e.get("label", "typed") for e in real]
    dictation = sum(1 for l in labels if l == "dictation") / max(len(labels), 1)

    bursts, run, prev = [], 0, None
    for l in labels:
        if l == prev:
            run += 1
        else:
            if run:
                bursts.append(run)
            run, prev = 1, l
    if run:
        bursts.append(run)

    return {
        "text_per_100ev": 100.0 * len(real) / max(n_events, 1),
        "iki_cv": iki_cv,
        "correction_rate": len(corr) / max(len(texts), 1),
        "correction_latency": corr_lat,
        "dictation_ratio": dictation,
        "burst_len_mean": st.fmean(bursts) if bursts else 0.0,
        "latency_ratio": corr_lat / iki_mean if iki_mean > 1e-9 else 0.0,
    }


def features(trace):
    """Feature dict for one recording. Order follows FEATURES."""
    n_events = max(len(trace.events), 1)
    f = {}
    f.update(_pointer(trace, n_events))
    f.update(_text(trace, n_events))
    return {k: float(f.get(k, 0.0)) for k in FEATURES}


def vector(trace):
    f = features(trace)
    return [f[k] for k in FEATURES]


def describe():
    return ("pointer: speed_mean, speed_cv, jerk_mean, pause_ratio, path_efficiency, "
            "reversal_per_100, click_per_100ev | text: text_per_100ev, iki_cv, "
            "correction_rate, correction_latency, dictation_ratio, burst_len_mean | "
            "ratios: tremor_ratio, latency_ratio")
