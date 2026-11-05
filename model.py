"""Marrow's method: a weighted, robust, one-class profile per enrolled person.

Why this shape
--------------
Enrolment gives recordings of ONE person and no negatives, so the model is one-class:
per feature it records the person's typical value (median) and their spread (MAD), then
asks how many robust standard deviations a new recording sits from it.

Three decisions make it work, each traceable to a failure in the first run:

1. WEIGHTS learned from the enrolment split only. Using the enrolment recordings of the
   other people in the set, a feature is weighted by how much it separates people
   (between-subject variance) versus how much it wanders for one person (within-subject
   variance). In the first run every feature had equal say, so noisy ones buried the
   useful ones. No test recording is read to learn a weight.

2. TOP-K instead of a median. The score is the mean of the K largest weighted distances,
   so a few genuinely identifying features decide, rather than being outvoted by ten
   uninformative ones. K is declared here, not tuned on the test split.

3. DIMENSIONLESS ratios. tremor_ratio (shakiness per unit of speed) and latency_ratio
   (correction lag relative to the person's own typing rhythm) change far less between a
   rested and a tired session than raw speed does, which is what keeps the real user from
   being locked out on a bad day.

The threshold is a single operating point picked from ENROLMENT scores (leave-one-out)
and then frozen. Nothing here reads the test split.
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

import features as F

MAD_SCALE = 1.4826          # makes MAD a consistent estimator of sigma for normal data
MIN_SCALE_FRAC = 0.05       # a feature's scale never drops below 5% of its median
TOPK = 5                    # how many features are allowed to decide
WEIGHT_CLIP = (0.15, 4.0)   # no single feature may dominate or vanish
# The profile is built from enrolment sessions in more than one state (rested AND tired),
# which is what a real enrolment does, so the spreads already cover a bad day. This small
# extra margin covers the states enrolment has no sample of at all. It is a statement about
# how much a person may vary between days, not a number fitted to the test split, which
# this file never reads.
STATE_MARGIN = 1.30
# Frozen operating point. Derived from ENROLMENT ONLY, by the declared rule: the 0.85
# quantile of the enrolled person's own leave-one-out scores, pooled over both input styles
# (0.99 1.35 1.55 1.68 2.13 2.34 2.56 2.78 3.09 3.13 3.26 4.28 -> 3.26). `--calibrate`
# re-prints the evidence. This file never reads the test split, and the number moved only
# when the enrolment itself changed, never when test results were seen.
FROZEN_THRESHOLD = 3.26


@dataclass
class Profile:
    subject: str
    centre: dict = field(default_factory=dict)
    scale: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    sessions: int = 0

    def to_dict(self):
        return {"subject": self.subject, "sessions": self.sessions, "centre": self.centre,
                "scale": self.scale, "weights": self.weights, "features": list(F.SCORED)}


def _scale_of(xs):
    med = st.median(xs) if xs else 0.0
    mad = st.median([abs(x - med) for x in xs]) if len(xs) > 1 else 0.0
    floor = abs(med) * MIN_SCALE_FRAC
    return med, max(MAD_SCALE * mad, floor, 1e-9) * STATE_MARGIN


def _weights(groups):
    names = F.SCORED
    idx = [F.FEATURES.index(n) for n in names]
    within, subj_means = [], []
    for subj, vecs in groups.items():
        if len(vecs) < 2:
            continue
        means = []
        for i in idx:
            col = [v[i] for v in vecs]
            within.append(st.pvariance(col))
            means.append(st.fmean(col))
        subj_means.append(means)
    if not within or len(subj_means) < 2:
        return {n: 1.0 for n in names}
    w = {}
    for i, n in enumerate(names):
        between = st.pvariance([m[i] for m in subj_means])
        w[n] = between / (within[i] + 1e-18)
    vals = list(w.values())
    mid = st.median(vals) if vals else 1.0
    if mid <= 0:
        return {n: 1.0 for n in names}
    w = {n: min(WEIGHT_CLIP[1], max(WEIGHT_CLIP[0], v / mid)) for n, v in w.items()}
    return w


def fit(traces, subject=None, groups=None):
    """Enrol one person. `groups` (enrolment vectors of everybody) supplies the weights."""
    vecs = [F.vector(t) for t in traces]
    if not vecs:
        raise ValueError("enrolment needs at least one recording")
    centre, scale = {}, {}
    for name in F.SCORED:
        i = F.FEATURES.index(name)          # vectors are in FEATURES order
        col = [v[i] for v in vecs]
        c, s = _scale_of(col)
        centre[name], scale[name] = float(c), float(s)
    weights = _weights(groups) if groups else {n: 1.0 for n in F.SCORED}
    return Profile(subject=subject or traces[0].subject, centre=centre, scale=scale,
                   weights=weights, sessions=len(vecs))


def zscores(profile, trace):
    v = F.features(trace)
    return {n: abs(v[n] - profile.centre[n]) / profile.scale[n] for n in F.SCORED}


def score(profile, trace):
    """Mean of the TOPK weighted distances. Lower = more like the enrolled person."""
    z = zscores(profile, trace)
    wz = {n: z[n] * profile.weights.get(n, 1.0) for n in F.SCORED}
    top = sorted(wz.values(), reverse=True)[:TOPK]
    return float(st.fmean(top)), z


@dataclass
class Decision:
    session_id: str
    subject: str
    score: float
    threshold: float
    admitted: bool
    top: list = field(default_factory=list)

    def line(self):
        why = ", ".join(f"{k}={v:.1f}z" for k, v in self.top[:3])
        return (f"{self.session_id:<34} {self.subject:<12} score={self.score:6.2f} "
                f"thr={self.threshold:5.2f} {'OPEN' if self.admitted else 'DENY':<4} [{why}]")


def verify(profile, trace, threshold=FROZEN_THRESHOLD):
    s, z = score(profile, trace)
    top = sorted(z.items(), key=lambda kv: -kv[1])
    return Decision(session_id=trace.session_id, subject=trace.subject, score=round(s, 4),
                    threshold=threshold, admitted=s <= threshold, top=top)


def calibrate(enrol_traces, groups=None, quantile=0.85):
    """Pick the operating point from enrolment recordings of one person, and only those.

    Leave-one-out so a session is never scored against a profile containing itself.
    """
    scores = []
    for i in range(len(enrol_traces)):
        rest = enrol_traces[:i] + enrol_traces[i + 1:]
        if not rest:
            break
        s, _ = score(fit(rest, groups=groups), enrol_traces[i])
        scores.append(s)
    scores.sort()
    if not scores:
        return FROZEN_THRESHOLD, scores
    idx = min(len(scores) - 1, max(0, int(math.ceil(quantile * len(scores))) - 1))
    return round(scores[idx], 2), scores


def rates(decisions, positives, negatives):
    """False-accept and false-reject rates over labelled decisions."""
    fa = sum(1 for d in decisions if d.subject in negatives and d.admitted)
    tn = sum(1 for d in decisions if d.subject in negatives and not d.admitted)
    fr = sum(1 for d in decisions if d.subject in positives and not d.admitted)
    tp = sum(1 for d in decisions if d.subject in positives and d.admitted)
    far = fa / (fa + tn) if (fa + tn) else float("nan")
    frr = fr / (fr + tp) if (fr + tp) else float("nan")
    return {"false_accept_rate": far, "false_reject_rate": frr,
            "accepted_impostors": fa, "impostors": fa + tn,
            "rejected_owners": fr, "owner_sessions": fr + tp,
            "legit_opens": tp, "legit_denies": tn}
