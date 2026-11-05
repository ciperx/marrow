#!/usr/bin/env python3
"""The shoulder-surfer test: how much of the decision rests on what an observer can SEE.

Why this exists. The sponsor's constraint says the fallback is part of the problem: her
current password entry is either repeatable by anyone in the room or too painful to keep
long. A verifier that decides from movement has an obvious version of that worry - somebody
watches her use the machine, then does the same thing. This script measures it instead of
arguing about it, with two opponents who are as close to that attacker as fake data allows.

    replay    the visible session, positions AND timestamps copied exactly, played back.
              This is a recording, and it is the attacker that no behavioural check can
              stop, because there is nothing left to measure but the recording itself.

    watcher   the visible path copied exactly - every position, every click, every text
              label and correction - but performed with somebody else's rhythm: the gaps
              between events come from a stranger's session, rescaled so the watcher spends
              the same total time and the copy is fair rather than obviously fast or slow.

The interesting number is the watcher. Positions are the part of a session a person in the
room can copy; the micro-timing and the tremor are not, not even after watching for years.
Each feature that decided is therefore labelled SHAPE (visible, so a watcher can reproduce
it) or DYNAMICS (not visible). If a watcher gets in, this says which features let it in and
the fix is named by the same table.

Run: python3 shoulder_surf.py [--data data] [--json shoulder_surf.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as E      # noqa: E402
import features as F      # noqa: E402
import model              # noqa: E402
import mtf                # noqa: E402

# What a person in the room can see, versus what they cannot. Position, order, and the
# result of an action are visible; speed, tremor and the lag before a correction are not.
SHAPE = ("path_efficiency", "reversal_per_100", "click_per_100ev", "text_per_100ev",
         "dictation_ratio", "burst_len_mean", "correction_rate")
DYNAMICS = ("speed_cv", "iki_cv", "tremor_ratio", "latency_ratio")


def kind(name):
    return "SHAPE" if name in SHAPE else ("DYNAMICS" if name in DYNAMICS else "?")


def gaps(events):
    return [b["t"] - a["t"] for a, b in zip(events, events[1:])]


def retime_like(trace, donor, session_id):
    """Same positions, clicks and text; the gaps between events come from the donor.

    The donor's gaps are repeated cyclically and rescaled so the total duration matches the
    original, which keeps the opponent honest: they take exactly as long as she did, they
    simply do not move like her inside that time.
    """
    ev = [dict(e) for e in trace.events]
    if len(ev) < 3:
        return mtf.Trace({**trace.meta, "session_id": session_id}, ev)
    d = [g for g in gaps(donor.events) if g > 0] or [1.0]
    target = trace.duration_ms or sum(gaps(ev)) or 1.0
    want = d * ((len(ev) - 1) // len(d) + 1)
    want = want[:len(ev) - 1]
    total = sum(want) or 1.0
    want = [g * target / total for g in want]
    t = ev[0]["t"]
    for e, g in zip(ev[1:], want):
        t += g
        e["t"] = t
    return mtf.Trace({**trace.meta, "subject": "watcher", "session_id": session_id}, ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--scored", choices=("all", "dynamics", "shape"), default="all",
                    help="which features may decide; 'dynamics' is the experiment: score only "
                         "the parts of a session that a person in the room cannot see")
    a = ap.parse_args()
    if a.scored == "dynamics":
        F.SCORED = DYNAMICS
    elif a.scored == "shape":
        F.SCORED = SHAPE
    thr = model.FROZEN_THRESHOLD if a.threshold is None else a.threshold

    _, by = E.load(a.data)
    report = {"scored_mode": a.scored, "scored_features": list(F.SCORED),
              "shape_features": list(SHAPE), "dynamics_features": list(DYNAMICS), "styles": {}}
    print(f"scored = {a.scored} ({len(F.SCORED)} features) | SHAPE {len(SHAPE)} | DYNAMICS {len(DYNAMICS)}\n")

    for style in sorted({k[0] for k in by}):
        enrol = (by.get((style, "owner", "enrol"), [])
                 + by.get((style, "owner_tired", "enrol"), []))
        owner_test = by.get((style, "owner", "test"), []) + by.get((style, "owner_tired", "test"), [])
        donor = (by.get((style, "stranger", "enrol"), []) + by.get((style, "mimic", "enrol"), []))
        if not (enrol and owner_test and donor):
            continue
        groups = {s: [F.vector(t) for t in by.get((style, s, "enrol"), [])]
                  for s in E.POSITIVE + E.NEGATIVE if by.get((style, s, "enrol"))}
        # a restricted scored set gets its threshold re-derived from enrolment alone, by the
        # same stated rule as ever: the 0.85 quantile of leave-one-out enrolment scores
        thr_s, ev_s = (thr, []) if a.scored == "all" else model.calibrate(enrol, groups=groups)
        report["styles"].setdefault(style, {})["threshold"] = round(float(thr_s), 3)
        report["styles"][style]["threshold_evidence"] = [round(float(x), 3) for x in ev_s]
        profile = model.fit(enrol, groups=groups)
        print(f"=== {style} ===  (threshold {thr_s:.2f}"
              f"{' - re-derived from enrolment for this scored set' if a.scored != 'all' else ''})")

        # the owner, so the two opponents can be read against a working baseline
        base = [model.verify(profile, t, thr_s) for t in owner_test]
        print(f"   owner (baseline)      {sum(d.admitted for d in base)}/{len(base)} open")

        for label, traces in (
            ("replay (exact copy)", [mtf.Trace({**t.meta, "subject": "replay"},
                                        [dict(e) for e in t.events]) for t in owner_test]),
            ("watcher (shape only)", [retime_like(t, donor[i % len(donor)], f"{t.session_id}-watch")
                                      for i, t in enumerate(owner_test)]),
        ):
            dec = [model.verify(profile, t, thr_s) for t in traces]
            opened = sum(d.admitted for d in dec)
            # which families of feature were pushing the score up for the ones that got in
            tally = {}
            for d in dec:
                for name, z in d.top[:3]:
                    tally.setdefault(name, []).append(round(z, 1))
            blamed = sorted(((k, st.fmean(v), len(v)) for k, v in tally.items()),
                            key=lambda x: -x[1])[:6]
            print(f"   {label:<22} {opened}/{len(dec)} open"
                  f"   {'<- gets in' if opened else '   refused'}")
            print("      deciding features: "
                  + ", ".join(f"{k}[{kind(k)}]={v:.1f}z x{n}" for k, v, n in blamed))
            report["styles"].setdefault(style, {})[label.split()[0]] = {
                "opened": opened, "presented": len(dec),
                "false_accept_rate": opened / len(dec),
                "deciding_features": [{"feature": k, "kind": kind(k), "mean_z": round(v, 2), "times_top3": n}
                                      for k, v, n in blamed]}

        worst = report["styles"][style]["watcher"]
        verdict = ("REFUSED - the visible path alone is not enough" if worst["opened"] == 0
                   else f"GETS IN on {worst['opened']}/{worst['presented']} - see deciding features")
        print(f"   -> watcher verdict: {verdict}\n")

    if a.json_out:
        json.dump(report, open(a.json_out, "w"), indent=1)
        print(f"report -> {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
