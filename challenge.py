#!/usr/bin/env python3
"""A challenge tied to the moment, because behaviour alone cannot survive a recording.

What the shoulder-surfer test showed: a replay of one recorded session gets into the
verifier 8 of 8, under every scored set. Nothing can be tuned out of that, because there is
nothing in a replayed recording that is not in the recording it replayed. A check that only
measures movement is repeatable by anyone in the room who can record her once.

So the exchange stops being "present a session" and becomes a challenge:

    1. the verifier issues a nonce - a target path that did not exist a moment ago, hashed
       and handed to the user as the route to walk;
    2. the user walks it;
    3. the verifier refuses any response whose nonce does not match the one it just issued,
       before it looks at the movement at all.

That first step is what a replay cannot answer. It is not a better statistical test, it is
the removal of the attacker: a recording made yesterday carries yesterday's nonce.

What the nonce does NOT solve is the watcher - somebody in the room who watches her walk the
fresh path and then does it themselves. They pass the nonce check honestly. The only thing
left to refuse them with is the movement itself, so this script measures that honestly and
reports the number whether it is good or bad.

Run: python3 challenge.py [--data data] [--nonces 6] [--json challenge.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as E        # noqa: E402
import features as F        # noqa: E402
import generate_fake as G   # noqa: E402
import model                # noqa: E402
import mtf                  # noqa: E402
from shoulder_surf import DYNAMICS, SHAPE, retime_like   # noqa: E402


def nonce_of(trace):
    """The identity of a route: the positions, in order, hashed. Nothing about timing."""
    pts = "|".join(f"{round(e.get('x', 0), 6)},{round(e.get('y', 0), 6)}"
                   for e in trace.events if e.get("type") in ("move", "click"))
    return hashlib.sha256(pts.encode()).hexdigest()[:16]


def pose(trace, secs):
    """Stamp a route as the answer to this challenge, and record the nonce it answers."""
    ev = [dict(e) for e in trace.events]
    if ev:
        shift = 0.0
        for e in ev:
            e["t"] = e["t"] - shift
    meta = dict(trace.meta)
    meta["subject"] = secs["who"]
    meta["session_id"] = secs["id"]
    meta["nonce"] = secs["nonce"]
    return mtf.Trace(meta, ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--nonces", type=int, default=6)
    ap.add_argument("--scored", choices=("all", "dynamics"), default="dynamics")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args()
    if a.scored == "dynamics":
        F.SCORED = DYNAMICS

    _, by = E.load(a.data)
    report = {"scored": a.scored, "nonces": a.nonces, "styles": {}}
    print(f"challenge mode | scored = {a.scored} ({len(F.SCORED)} features) | {a.nonces} nonces\n")

    for style in sorted({k[0] for k in by}):
        enrol = by.get((style, "owner", "enrol"), []) + by.get((style, "owner_tired", "enrol"), [])
        donor_owner = by.get((style, "owner", "enrol"), []) + by.get((style, "owner_tired", "enrol"), [])
        donor_other = by.get((style, "stranger", "enrol"), []) + by.get((style, "mimic", "enrol"), [])
        old = by.get((style, "owner", "test"), [])
        if not (enrol and donor_owner and donor_other and old):
            continue
        groups = {s: [F.vector(t) for t in by.get((style, s, "enrol"), [])]
                  for s in E.POSITIVE + E.NEGATIVE if by.get((style, s, "enrol"))}
        thr, _ = model.calibrate(enrol, groups=groups)
        profile = model.fit(enrol, groups=groups)
        print(f"=== {style} ===  (threshold {thr:.2f} from enrolment)")

        rows = {"owner": [0, 0], "watcher": [0, 0], "replay_nonce": [0, 0], "replay_scored": [0, 0]}
        for n in range(a.nonces):
            seed = 90000 + n * 7919
            skeleton = G.make_session(style, "o", G.Params(), f"nonce-{style}-{n}", seed=seed)
            want = nonce_of(skeleton)
            # the owner walks the fresh route with her own rhythm
            hers = pose(retime_like(skeleton, donor_owner[n % len(donor_owner)], f"own-{n}"),
                        {"who": "owner", "id": f"own-{n}", "nonce": want})
            # a watcher in the room walks the same fresh route with somebody else's rhythm
            theirs = pose(retime_like(skeleton, donor_other[n % len(donor_other)], f"watch-{n}"),
                          {"who": "watcher", "id": f"watch-{n}", "nonce": want})
            # a recording from yesterday, offered as an answer to today's challenge
            past = dict(old[n % len(old)].meta)
            replay = mtf.Trace({**past, "subject": "replay"}, [dict(e) for e in old[n % len(old)].events])

            for label, t in (("owner", hers), ("watcher", theirs)):
                ok_nonce = t.meta.get("nonce") == want
                rows[label][1] += 1
                if ok_nonce and model.verify(profile, t, thr).admitted:
                    rows[label][0] += 1
            # the replay, twice: does the nonce check refuse it, and would scoring have?
            rows["replay_nonce"][1] += 1
            if replay.meta.get("nonce") == want:
                rows["replay_nonce"][0] += 1
            rows["replay_scored"][1] += 1
            if model.verify(profile, replay, thr).admitted:
                rows["replay_scored"][0] += 1

        for label, (ok, tot) in rows.items():
            print(f"   {label:<14} {ok}/{tot}")
        report["styles"][style] = {
            "threshold": round(float(thr), 2),
            **{k: {"opened": v[0], "presented": v[1]} for k, v in rows.items()},
            "nonce_defeats_replay": rows["replay_nonce"][0] == 0,
            "watcher_still_gets_in": rows["watcher"][0] > 0,
        }
        print(f"   -> nonce refuses the replay: {rows['replay_nonce'][0] == 0}"
              f" | watcher after the nonce: {rows['watcher'][0]}/{rows['watcher'][1]}\n")

    if a.json_out:
        json.dump(report, open(a.json_out, "w"), indent=1)
        print(f"report -> {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
