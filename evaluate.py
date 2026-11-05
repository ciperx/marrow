"""Train and test Marrow on any recording set in Marrow Trace Format, offline.

Two things are measured, both on recordings the profile never trained on:

1. operating rates
   positives = the enrolled person: `owner` and `owner_tired` (same identity, worse day)
   negatives = `stranger` and `mimic`
   false-accept rate  = impostors let in / impostors
   false-reject rate  = the real person doubted / their sessions

2. the swap test
   enrol on somebody else entirely (`stranger`), then present the owner's recordings.
   A method that opens here is not identifying anybody, it is just checking that a
   recording looks like a recording. The number must be zero.

The threshold is read from model.FROZEN_THRESHOLD, chosen from enrolment data before this
script ever saw the test split. `--calibrate` re-derives it and prints the evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import features as F
import model
import mtf

POSITIVE = ("owner", "owner_tired")
NEGATIVE = ("stranger", "mimic")


def load(out_dir):
    man = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
    by = {}
    for row in man["sessions"]:
        tr = mtf.read(os.path.join(out_dir, row["path"]))
        by.setdefault((row["style"], row["subject"], row["split"]), []).append(tr)
    return man, by


def run(out_dir, threshold=None, quiet=False):
    man, by = load(out_dir)
    thr = model.FROZEN_THRESHOLD if threshold is None else threshold
    styles = sorted({k[0] for k in by})
    report = {"threshold": thr, "corpus": os.path.abspath(out_dir),
              "sessions": len(man["sessions"]), "styles": {}}
    all_dec = []

    for style in styles:
        # Enrolment deliberately holds the owner in more than one state: sessions from a
        # rested day AND from a tired day. That is what a real enrolment looks like, and it
        # is why a bad day later does not read as a different person.
        enrol = (by.get((style, "owner", "enrol"), [])
                 + by.get((style, "owner_tired", "enrol"), []))
        if not enrol:
            continue
        # weights come from the enrolment split of everybody in this style — never the test split
        groups = {}
        for subject in POSITIVE + NEGATIVE:
            vecs = [F.vector(t) for t in by.get((style, subject, "enrol"), [])]
            if vecs:
                groups[subject] = vecs
        profile = model.fit(enrol, groups=groups)
        dec = []
        for subject in POSITIVE + NEGATIVE:
            for tr in by.get((style, subject, "test"), []):
                dec.append(model.verify(profile, tr, thr))
        r = model.rates(dec, POSITIVE, NEGATIVE)

        swap_numbers = {}
        for impostor in ("stranger", "mimic"):
            other = by.get((style, impostor, "enrol"), [])
            if not other:
                continue
            swap = model.fit(other, groups=groups)
            opened = [model.verify(swap, tr, thr).admitted
                      for tr in by.get((style, "owner", "test"), [])]
            swap_numbers[impostor] = {"profile_trained_on": impostor,
                                      "owner_sessions_presented": len(opened),
                                      "owner_sessions_opened": sum(opened)}
        report["styles"][style] = {"enrolment_sessions": profile.sessions,
                                   "enrolment_composition": "owner (rested) + owner_tired", **r,
                                   "swap_test": swap_numbers}
        all_dec += dec
        if not quiet:
            print(f"\n=== {style} ===")
            print(f"enrolled on {profile.sessions} recordings of 'owner'; "
                  f"threshold {thr} (fixed before testing)")
            for d in dec:
                print("   " + d.line())
            print(f"   false-accept {r['false_accept_rate']:.3f} "
                  f"({r['accepted_impostors']}/{r['impostors']} impostor sessions let in)"
                  f"   false-reject {r['false_reject_rate']:.3f} "
                  f"({r['rejected_owners']}/{r['owner_sessions']} owner sessions doubted)")
            for imp, s in swap_numbers.items():
                print(f"   swap test vs {imp}: {s['owner_sessions_opened']}/"
                      f"{s['owner_sessions_presented']} owner sessions opened "
                      f"{'(pass)' if s['owner_sessions_opened'] == 0 else '(FAIL)'}")

    if all_dec:
        r = model.rates(all_dec, POSITIVE, NEGATIVE)
        report["overall"] = r
        swap_open = sum(s["owner_sessions_opened"]
                        for st in report["styles"].values()
                        for s in st["swap_test"].values())
        report["swap_test_owner_sessions_opened"] = swap_open
        report["swap_test_passed"] = swap_open == 0
        if not quiet:
            print("\n=== overall (both input styles pooled) ===")
            print(f"   false-accept rate {r['false_accept_rate']:.3f}  |  "
                  f"false-reject rate {r['false_reject_rate']:.3f}")
            print(f"   swap test: {swap_open} owner sessions opened under a profile "
                  f"trained on somebody else -> {'PASS' if swap_open == 0 else 'FAIL'}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="re-derive the operating point from enrolment and print it")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args()

    if a.calibrate:
        _, by = load(a.data)
        for style in sorted({k[0] for k in by}):
            enrol = by.get((style, "owner", "enrol"), [])
            if not enrol:
                continue
            groups = {s: [F.vector(t) for t in by.get((style, s, "enrol"), [])]
                      for s in POSITIVE + NEGATIVE if by.get((style, s, "enrol"))}
            thr, scores = model.calibrate(enrol, groups=groups)
            print(f"{style}: enrolment-only 0.85-quantile threshold = {thr}")
            print(f"   enrolment leave-one-out scores: "
                  f"{', '.join(f'{s:.2f}' for s in scores)}")
        print(f"   frozen in model.py as FROZEN_THRESHOLD = {model.FROZEN_THRESHOLD}")
        print("   the test split was not read to produce any of the above")

    rep = run(a.data, a.threshold)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"\nreport -> {a.json_out}")
    if not rep["overall"]["false_accept_rate"] == rep["overall"]["false_accept_rate"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
