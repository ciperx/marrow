#!/usr/bin/env python3
"""Train and test Marrow on ANY recordings in Marrow Trace Format. Offline, stdlib only.

No manifest, no index file, no capture software, no network. Point it at recordings and
it works: labels come from the `subject` field that every MTF header already carries.

    # enrol on this person, test against a mixed pile, classify by the subject field
    python3 train_test.py --enrol recordings/owner/ --test recordings/

    # name the two sides yourself, when the labels are not self-explanatory
    python3 train_test.py --enrol recordings/owner/ --test recordings/ \
        --positive alice,alice_tired --negative bob,carol

    # score one recording against an existing profile
    python3 train_test.py --enrol recordings/owner/ --verify recordings/new_session.jsonl

    # also run the swap test: enrol on somebody else, present the owner's recordings
    python3 train_test.py --enrol recordings/owner/ --test recordings/ --swap-with recordings/bob/

Enrolment may contain several states of the same person (a rested day and a tired day) —
that is the intended use, and it is what keeps a bad day from reading as a stranger.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import features as F          # noqa: E402
import model                  # noqa: E402
import mtf                    # noqa: E402


def collect(paths):
    """Every .jsonl under the given files or directories (recursively)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                out += [os.path.join(root, n) for n in sorted(names) if n.endswith(".jsonl")]
        elif p.endswith(".jsonl"):
            out.append(p)
        else:
            raise SystemExit(f"not a recording or directory: {p}")
    return out


def load(paths, style=None):
    traces = []
    for p in collect(paths):
        try:
            tr = mtf.read(p)
        except ValueError as exc:
            print(f"  skipping {p}: {exc}", file=sys.stderr)
            continue
        if style is None or tr.style == style:
            traces.append(tr)
    return traces


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--style", default=None,
                    help="keep only recordings with this input_style. A profile is per input "
                         "style: a gaze user's profile is not a mouse profile, and mixing them "
                         "in one enrolment makes every stranger look close")
    ap.add_argument("--enrol", nargs="+", required=True,
                    help="enrolment recordings: files and/or directories (any state of the person)")
    ap.add_argument("--test", nargs="*", default=[],
                    help="recordings to test against the profile")
    ap.add_argument("--positive", default=None,
                    help="comma list of subject labels that ARE the enrolled person (default: "
                         "every label seen in the enrolment)")
    ap.add_argument("--negative", default=None,
                    help="comma list of subject labels that are impostors (default: every other "
                         "label present in the test set)")
    ap.add_argument("--swap-with", dest="swap", nargs="*", default=None,
                    help="enrol a second profile on THESE recordings (somebody else) and present "
                         "the enrolled person's test recordings to it. Must open none")
    ap.add_argument("--verify", nargs="*", default=None,
                    help="score these recordings against the enrolment profile and print the decision")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"operating point (default: the frozen {model.FROZEN_THRESHOLD})")
    ap.add_argument("--weights-from", choices=("enrol", "none"), default="enrol",
                    help="learn feature weights from the enrolment recordings (default), or "
                         "give every feature equal say")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args()

    enrol = load(a.enrol, a.style)
    if not enrol:
        raise SystemExit("no enrolment recordings found"
                           + (f" with input_style={a.style}" if a.style else ""))
    thr = model.FROZEN_THRESHOLD if a.threshold is None else a.threshold

    pos = set(a.positive.split(",")) if a.positive else {t.subject for t in enrol}
    groups = {}
    if a.weights_from == "enrol":
        for t in enrol:
            groups.setdefault(t.subject, []).append(F.vector(t))
    profile = model.fit(enrol, subject=sorted(pos)[0] if len(pos) == 1 else "enrolled",
                        groups=groups if len(groups) > 1 else None)

    print(f"enrolled on {len(enrol)} recordings: {', '.join(sorted({t.subject for t in enrol}))}")
    print(f"styles seen: {', '.join(sorted({t.style for t in enrol}))}")
    print(f"threshold {thr}  |  weights: {a.weights_from}")
    top = sorted(profile.weights.items(), key=lambda kv: -kv[1])[:4]
    print("heaviest features: " + ", ".join(f"{k}={v:.2f}" for k, v in top))
    report = {"enrolment": {"recordings": len(enrol), "subjects": sorted(pos), "threshold": thr},
              "profile": profile.to_dict(), "positives": sorted(pos)}

    if a.verify:
        print("\n-- verify --")
        for t in load(a.verify, a.style):
            d = model.verify(profile, t, thr)
            print("   " + d.line())
            report.setdefault("verify", []).append(
                {"session_id": d.session_id, "subject": d.subject, "score": d.score,
                 "admitted": d.admitted})

    if a.test:
        test = load(a.test, a.style)
        labels = {t.subject for t in test}
        neg = set(a.negative.split(",")) if a.negative else (labels - pos)
        print(f"\n-- test ({len(test)} recordings; positives {sorted(pos & labels) or 'none'}, "
              f"negatives {sorted(neg & labels) or 'none'}) --")
        dec = [model.verify(profile, t, thr) for t in test]
        for d in dec:
            if d.subject in pos or d.subject in neg:
                print("   " + d.line())
        r = model.rates(dec, pos, neg)
        print(f"   false-accept {r['false_accept_rate']:.3f} ({r['accepted_impostors']}/"
              f"{r['impostors']})   false-reject {r['false_reject_rate']:.3f} "
              f"({r['rejected_owners']}/{r['owner_sessions']})")
        report["test"] = {"recordings": len(test), "positive_labels": sorted(pos & labels),
                          "negative_labels": sorted(neg & labels), **r}

        if a.swap:
            other = load(a.swap, a.style)
            swap = model.fit(other, groups=groups if len(groups) > 1 else None)
            opened = [model.verify(swap, t, thr).admitted for t in test if t.subject in pos]
            print(f"   swap test (profile on {sorted({t.subject for t in other})}, "
                  f"{len(opened)} owner recordings presented): {sum(opened)} opened -> "
                  f"{'PASS' if sum(opened) == 0 else 'FAIL'}")
            report["swap_test"] = {"trained_on": sorted({t.subject for t in other}),
                                   "owner_recordings_presented": len(opened),
                                   "owner_recordings_opened": sum(opened),
                                   "passed": sum(opened) == 0}

    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nreport -> {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
