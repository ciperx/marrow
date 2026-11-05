#!/usr/bin/env python3
"""Marrow — one command to run everything.

    python3 run.py

Generates the fake recording set (if not there yet), enrols the owner from the
enrolment split alone, freezes the threshold, then trains and tests end to end and
prints the operating rates and the swap-test result. Stdlib only, offline, no capture
software, no real movement data.

    python3 run.py --fresh --enrol 8 --test 5      # regenerate with a bigger set
    python3 run.py --calibrate                     # show how the threshold was chosen
    python3 run.py --data my_recordings/           # test on YOUR recordings in MTF
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate                        # noqa: E402
import generate_fake                   # noqa: E402
import model                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--fresh", action="store_true", help="regenerate the fake corpus")
    ap.add_argument("--enrol", type=int, default=6)
    ap.add_argument("--test", dest="test_n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--json", dest="json_out", default=None)
    a = ap.parse_args()

    have = os.path.exists(os.path.join(a.data, "manifest.json"))
    if a.fresh or not have:
        m = generate_fake.build(a.data, a.enrol, a.test_n, a.seed)
        print(f"[1/4] generated {len(m)} fake recordings into {a.data}/")
    else:
        print(f"[1/4] using the recording set already at {a.data}/ (--fresh to rebuild)")

    if a.calibrate:
        _, by = evaluate.load(a.data)
        for style in sorted({k[0] for k in by}):
            enrol = by.get((style, "owner", "enrol"), [])
            if enrol:
                thr, scores = model.calibrate(enrol)
                print(f"[2/4] {style}: enrolment-only 0.90-quantile = {thr} "
                      f"(scores {', '.join(f'{s:.2f}' for s in scores)})")
        print(f"[2/4] frozen threshold used for the test split: {model.FROZEN_THRESHOLD}")
    else:
        print(f"[2/4] threshold {model.FROZEN_THRESHOLD} fixed from enrolment "
              f"before testing (--calibrate shows the evidence)")

    print("[3/4] training on the enrolment split and testing on unseen recordings")
    rep = evaluate.run(a.data)

    print("\n[4/4] summary")
    o = rep["overall"]
    print(f"   false-accept rate {o['false_accept_rate']:.3f} "
          f"| false-reject rate {o['false_reject_rate']:.3f}")
    print(f"   swap test: {'PASS' if rep['swap_test_passed'] else 'FAIL'} "
          f"({rep['swap_test_owner_sessions_opened']} owner sessions opened under a "
          f"profile trained on someone else)")
    if a.json_out:
        import json
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        print(f"   report -> {a.json_out}")
    return 0 if rep["swap_test_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
