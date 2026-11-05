"""Fake-data generator for Marrow.

Produces MTF v1 recordings only — nobody's real movement data is used or needed.
Two input styles, and in each style four subjects:

  owner        the enrolled person
  owner_tired  the same person on a bad day (identity unchanged, state shifted)
  stranger     somebody else entirely
  mimic        a person who has watched the owner and copies the visible habits
               (their pointer speed and typing rate) without the micro-dynamics

The mimic is the interesting case: a verifier that only compares averages will let it
in, which is the point of testing it.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, asdict

import mtf

STYLES = ("mouse_keyboard", "gaze_dictation")


@dataclass
class Params:
    # pointer
    speed: float = 0.00055        # fraction of screen per ms
    tremor: float = 0.0042        # jitter amplitude on each move sample
    pause_mean: float = 260.0     # ms of no movement between strokes
    curvature: float = 0.35       # 0 = dead straight, 1 = very wandering
    dir_change: float = 0.30      # chance a stroke reverses direction mid-way
    # clicks
    click_gap_mean: float = 900.0
    # text
    iki_mean: float = 138.0       # ms between text events
    iki_cv: float = 0.42          # rhythm irregularity (dispersion / mean)
    burst_mean: float = 7.0       # text events back to back
    correction_rate: float = 0.055  # corrections per text event
    dictation_ratio: float = 0.0  # share of text that arrives by dictation
    onscreen_ratio: float = 0.0   # share that arrives by on-screen keyboard
    # session shape
    strokes: int = 46
    text_events: int = 150

    def scaled(self, **kw):
        d = asdict(self)
        d.update(kw)
        return Params(**d)


# --- the subjects -----------------------------------------------------------
BASE = {
    "mouse_keyboard": Params(speed=0.00062, tremor=0.0038, pause_mean=230, curvature=0.30,
                             dir_change=0.26, click_gap_mean=820, iki_mean=132, iki_cv=0.44,
                             burst_mean=8.0, correction_rate=0.05, dictation_ratio=0.0),
    "gaze_dictation": Params(speed=0.00024, tremor=0.0090, pause_mean=520, curvature=0.62,
                             dir_change=0.45, click_gap_mean=1700, iki_mean=210, iki_cv=0.55,
                             burst_mean=13.0, correction_rate=0.11,
                             dictation_ratio=0.80, onscreen_ratio=0.12, strokes=34,
                             text_events=120),
}

# a second person, for the stranger seat
OTHER = {
    "mouse_keyboard": Params(speed=0.00041, tremor=0.0019, pause_mean=380, curvature=0.14,
                             dir_change=0.12, click_gap_mean=1450, iki_mean=196, iki_cv=0.26,
                             burst_mean=4.5, correction_rate=0.018, dictation_ratio=0.0),
    "gaze_dictation": Params(speed=0.00051, tremor=0.0051, pause_mean=300, curvature=0.25,
                             dir_change=0.66, click_gap_mean=1100, iki_mean=150, iki_cv=0.31,
                             burst_mean=6.0, correction_rate=0.035,
                             dictation_ratio=0.55, onscreen_ratio=0.30, strokes=40,
                             text_events=140),
}


def _copycat(style):
    """Copies the owner's averages (speed, typing rate) but not the micro-dynamics."""
    o = BASE[style]
    return o.scaled(
        speed=o.speed * 1.02,
        iki_mean=o.iki_mean * 0.97,
        tremor=o.tremor * 0.34,         # steadier hand than the owner
        iki_cv=o.iki_cv * 0.55,         # metronome-like rhythm
        curvature=o.curvature * 0.55,
        dir_change=o.dir_change * 0.5,
        correction_rate=o.correction_rate * 0.25,
        pause_mean=o.pause_mean * 0.72,
    )


def subjects(style):
    o = BASE[style]
    return {
        "owner": o,
        # A tired day is a real but moderate shift: noticeably slower, longer pauses,
        # more corrections. The first version pushed speed to 0.55x and pauses to 2.2x,
        # which is not a tired person, it is a different person — and no method that
        # compares absolute speeds can both admit that and refuse a stranger.
        "owner_tired": o.scaled(speed=o.speed * 0.78, pause_mean=o.pause_mean * 1.45,
                                iki_mean=o.iki_mean * 1.22,
                                correction_rate=o.correction_rate * 1.70,
                                tremor=o.tremor * 1.08),
        "stranger": OTHER[style],
        "mimic": _copycat(style),
    }


# --- one session ------------------------------------------------------------
WORDS = ("the quick brown fox jumps over a lazy dog while the rain keeps falling "
         "and nobody in the room says a word about any of it at all").split()


def _stroke_events(rng, p, t, x, y, out, clicks):
    """One pointer stroke toward a target, with tremor and a possible reversal."""
    tx, ty = rng.random(), rng.random()
    dist = math.hypot(tx - x, ty - y)
    if dist < 1e-6:
        return t, x, y
    dur = max(40.0, dist / max(p.speed, 1e-6) * rng.uniform(0.75, 1.35))
    steps = max(3, int(dur / rng.uniform(14, 26)))
    sign = 1.0
    for i in range(1, steps + 1):
        u = i / steps
        ease = u * u * (3 - 2 * u)                       # smooth approach
        if p.dir_change and 0.25 < u < 0.6 and rng.random() < p.dir_change * 0.35:
            sign = -sign * rng.uniform(0.15, 0.5)        # overshoot / correction
        wob = math.sin(u * math.pi * rng.uniform(1.5, 3.5)) * p.curvature * 0.05
        nx = x + (tx - x) * ease + wob * rng.uniform(-1, 1) + sign * 0.0
        ny = y + (ty - y) * ease + wob * rng.uniform(-1, 1)
        nx += rng.gauss(0, p.tremor)
        ny += rng.gauss(0, p.tremor)
        t += int(rng.uniform(11, 27))
        out.append({"t": t, "type": "move",
                    "x": round(min(1.0, max(0.0, nx)), 4),
                    "y": round(min(1.0, max(0.0, ny)), 4)})
        t += int(max(0.0, rng.gauss(p.pause_mean * 0.16, p.pause_mean * 0.05)))
    t += int(max(0.0, rng.gauss(p.pause_mean * 0.7, p.pause_mean * 0.25)))
    out.append({"t": t, "type": "click", "x": round(tx, 4), "y": round(ty, 4), "button": "left"})
    clicks.append(t)
    return t, tx, ty


def _text_events(rng, p, t, out):
    """A burst of text in the style's mix, with corrections at the owner's rhythm."""
    label = ("dictation" if rng.random() < p.dictation_ratio
             else "onscreen" if rng.random() < p.onscreen_ratio
             else "typed")
    for _ in range(max(1, int(rng.gauss(p.burst_mean, p.burst_mean * 0.35)))):
        t += int(max(20.0, rng.gauss(p.iki_mean, p.iki_mean * p.iki_cv)))
        if rng.random() < p.correction_rate:
            out.append({"t": t, "type": "text", "label": label, "text": "", "correction": True})
            t += int(max(30.0, rng.gauss(p.iki_mean * 1.4, p.iki_mean * 0.4)))
        out.append({"t": t, "type": "text", "label": label,
                    "text": rng.choice(WORDS)})
    t += int(max(60.0, rng.gauss(p.pause_mean * 1.6, p.pause_mean * 0.6)))
    return t


def make_session(style, subject, params, session_id, seed):
    rng = random.Random(seed)
    events, t, x, y = [], 0, rng.random(), rng.random()
    clicks = []
    text_done = 0
    for _ in range(params.strokes):
        t, x, y = _stroke_events(rng, params, t, x, y, events, clicks)
        if rng.random() < 0.55:
            t = _text_events(rng, params, t, events)
            text_done += 1
            if len([e for e in events if e["type"] == "text"]) > params.text_events:
                break
    if not clicks:
        events.append({"t": t + 100, "type": "click", "x": round(x, 4), "y": round(y, 4)})
    meta = {"format": mtf.FORMAT, "version": mtf.VERSION, "session_id": session_id,
            "subject": subject, "input_style": style, "screen": {"w": 1920, "h": 1080},
            "generator": "marrow-generate_fake"}
    return mtf.Trace(meta=meta, events=events)


def build(out_dir, enrolment=6, test=4, seed=20260101):
    """Write the whole fake corpus. Deterministic for a given seed."""
    manifest = []
    for style in STYLES:
        subs = subjects(style)
        for subject, params in subs.items():
            for split, count in (("enrol", enrolment), ("test", test)):
                for i in range(count):
                    sid = f"{style}-{subject}-{split}-{i:02d}"
                    seed_i = abs(hash((seed, sid))) % (2 ** 31)
                    tr = make_session(style, subject, params, sid, seed_i)
                    path = os.path.join(out_dir, style, f"{sid}.jsonl")
                    mtf.write(tr, path)
                    manifest.append({"session_id": sid, "style": style, "subject": subject,
                                     "split": split, "path": os.path.relpath(path, out_dir),
                                     "tired": subject == "owner_tired", "events": len(tr.events)})
    import json
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"note": "fake data only; no real movement recordings anywhere",
                   "enrolment_per_subject": enrolment, "test_per_subject": test,
                   "sessions": manifest}, fh, indent=1)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Generate the Marrow fake recording set.")
    ap.add_argument("--out", default="data", help="output directory (default: data)")
    ap.add_argument("--enrolment", type=int, default=6)
    ap.add_argument("--test", dest="test_n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260101)
    a = ap.parse_args()
    m = build(a.out, a.enrolment, a.test_n, a.seed)
    styles = sorted({r["style"] for r in m})
    subs = sorted({r["subject"] for r in m})
    print(f"wrote {len(m)} recordings to {a.out}/")
    print(f"  styles:   {', '.join(styles)}")
    print(f"  subjects: {', '.join(subs)}")
    print(f"  note:     fake data only; the manifest says so at {a.out}/manifest.json")


if __name__ == "__main__":
    main()
