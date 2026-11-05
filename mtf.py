"""Reader and writer for Marrow Trace Format (MTF) v1.

Spec: format.md beside this file. Stdlib only, no capture software, offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

FORMAT = "marrow-trace"
VERSION = 1
TEXT_LABELS = ("typed", "onscreen", "dictation")
EVENT_TYPES = ("move", "click", "text")


@dataclass
class Trace:
    """One recording: a header plus ordered events."""

    meta: dict
    events: list = field(default_factory=list)

    # -- convenience -------------------------------------------------------
    @property
    def subject(self):
        return str(self.meta.get("subject", "unknown"))

    @property
    def style(self):
        return str(self.meta.get("input_style", "unknown"))

    @property
    def session_id(self):
        return str(self.meta.get("session_id", "unknown"))

    @property
    def duration_ms(self):
        return self.events[-1]["t"] if self.events else 0

    def of(self, *types):
        return [e for e in self.events if e.get("type") in types]


def write(trace, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(trace.meta, ensure_ascii=False, sort_keys=True) + "\n")
        for ev in trace.events:
            fh.write(json.dumps(ev, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read(path):
    """Read one MTF file. Unknown fields and unknown event types are ignored."""
    meta, events = None, []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not JSON: {exc}") from None
            if meta is None:
                if obj.get("format") != FORMAT:
                    raise ValueError(f"{path}: header missing format={FORMAT!r}")
                meta = obj
                continue
            etype = obj.get("type")
            if etype not in EVENT_TYPES:
                continue                      # rule 3: ignore what we do not know
            if not isinstance(obj.get("t"), (int, float)):
                continue
            events.append(obj)
    if meta is None:
        raise ValueError(f"{path}: empty recording, no header")
    events.sort(key=lambda e: e["t"])         # rule 1 is enforced leniently on read
    return Trace(meta=meta, events=events)


def load_dir(path, style=None):
    """Every .jsonl recording in a directory (non-recursive), optionally one style."""
    out = []
    for name in sorted(os.listdir(path)):
        if not name.endswith(".jsonl"):
            continue
        tr = read(os.path.join(path, name))
        if style is None or tr.style == style:
            out.append(tr)
    return out


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        tr = read(arg)
        print(f"{tr.session_id} subject={tr.subject} style={tr.style} "
              f"events={len(tr.events)} duration={tr.duration_ms}ms")
