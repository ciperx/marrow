# Marrow Trace Format (MTF) v1

A plain, documented, operating-system-neutral recording format. It is newline-delimited
JSON (one JSON object per line, `.jsonl`), UTF-8, LF endings. A recording is
self-describing: nothing about the machine, browser or app that captured it is required
to read it, and no field carries a device-specific identifier.

## Line 1 — header

```json
{"format":"marrow-trace","version":1,"session_id":"s-0001",
 "subject":"owner","input_style":"mouse_keyboard",
 "screen":{"w":1920,"h":1080},"started_utc":"2026-01-01T00:00:00Z"}
```

| field | required | meaning |
|---|---|---|
| `format` | yes | always `"marrow-trace"` |
| `version` | yes | `1` |
| `session_id` | yes | opaque id, unique inside a recording set |
| `subject` | yes | opaque label chosen by the recorder. The name carries no meaning to the method |
| `input_style` | yes | free-text label, e.g. `mouse_keyboard`, `gaze_dictation` |
| `screen` | yes | the pixel box the pointer positions were taken in. Used only to *normalise*; recorded positions are already fractions, so a reader may ignore it |
| `started_utc` | no | ISO-8601, informational only |

## Lines 2..N — events

Every event has `t` (integer milliseconds since the start of the session) and `type`.
No event references a device, window, application, URL or file path.

| `type` | fields | meaning |
|---|---|---|
| `move` | `x`,`y` | pointer position as a **fraction of the screen** in `[0,1]`; `0,0` top-left |
| `click` | `x`,`y`, optional `button` (`left`/`right`/`middle`) | a click at that position |
| `text` | `label`, `text`, optional `correction` (bool) | text entered, labelled by **how it arrived** |

`label` is one of exactly three values:

* `typed` — physical keyboard
* `onscreen` — on-screen keyboard
* `dictation` — inserted by software (speech-to-text and similar)

`correction: true` marks an event that *removes* or replaces already-entered text
(backspace, select-and-retype, a dictation re-do). It is the only place where `text` may
be empty.

## Rules

1. Timestamps are monotonic non-decreasing integers in milliseconds.
2. `x`,`y` are fractions, so a recording taken on any screen reads the same.
3. Readers must ignore unknown fields and unknown event types rather than fail.
4. A recording is one session. Sessions are separate files in a directory.

## Example

```jsonl
{"format":"marrow-trace","version":1,"session_id":"s-0001","subject":"owner","input_style":"mouse_keyboard","screen":{"w":1920,"h":1080}}
{"t":120,"type":"move","x":0.2113,"y":0.4402}
{"t":168,"type":"move","x":0.2240,"y":0.4390}
{"t":410,"type":"click","x":0.2240,"y":0.4390,"button":"left"}
{"t":900,"type":"text","label":"typed","text":"hello"}
{"t":1180,"type":"text","label":"typed","text":"world"}
{"t":1310,"type":"text","label":"typed","text":"","correction":true}
```
