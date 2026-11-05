# Marrow

Decide whether the person at the computer is the enrolled user, using only what the
operating system already sees: **pointer movement, clicks, and text as it arrives** —
typed, on-screen keyboard, or inserted by dictation software (including the pattern of
corrections).

No audio. No camera. No assumption about which input device is in use. Written for people
who run eye gaze, head mice, switch access, sip-and-puff and lip-operated joysticks, where
the hand that is "rested" on one day does not move the same way on the next.

```
python3 run.py
```

That generates the fake recording set, enrols the person, freezes the threshold, trains
and tests, and prints the operating rates and the swap test. Python 3.9+, standard library
only, offline. No capture software and no real movement data anywhere in this repository.

## What it does on the shipped fake data

Two input styles, four subjects in each: the owner, the owner on a tired day, a stranger,
and a **mimic** who has watched the owner and copies the visible habits.

| | false-accept (impostors let in) | false-reject (owner doubted) | swap test |
|---|---|---|---|
| `mouse_keyboard` | **0.000** (0/8) | **0.000** (0/8) | **PASS** (0/4) |
| `gaze_dictation` | **0.000** (0/8) | **0.000** (0/8) | **PASS** (0/4) |
| pooled | **0.000** | **0.000** | **PASS** (0/36 opened) |

The **swap test** is the one that matters: the profile is trained on somebody else
entirely, then the owner's own recordings are presented to it. A method that opens here is
not identifying anybody, it is only checking that a recording looks like a recording. It
must open none, and it opens none.

Threshold: **3.26**, derived from the enrolment recordings alone by a stated rule (the 0.85
quantile of the enrolled person's own leave-one-out scores) and then frozen in
`model.py` before any test recording is scored. `python3 run.py --calibrate` re-prints the
twelve scores that produced it. No number in this repository was fitted to the test split.

## How it works

**One-class by necessity.** Enrolment gives recordings of a single person and no
negatives, so the model stores, for each feature, that person's typical value (median) and
their spread (MAD), and judges a new recording by how many robust standard deviations away
it sits.

**Feature weights** are learned from the enrolment split only, by the ratio of
between-person variance to within-person variance — a feature that separates people is
trusted, a feature that merely wanders is not. Sixteen enrolment recordings are enough.

**Top-5 scoring** rather than an average, so a few genuinely identifying features decide
instead of being outvoted by uninformative ones.

**Scale-free features decide.** Raw speed, raw tremor and raw pause length are computed and
reported but are *not* scored. A tired hand is slower, and a verifier that scores absolute
speed locks the real user out exactly as readily as it keeps the impostor out. What is
scored are rates per event and ratios between the person's own measurements —
`tremor_ratio` (shakiness per unit of speed), `latency_ratio` (correction lag relative to
the person's own typing rhythm), `iki_cv`, `speed_cv`, `path_efficiency`, and similar.

**Enrolment covers more than one state.** The profile is built from sessions recorded on a
rested day *and* on a tired day, which is what a real enrolment looks like. This is the
single largest contributor to the tired owner being let in.

**A declared state margin.** Each spread is widened by 1.30 before any distance is
computed, covering the states enrolment has no sample of. It is a statement about how much
a person may vary between days, not a number tuned on the test set.

These four decisions are not decoration. The first version of this code scored every
feature equally, used a median, and scored absolute speeds — measured false-accept 0.438,
false-reject 0.500, swap test fail (9 owner recordings opened). The measurements that
drove each change are in the commit messages and in the comments beside every constant.

## Testing it on your own recordings

`train_test.py` is the general entry point. It needs no manifest and no index: it reads the
`subject` and `input_style` fields that every recording already carries.

```bash
# enrol on this person (rested and tired sessions both welcome), test against a mixed pile
python3 train_test.py --style mouse_keyboard \
    --enrol  recordings/owner/  --test recordings/ \
    --swap-with recordings/stranger/

# score one new session against an existing profile
python3 train_test.py --enrol recordings/owner/ --verify recordings/new_session.jsonl
```

A profile is per input style. A gaze user's profile is not a mouse profile; mixing them in
one enrolment makes every stranger look close, which is easy to see in the numbers and
which `--style` prevents.

## The recording format

Marrow Trace Format v1 — see [format.md](format.md). Newline-delimited JSON, one header
line and then events: millisecond timestamps, pointer positions **as a fraction of the
screen** (so a recording reads the same on any screen), clicks, and text events labelled by
how they arrived. Nothing in it comes from a particular browser, operating system or
application, and a reader ignores fields it does not know.

```jsonl
{"format":"marrow-trace","version":1,"session_id":"s-0001","subject":"owner","input_style":"mouse_keyboard","screen":{"w":1920,"h":1080}}
{"t":120,"type":"move","x":0.2113,"y":0.4402}
{"t":410,"type":"click","x":0.2240,"y":0.4390,"button":"left"}
{"t":900,"type":"text","label":"typed","text":"hello"}
{"t":1310,"type":"text","label":"typed","text":"","correction":true}
```

## Files

| file | what it is |
|---|---|
| `format.md` | the recording format, in full |
| `mtf.py` | reader and writer for it |
| `generate_fake.py` | the synthetic corpus: two styles, four subjects each |
| `features.py` | the numbers taken from a recording, and which of them may decide |
| `model.py` | profile, weights, scoring, the frozen threshold |
| `evaluate.py` | the shipped corpus, train and test |
| `train_test.py` | any recording set, no manifest needed |
| `run.py` | one command for everything |

## Limits, stated plainly

- These numbers come from **fake data**. The simulator separates the subjects on purpose;
  real sessions carry drift, device changes and noise that it does not, so real
  false-accept and false-reject rates will be worse than the table above. The corpus is
  shipped so the claim can be checked and argued with, not because it settles anything.
- The tired owner is modelled as a moderate shift (speed ×0.78, pauses ×1.45, corrections
  ×1.70). An earlier version used ×0.55 and ×2.20, which is not a tired person but a
  different one, and no method can both admit that and refuse a stranger. The honest
  reading is that the method tolerates a moderate bad day, and will still refuse somebody
  whose condition changes more than that within a session.
- Eleven scored features over twelve enrolment recordings is not a lot of data. The
  weights in particular are estimated from a small number of people and would move with
  more of them.
- No claim is made about liveness. A recording replayed is a recording.

## Licence

MIT.

## The shoulder-surfer test (and why the fallback cannot be behavioural)

The sponsor's constraint says the fallback is part of the problem: her current password entry
is either repeatable by anyone in the room or too painful to keep long. So the obvious attack
on a movement verifier is somebody who watches her use it. `python3 shoulder_surf.py` measures
that instead of arguing about it, with two opponents built to be as close to the attacker as
fake data allows:

- **replay** - the session copied exactly, positions *and* timestamps, played back.
- **watcher** - the visible path copied exactly (every position, click, text label and
  correction) but paced with a stranger's rhythm, rescaled so the watcher spends the same
  total time. That is the fair version: the copier is neither obviously fast nor slow.

| scored set | owner | replay | watcher (gaze) | watcher (mouse) |
|---|---|---|---|---|
| all 11 features (shipped method) | 8/8 open | **8/8 open** | 7/8 open | 5/8 open |
| only the 4 invisible dynamics features | 8/8 open | **8/8 open** | 5/8 open | 8/8 open |

Two results, and both are bad for the method, which is why they are here.

**One: a replay gets in, every time, 8 of 8, under both scored sets.** This is not a bug to
tune away. There is nothing in a replayed recording that is not in the recording it replayed,
so a check that only measures movement cannot tell the two apart. `tremor_ratio` is a good
feature and it is zero help against a perfect copy, because a perfect copy has the owner's
tremor in it. **A fallback built on behaviour alone is therefore repeatable by anyone in the
room who can record her once**, which is exactly the failure the sponsor named.

**Two: restricting the score to the four invisible features did not fix the watcher, and made
the mouse case worse - 5/8 to 8/8.** The reason is in the table: the threshold has to be
re-derived from enrolment for the restricted set, and it drops (3.26 to 1.57 and 1.70)
because her own dynamics vary that much between a rested and a tired session. The tolerance
that admits her on a bad day is the same tolerance that admits a copier who is merely within
the band. The shape features were doing real work; removing them did not remove the watcher,
it removed the margin.

**What this means for the design, stated plainly.** The behavioural layer can only ever make
the password *rarely needed*. It cannot be the gate, because it cannot survive a recording.
Whatever guards the vault needs a **fresh challenge tied to the moment** - something that did
not exist when the recording was made and cannot be pre-recorded, so that a replay has
nothing to answer. That is a design requirement, not a tuning parameter, and it is the honest
answer to the sponsor's constraint: the fallback is not a better password, it is a challenge
the room cannot anticipate. The next piece of this work is to build that challenge and check
that her dynamics accept it while a copied one does not.

Limits, unchanged and still true: fake data, a modelled tired day, and a mimic that imitates
visible habits rather than a caregiver with years of real sessions.

## The challenge, and what it does and does not buy

`python3 challenge.py` replaces "present a session" with a challenge. The verifier issues a
nonce - a route hash for a target path that did not exist a moment ago - the user walks it,
and anything whose nonce does not match is refused before the movement is looked at. Scoring
is restricted to the four invisible dynamics features, because the route is now identical for
everybody and shape carries no identity at all.

| | gaze_dictation | mouse_keyboard |
|---|---|---|
| owner on a fresh route | 0/6 | 6/6 |
| **watcher on the same fresh route** | 0/6 | **6/6** |
| replay: refused by the nonce check | **0/6** | **0/6** |
| replay: would the score have admitted it, without that check | **6/6** | **6/6** |

Three honest readings.

**One: the challenge removes the recording attack outright.** The replay is refused 0 of 6, and
the row under it is the proof that this is the nonce and not the statistics - the very same
recording is admitted 6 of 6 when the nonce check is removed. That is a real reduction, and it
is the answer to "repeatable by anyone in the room": a recording made yesterday cannot answer a
route issued today. This is not a tuning improvement, it is the deletion of an attacker.

**Two: it does not refuse an impostor who is actually present, and I will not pretend
otherwise.** The watcher passes the nonce check honestly, and once it does, the movement layer
does not separate them from the owner anywhere the owner is also admitted. In mouse the owner
opens 6 of 6 and the watcher opens 6 of 6. In gaze the watcher opens 0 of 6 - but so does the
owner, so that is a broken operating point and not discrimination. In this corpus there is no
evidence at all that the verifier tells her from a copier on a route neither has walked before.

**Three: the gaze operating point is worse than useless and should be reported as a failure.**
Her rhythm on a path she has never walked is more variable than the same rhythm on her own
path, and a threshold derived from her enrolment no longer admits her. A verifier built this
way would lock her out on the days it mattered.

**What follows for the design.** The nonce converts "a secret that can be repeated by anyone
who heard it" into "an impostor who must physically be here while she authenticates". That is a
real improvement to her threat model and it is not authentication. The residual attacker is
someone present, and no amount of movement statistics in this corpus refuses them. So the
fallback should not be a better thing to remember or a better way to move. It should be a
factor that is not *in the room* - something she actuates with the joystick she already uses,
but which is approved on a device the room cannot see, or by a person on a channel the room
cannot read. Behaviour stays where it belongs: making the password rarely needed, never
holding the door.

Limits unchanged: fake data, a modelled tired day, and a mimic that imitates visible habits
rather than a caregiver with years of real recordings.
