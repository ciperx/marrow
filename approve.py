#!/usr/bin/env python3
"""Out-of-room approval: the gate that observation cannot defeat.

The measurements in this repo say something uncomfortable and specific. A recording of her
walking a fresh route is refused, because the route did not exist when the recording was made
(challenge.py, replay 0/6). But an impostor who is *standing in the room* while she
authenticates is admitted 6 of 6 in mouse, and the movement features do not separate them from
her anywhere she is also admitted. So the movement layer cannot hold the door. Something else
has to, and it must not be something that is visible in the room.

This file is that gate. The rule it implements: **the authority to approve must live off the
machine and off the room, and the only thing that crosses back is a value bound to a nonce
that did not exist a moment ago.**

    local machine  --issues-->  nonce + 4-character code            (visible in the room)
    off-room owner --signs-->   HMAC(key, nonce:code), single use   (not visible in the room)
    local machine  --verifies--> bound to this nonce, not used before

The secret never enters the room, so a room full of people watching her move learns nothing
that can be replayed. What she does with the joystick is *initiate* and *cancel* - movements
that can be copied, and whose copying now buys the copier nothing, because an approval they did
not sign is not an approval.

    python3 approve.py selftest      # the three claims, measured, no arguments
    python3 approve.py keygen        # writes remote.key (0600) - KEEP THIS OFF THE MACHINE
    python3 approve.py issue         # local machine prints a challenge, waits for the token
    python3 approve.py sign <nonce> <code>   # run on the off-room device / her phone
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = os.path.join(HERE, "remote.key")
STATE = os.path.join(HERE, "approval_state.json")
LEDGER = os.path.join(HERE, "approval_used.json")
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no O/0/I/1 - the code gets read aloud
CODE_LEN = 4


def _read(path, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default


def _write(path, obj, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh, indent=1)


def keygen(path=KEY):
    """The possession. Its whole value is that this file never sits on the machine."""
    if os.path.exists(path):
        raise SystemExit(f"{path} exists - refusing to overwrite a key that may already be in use")
    import base64
    k = secrets.token_bytes(32)
    _write(path, {"k": base64.b64encode(k).decode(), "created": int(time.time())})
    return path


def _key(path=KEY):
    import base64
    d = _read(path, None)
    if not d:
        raise SystemExit(f"no key at {path} - run `keygen` on the device that stays out of the room")
    return base64.b64decode(d["k"])


def token(key: bytes, nonce: str, code: str) -> str:
    """What the off-room device returns. Binds key + nonce + code; nothing else is accepted."""
    return hmac.new(key, f"{nonce}:{code}".encode(), hashlib.sha256).hexdigest()[:32]


def issue(path=STATE, ttl=180):
    """Local machine: a challenge that did not exist before this call."""
    nonce = secrets.token_hex(8)
    code = "".join(secrets.choice(ALPHABET) for _ in range(CODE_LEN))
    rec = {"nonce": nonce, "code": code, "issued": int(time.time()), "ttl": ttl}
    _write(path, rec)
    return rec


def verify(presented: str, nonce: str, path=STATE, key_path=KEY, now=None, ledger=LEDGER):
    """Refuse unless: the nonce is the one we just issued, it has not been used, it has not
    expired, and the token is the HMAC the off-room device would have produced."""
    st = _read(path, None)
    if not st or st.get("nonce") != nonce:
        return False, "nonce is not the one this machine issued"
    now = now if now is not None else int(time.time())
    if now - int(st.get("issued", 0)) > int(st.get("ttl", 180)):
        return False, "challenge expired"
    used = _read(ledger, {"used": []})
    if nonce in used["used"]:
        return False, "this nonce was already spent (single use)"
    want = token(_key(key_path), nonce, st["code"])
    if not hmac.compare_digest(str(presented or ""), want):
        return False, "token does not match - not signed with the off-room key"
    used["used"].append(nonce)
    _write(ledger, used)
    return True, "approved"


def selftest():
    """Three claims. Each one is a measurement, not an assertion."""
    import base64
    tmp = os.path.join(HERE, "cache_selftest")
    os.makedirs(tmp, exist_ok=True)
    kp = os.path.join(tmp, "remote.key")
    if os.path.exists(kp):
        os.remove(kp)
    keygen(kp)
    key = _key(kp)
    st_path = os.path.join(tmp, "state.json")
    led = os.path.join(tmp, "used.json")
    if os.path.exists(led):
        os.remove(led)

    out = []
    # 1. the owner, holding the key, gets in
    st = issue(st_path)
    ok, why = verify(token(key, st["nonce"], st["code"]), st["nonce"], st_path, kp, ledger=led)
    out.append(("owner with the off-room key", ok, why))

    # 2. the same approval offered a second time - refused, single use
    ok2, why2 = verify(token(key, st["nonce"], st["code"]), st["nonce"], st_path, kp, ledger=led)
    out.append(("the same token, replayed", ok2, why2))

    # 3. somebody in the room: they watched the screen, so they know nonce and code.
    #    They have no key, so the best they can do is guess or replay. Both are refused.
    st3 = issue(st_path)
    watched_nonce, watched_code = st3["nonce"], st3["code"]
    guess = hashlib.sha256(f"{watched_nonce}:{watched_code}".encode()).hexdigest()[:32]  # no key
    ok3, why3 = verify(guess, watched_nonce, st_path, kp, ledger=led)
    out.append((f"an impostor who saw nonce+code ({watched_code})", ok3, why3))

    # 4. yesterday's approval against today's challenge
    st4 = issue(st_path)
    old = token(key, st3["nonce"], watched_code)
    ok4, why4 = verify(old, st4["nonce"], st_path, kp, ledger=led)
    out.append(("an approval recorded for an older nonce", ok4, why4))

    print("out-of-room approval - measured, not asserted\n")
    for label, ok, why in out:
        print(f"  {'ADMITTED' if ok else 'REFUSED ':>8}  {label:<46} {why}")
    print("\n  what this does not solve: whoever holds the key. That is a possession problem with")
    print("  a physical answer (the key stays on a device that stays out of the room), not a")
    print("  statistical one - which is exactly why it is the right thing to put at the gate.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "keygen", "issue", "verify", "sign"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "keygen":
        print("key written to", keygen())
        print("MOVE IT OFF THIS MACHINE. On the machine it protects, it protects nothing.")
        return 0
    if a.cmd == "issue":
        st = issue()
        print(f"challenge  nonce={st['nonce']}  code={st['code']}  ttl={st['ttl']}s")
        print("read the code to the device that stays out of the room, then paste its token:")
        got = sys.stdin.readline().strip()
        ok, why = verify(got, st["nonce"])
        print(("APPROVED" if ok else "REFUSED"), "-", why)
        return 0 if ok else 1
    if a.cmd == "sign":
        nonce, *rest = a.args
        st = _read(STATE, None)
        code = rest[0] if rest else (st or {}).get("code")
        print(token(_key(), nonce, code))
        return 0
    nonce, presented = a.args[0], a.args[1]
    ok, why = verify(presented, nonce)
    print(("APPROVED" if ok else "REFUSED"), "-", why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
