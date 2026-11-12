"""Round-2 material rules (CHARTER-2.md, frozen 8 Nov 2026 16:54). Imports round-1 helpers from common.py.

r2_part(n) gives a number's round-2 status:
  'all'          1-99: in every part (all were read before; held2 items differ by frame only)
  'explore'      usable in round-2 exploration: round-1 exploration class (<= 7 digits), or explore2 (8-10 digits)
  'held_reused'  3-5 digits, round-1 conf or reserve class: used in held2 frames only (read before, new frames)
  'held2'        6-7 digits of round-1 reserve class (never read), or 8-10 digits with SHA-256('D4R2-num-<N>') % 10 >= 6
  'unused'       6-7 digits of round-1 conf class (read in round 1's confirmation; not used in round 2)
Held2 frames, 'held_reused' and 'held2' numbers are refused until PROTOCOL-2.md exists and is read-only.
"""
import json, hashlib, stat
from common import *

DATA2 = ROOT / "data2"
OUT2 = OUT / "r2"; OUT2.mkdir(exist_ok=True)


def _check2(name):
    want = dict(l.split()[::-1] for l in open(DATA2 / "SHA256SUMS"))
    assert want[name] == hashlib.sha256((DATA2 / name).read_bytes()).hexdigest(), f"{name} differs from its frozen digest"


def protocol2_frozen():
    p = ROOT / "PROTOCOL-2.md"
    return p.exists() and not (p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def require_protocol2(part):
    if part in ("explore", "explore2"): return
    if not protocol2_frozen():
        raise SystemExit(f"refusing to use {part}: PROTOCOL-2.md is not frozen")


def r2_part(n):
    if n < 100: return "all"
    d = len(str(n))
    if d <= 7:
        c = number_part_ext(n)
        if c == "explore": return "explore"
        if d <= 5: return "held_reused"
        return "held2" if c == "reserve" else "unused"
    assert d <= 10
    r = int(hashlib.sha256(f"D4R2-num-{n}".encode()).hexdigest(), 16) % 10
    return "explore" if r < 6 else "held2"


def r2_number_ok(n, part):
    """part 'explore': exploration numbers; part 'held': numbers for held2 items."""
    p = r2_part(n)
    if part == "explore": return p in ("all", "explore")
    require_protocol2("held")
    return p in ("all", "held_reused", "held2")


def r2_frames(part):
    """'explore': round-1 exploration frames + explore2 frames; 'held': held2 frames (after PROTOCOL-2.md is frozen)."""
    _check2("frames2.jsonl")
    f2 = [json.loads(l) for l in open(DATA2 / "frames2.jsonl")]
    if part == "explore":
        return explore_frames() + [f for f in f2 if f["part"] == "explore2"]
    require_protocol2("held")
    return [f for f in f2 if f["part"] == "held2"]
