"""Pair builder shared by exp02+ (same rules as exp01_patch.py): source/target numbers that differ in one digit,
in the same exploration frame, with the same token count and token split. Exploration numbers only."""
from common import part_frames, in_part, suffix, encode, number_pieces


def pairs(tok, kind, digits, N, rng, part="explore"):
    """part='conf' only under the frozen PROTOCOL.md (part_frames refuses otherwise)."""
    _FR = [f for f in part_frames(part) if f["family"] in ("rankings", "positions", "anniversaries")]
    out = []
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    _NUMS = set(n for n in range(lo, hi + 1) if in_part(n, part))
    cand = sorted(_NUMS)
    for _ in range(40000):
        t = rng.choice(cand); s = None
        tens = (t // 10) % 10
        if kind == "last":
            if tens == 1 or t % 10 not in (1, 2, 3, 4): continue
            s = t - t % 10 + rng.choice([d for d in (1, 2, 3, 4) if d != t % 10])
        elif kind == "intoteen":
            if tens != 1 or t % 10 not in (1, 2, 3): continue
            s = t + 10 * rng.randint(1, 8)
        elif kind == "outteen":
            if tens in (0, 1) or t % 10 not in (1, 2, 3): continue
            s = t - 10 * (tens - 1)
        if s not in _NUMS or suffix(s) == suffix(t): continue
        fs = [f for f in _FR if f["hi"] >= max(s, t)]
        f = rng.choice(fs)
        ps, pt = f["text"].replace("{N}", str(s)), f["text"].replace("{N}", str(t))
        if len(encode(tok, ps)) != len(encode(tok, pt)): continue
        if [len(x) for x in number_pieces(tok, ps, s)[1]] != [len(x) for x in number_pieces(tok, pt, t)[1]]: continue
        out.append((ps, pt, s, t))
        if len(out) == N: break
    return out
