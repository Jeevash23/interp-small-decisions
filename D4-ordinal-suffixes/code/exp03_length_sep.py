"""Exp03 (exploratory, constructed inputs): suffix accuracy by number length (2-7 digits) and by form
(plain '1112' vs thousands separator '1,112'). Exploration frames (rankings, positions) and exploration
numbers only (charter split, extended by AMENDMENTS.md Amendment 1).
Usage: exp03_length_sep.py <model> [--per 120]
Writes outputs/exp03_<model>.jsonl (one line per item) and prints a table.
"""
import sys, random, json
from collections import defaultdict
from common import *

name = sys.argv[1]
PER = int(sys.argv[sys.argv.index("--per") + 1]) if "--per" in sys.argv else 120
rng = random.Random("d4-exp03")
frames = [f for f in explore_frames() if f["family"] in ("rankings", "positions")]


def draw(digits, cls):
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    while True:
        n = rng.randint(lo, hi)
        if cls == "teen": n = n - n % 100 + rng.choice([11, 12, 13])
        elif cls == "end123": n = n - n % 10 + rng.choice([1, 2, 3])
        elif cls == "th": n = n - n % 10 + rng.choice([0, 4, 5, 6, 7, 8, 9])
        if n < lo or n > hi or not is_explore_number(n): continue
        if cls == "end123" and n % 100 in (11, 12, 13): continue
        return n


def fmt(n, form):
    return f"{n:,}" if form == "comma" else str(n)


items = []
for digits in range(2, 8):
    for cls in ("teen", "end123", "th"):
        k = PER if cls != "th" else PER // 2
        for _ in range(k):
            n = draw(digits, cls); f = rng.choice(frames)
            for form in (("plain", "comma") if digits >= 4 else ("plain",)):
                items.append(dict(n=n, digits=digits, cls=cls, form=form, fid=f["fid"],
                                  prefix=f["text"].replace("{N}", fmt(n, form))))
tok, model = load(name)
cols = suffix_ids(tok)
lg, lse = last_logits(model, tok, [it["prefix"] for it in items], cols, bs=48)
T = defaultdict(list)
with open(OUT / f"exp03_{name}.jsonl", "w") as fo:
    for it, l, s in zip(items, lg, lse):
        ids = encode(tok, it["prefix"])
        last = tok.decode([ids[-1]])
        ok = SUFFIXES[int(np.argmax(l))] == suffix(it["n"])
        it.update(logits=[round(float(x), 4) for x in l], lse=round(float(s), 4), ok=bool(ok), last_tok=last)
        fo.write(json.dumps(it) + "\n")
        # does the last token contain the tens digit? (only digits counted)
        lastdig = "".join(c for c in last if c.isdigit())
        T[(it["digits"], it["form"], it["cls"], "tens_in_last" if len(lastdig) >= 2 else "tens_before")].append(ok)
print(f"{name}: {len(items)} items")
for k in sorted(T):
    print(f"  digits={k[0]} form={k[1]:5s} {k[2]:6s} {k[3]:13s} n={len(T[k]):4d} acc={np.mean(T[k]):.3f}")
