"""Confirmation script 1 (PROTOCOL.md, tests B1-B4): the token-boundary account, behaviour only.
Usage: conf01_boundary.py <model> --part conf|explore [--per 200]

Frames: the part's frames in the rankings and positions families. Numbers: the part's numbers (charter split,
AMENDMENTS.md Amendment 1 above 9,999). With --part explore it is a pipeline check on exploration material.

  olmo2-1b : for each length 3-7 digits, `per` teen (x11-x13) and `per` non-teen numbers ending in 1-3, each in a
             random frame of the part; plain form for every length and, for 4-7 digits, the same item with a
             thousands separator ('1,112').
  gpt2, pythia-1b : 'boundary' set = every part number from 1,000 to 99,999 ending in 1-3 whose last token
             holds a single digit in this tokeniser (tens digit in the previous token), each in every frame;
             'control' set = `per` teen and `per` non-teen part numbers from 1,000 to 99,999 whose last token
             holds the tens digit, each in a random frame.
Per item: m = logit(last-digit suffix) - logit(th), whether the top suffix among the four is correct.
Writes outputs/<part>/conf01_<model>.jsonl and outputs/<part>/conf01_<model>_summary.json.
"""
import sys, random, json
from collections import defaultdict
from common import *
from teen_effect import auroc

name = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PART, PER = arg("--part", "explore"), int(arg("--per", "200"))
require_frozen_protocol(PART)
OD = OUT / PART; OD.mkdir(exist_ok=True)
rng = random.Random(f"d4-conf01-{PART}-{name}")
frames = [f for f in part_frames(PART) if f["family"] in ("rankings", "positions")]
from transformers import AutoTokenizer
tok0 = AutoTokenizer.from_pretrained(MODELS[name][0])


def tens_in_last(prefix):
    ids = encode(tok0, prefix)
    return sum(c.isdigit() for c in tok0.decode([ids[-1]])) >= 2


def draw(digits, cls):
    lo, hi = 10 ** (digits - 1), 10 ** digits - 1
    while True:
        n = rng.randint(lo, hi)
        n = n - n % 100 + rng.choice([11, 12, 13]) if cls == "teen" else n - n % 10 + rng.choice([1, 2, 3])
        if lo <= n <= hi and in_part(n, PART) and (cls == "teen" or n % 100 not in (11, 12, 13)):
            return n


items = []
if name == "olmo2-1b":
    for digits in range(3, 8):
        for cls in ("teen", "end123"):
            for _ in range(PER):
                n = draw(digits, cls); f = rng.choice(frames)
                for form in (("plain", "comma") if digits >= 4 else ("plain",)):
                    s = f"{n:,}" if form == "comma" else str(n)
                    items.append(dict(set="length", n=n, digits=digits, cls=cls, form=form, fid=f["fid"],
                                      prefix=f["text"].replace("{N}", s)))
else:
    f0 = frames[0]["text"]
    for n in range(1000, 100000):
        if n % 10 not in (1, 2, 3) or not in_part(n, PART): continue
        if tens_in_last(f0.replace("{N}", str(n))): continue
        cls = "teen" if n % 100 in (11, 12, 13) else "end123"
        for f in frames:
            items.append(dict(set="boundary", n=n, digits=len(str(n)), cls=cls, form="plain", fid=f["fid"],
                              prefix=f["text"].replace("{N}", str(n))))
    for cls in ("teen", "end123"):
        k = 0
        while k < PER:
            n = draw(rng.choice([4, 5]), cls); f = rng.choice(frames); p = f["text"].replace("{N}", str(n))
            if not tens_in_last(p): continue
            items.append(dict(set="control", n=n, digits=len(str(n)), cls=cls, form="plain", fid=f["fid"], prefix=p)); k += 1
for it in items:
    it["tens_in_last"] = tens_in_last(it["prefix"])

tok, model = load(name)
cols = suffix_ids(tok)
lg, lse = last_logits(model, tok, [it["prefix"] for it in items], cols, bs=48)
G = defaultdict(lambda: dict(teen=[], end123=[], ok_teen=[], ok_end123=[]))
with open(OD / f"conf01_{name}.jsonl", "w") as fo:
    for it, l in zip(items, lg):
        n = it["n"]; m = float(l[n % 10 - 1] - l[3]); ok = SUFFIXES[int(np.argmax(l))] == suffix(n)
        it.update(m=m, ok=bool(ok), logits=[round(float(x), 4) for x in l])
        fo.write(json.dumps(it) + "\n")
        key = f"{it['set']}|{it['digits']}|{it['form']}|{'in' if it['tens_in_last'] else 'before'}" if it["set"] == "length" \
            else f"{it['set']}|{'in' if it['tens_in_last'] else 'before'}"
        G[key][it["cls"]].append(m); G[key]["ok_" + it["cls"]].append(ok)
summ = {}
for k in sorted(G):
    g = G[k]
    summ[k] = dict(n_teen=len(g["teen"]), n_end123=len(g["end123"]),
                   auroc=auroc(g["end123"], g["teen"]) if g["teen"] and g["end123"] else None,
                   effect=float(np.mean(g["end123"]) - np.mean(g["teen"])) if g["teen"] and g["end123"] else None,
                   acc_teen=float(np.mean(g["ok_teen"])) if g["teen"] else None,
                   acc_end123=float(np.mean(g["ok_end123"])) if g["end123"] else None)
save_json(summ, f"{PART}/conf01_{name}_summary.json")
for k, v in summ.items():
    print(f"{k:28s} " + " ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
