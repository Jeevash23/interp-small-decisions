"""Exploratory, no model weights: a bias-free teen measure from exp03 outputs.
For each item, m = logit(last-digit suffix: st/nd/rd for last digit 1/2/3) - logit(th).
Teen effect = mean m over non-teen items ending in 1-3 minus mean m over teen items (x11-x13),
per (digits, form, whether the last token holds the tens digit: in/before). Large when the model reads the tens digit; near zero when it ignores it.
A pure 'th' bias lowers both means equally and leaves the effect unchanged.
Also: AUROC of m separating non-teen (positive) from teen items.
Usage: teen_effect.py <model> [...]   -> outputs/teen_effect_<model>.json"""
import sys, json
from collections import defaultdict
from common import *


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return float("nan")
    r = np.concatenate([pos, neg]).argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(name, fname=None, keyf=None):
    rows = [json.loads(l) for l in open(OUT / (fname or f"exp03_{name}.jsonl"))]
    G = defaultdict(lambda: dict(teen=[], end123=[]))
    for r in rows:
        if r["cls"] not in ("teen", "end123"): continue
        lg = r["logits"]; d = r["n"] % 10
        m = lg[d - 1] - lg[3]
        last_in = "in" if sum(c.isdigit() for c in r["last_tok"]) >= 2 else "before"
        G[keyf(r) if keyf else (r["digits"], r["form"], last_in)][r["cls"]].append(m)
    out = {}
    for k in sorted(G):
        t, e = G[k]["teen"], G[k]["end123"]
        out["|".join(map(str, k))] = dict(n_teen=len(t), n_end123=len(e), mean_m_end123=float(np.mean(e)),
                                         mean_m_teen=float(np.mean(t)), effect=float(np.mean(e) - np.mean(t)), auroc=auroc(e, t))
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        o = run(name)
        save_json(o, f"teen_effect_{name}.json")
        print("==", name)
        for k, v in o.items():
            print(f"  {k:10s} m(end123)={v['mean_m_end123']:+6.2f} m(teen)={v['mean_m_teen']:+6.2f} effect={v['effect']:+6.2f} auroc={v['auroc']:.3f}")
