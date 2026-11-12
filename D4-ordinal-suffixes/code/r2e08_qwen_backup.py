"""R2 exploration E08 (CHARTER-2.md question 2), Qwen3-0.6B-Base: are L2H4 and L5H2 the second (backup) route?
Items: 96 teen and 96 non-teen at 2, 3 and 4 digits (E02's seeds for 2 digits, E01's for 3 and 4).
Mean ablation at the last token of: A = L0H2+L1H6; A+L2H4; A+L5H2; A+L2H4+L5H2 (A+B); B = L2H4+L5H2 alone;
and 20 random controls A + 2 random heads from layers 2-6 (not L2H4, L5H2). Teen effect, pull, AUROC, teen accuracy.
Also residual patching under A at 3 digits (layers 0-9, last and tens position), as E02 did at 2 digits.
Writes outputs/r2/e08_<model>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

NAME = "qwen3-06b"; PER = 96; NR = 20
M = Hooked(NAME); L, Hn = M.L, M.H
A = [(0, 2), (1, 6)]; B = [(2, 4), (5, 2)]
pool = [(l, h) for l in range(2, 7) for h in range(Hn) if (l, h) not in B]
res = dict(model=NAME, digits={}); t0 = time.time()
for d in (2, 3, 4):
    items = make_items(d, PER, f"d4r2-e02-{d}" if d == 2 else f"d4r2-e01-{d}")
    M.clear(); clean = M.run(items, cap=("oin",)); means = {l: v.mean(0) for l, v in M.c["oin"].items()}
    R = dict(base=summary(clean, items), sets={}, random=[])
    for k, S in {"A": A, "A+L2H4": A + [(2, 4)], "A+L5H2": A + [(5, 2)], "A+B": A + B, "B": B}.items():
        M.clear(); M.ablate_heads(S, means); R["sets"][k] = summary(M.run(items), items)
    rng = random.Random(f"d4r2-e08-{d}")
    for _ in range(NR):
        M.clear(); M.ablate_heads(A + rng.sample(pool, 2), means); R["random"].append(summary(M.run(items), items))
    if d == 3:
        tgt = [it for it in items if it["cls"] == "teen"]; rng2 = random.Random("d4r2-e08-src")
        src = []
        for it in tgt:
            n = it["n"]; ns = n - (n // 10 % 10) * 10 + rng2.choice(range(2, 10)) * 10
            src.append(dict(it, n=ns, cls="end123", prefix=it["prefix"][: len(it["prefix"]) - len(str(n))] + str(ns)))
        M.clear(); M.ablate_heads(A, means); M.offsets = (0, 1)
        lt = M.run(tgt); ls = M.run(src, cap=("residoff",)); SC = M.c
        mt, ms = mvals(lt, tgt), mvals(ls, src); gap = float((ms - mt).sum()); P = {}
        for l in range(10):
            for k in (0, 1):
                M.rpatch = {l: (k, SC[f"resid-{k}"][l])}; P[f"{l}|{k}"] = float((mvals(M.run(tgt), tgt) - mt).sum() / gap)
        M.clear(); R["patch_A"] = P
    res["digits"][str(d)] = R
    json.dump(res, open(OUT2 / f"e08_{NAME}.json", "w"), default=float)
    b = R["base"]; rt = [x["teen"] / b["teen"] for x in R["random"]]
    print(f"== {d} digits ({time.time() - t0:.0f}s) teen {b['teen']:+.2f}; kept: " + " ".join(f"{k} {v['teen'] / b['teen']:.2f} (acc {v['acc_teen']:.2f})" for k, v in R["sets"].items())
          + f"; A+random pair {min(rt):.2f}-{max(rt):.2f}")
    if d == 3: print("  patch A, 3 digits:", " ".join(f"L{l}:{R['patch_A'][f'{l}|0']:+.2f}/{R['patch_A'][f'{l}|1']:+.2f}" for l in range(10)))
print(f"done {time.time() - t0:.0f}s")
