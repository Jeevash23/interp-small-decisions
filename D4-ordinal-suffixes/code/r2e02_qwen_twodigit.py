"""R2 exploration E02 (CHARTER-2.md question 2), Qwen3-0.6B-Base: what carries the two-digit teen check that
survives mean ablation of L0H2+L1H6 at the last token?
Items: 96 teen (11-13) and 96 non-teen (21-93 ending 1-3) in exploration frames; also 3 digits for comparison.
  A = L0H2+L1H6 mean-ablated (last token).
  1. With A: each further head, each attention layer (all heads) and each MLP mean-ablated at the last token;
     kept = teen effect / teen effect under A.
  2. With and without A: residual patching from a non-teen source (same frame, same last digit, tens digit 2-9)
     into the teen target, by layer and position from the end (0 last digit, 1 tens digit, 2 space, 3 word before);
     fraction of the target's teen effect gap moved, ratio of sums.
  3. Attention from the last token to positions -1..-3 for every head (clean run).
Writes outputs/r2/e02_<model>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PER = int(arg("--per", "96")); NAME = arg("--model", "qwen3-06b"); DIG = [int(x) for x in arg("--digits", "2,3").split(",")]
A_HEADS = [(0, 2), (1, 6)]
M = Hooked(NAME); L, Hn = M.L, M.H
res = dict(model=NAME, digits={}); t0 = time.time()
for d in DIG:
    items = make_items(d, PER, f"d4r2-e02-{d}")
    M.clear(); clean = M.run(items, cap=("oin", "mlpout")); C = M.c
    means = {l: v.mean(0) for l, v in C["oin"].items()}; mmeans = {l: v.mean(0) for l, v in C["mlpout"].items()}
    base = summary(clean, items)
    M.ablate_heads(A_HEADS, means); A0 = M.abl
    a = summary(M.run(items), items)
    R = dict(base=base, A=a, head={}, layer={}, mlp={})
    for l in range(L):
        for h in range(Hn):
            if (l, h) in A_HEADS: continue
            M.clear(); M.ablate_heads(A_HEADS + [(l, h)], means)
            R["head"][f"L{l}H{h}"] = summary(M.run(items), items)
        M.clear(); M.ablate_heads(sorted(set(A_HEADS + [(l, h) for h in range(Hn)])), means)
        R["layer"][str(l)] = summary(M.run(items), items)
        M.clear(); M.ablate_heads(A_HEADS, means); M.mabl = {l: mmeans[l]}
        R["mlp"][str(l)] = summary(M.run(items), items)
    # residual patching, teen targets only, sources = same frame/last digit, tens digit 2-9
    rng = random.Random(f"d4r2-e02-src-{d}")
    tgt = [it for it in items if it["cls"] == "teen"]
    src = []
    for it in tgt:
        n = it["n"]; tens = rng.choice(range(2, 10)); ns = n - (n // 10 % 10) * 10 + tens * 10
        src.append(dict(it, n=ns, cls="end123", prefix=it["prefix"][: len(it["prefix"]) - len(str(n))] + str(ns)))
    assert all(len(encode(M.tok, s["prefix"])) == len(encode(M.tok, t["prefix"])) for s, t in zip(src, tgt))
    pairs_items = tgt + src
    for tag, useA in (("clean", False), ("A", True)):
        M.clear()
        if useA: M.ablate_heads(A_HEADS, means)
        M.offsets = (0, 1, 2, 3)
        lt = M.run(tgt); ls = M.run(src, cap=("residoff",)); SC = M.c
        mt, ms = mvals(lt, tgt), mvals(ls, src)
        gap = float((ms - mt).sum())
        P = {}
        for l in range(L):
            for k in (0, 1, 2, 3):
                M.rpatch = {l: (k, SC[f"resid-{k}"][l])}
                P[f"{l}|{k}"] = float((mvals(M.run(tgt), tgt) - mt).sum() / gap)
        M.rpatch = {}
        R[f"patch_{tag}"] = dict(gap_mean=gap / len(tgt), frac=P)
    # attention from the last token
    M.clear(); ids, att, last = M.encode_batch([it["prefix"] for it in items[:64]]); M.last = last
    o = M.model.base_model(input_ids=ids, attention_mask=att, output_attentions=True)
    A = np.zeros((L, Hn, 3))
    for l in range(L):
        w = o.attentions[l].float()
        for k in range(1, 4):
            A[l, :, k - 1] = w[torch.arange(len(last)), :, last, last - k + 1 if k == 1 else last - k + 1].mean(0).cpu().numpy()
    R["att_self_prev_prev2"] = A.tolist()
    res["digits"][str(d)] = R
    bt = a["teen"]
    print(f"== {d} digits ({time.time() - t0:.0f}s): base teen {base['teen']:+.2f} auroc {base['auroc']:.3f}; under A teen {bt:+.2f} (kept {bt / base['teen']:.2f}) auroc {a['auroc']:.3f}")
    top = sorted((v["teen"] / bt, k) for k, v in R["head"].items())[:12]
    print("  extra head (kept of A):", " ".join(f"{k}:{v:.2f}" for v, k in top))
    print("  extra layer:", " ".join(f"{k}:{v['teen'] / bt:.2f}" for k, v in R["layer"].items() if v["teen"] / bt < 0.9))
    print("  extra MLP:", " ".join(f"{k}:{v['teen'] / bt:.2f}" for k, v in R["mlp"].items() if v["teen"] / bt < 0.9))
    for tag in ("clean", "A"):
        P = R[f"patch_{tag}"]["frac"]
        print(f"  patch {tag}:", " | ".join(f"L{l} " + " ".join(f"{P[f'{l}|{k}']:+.2f}" for k in range(4)) for l in range(0, L, 3)))
    json.dump(res, open(OUT2 / f"e02_{NAME}_{'-'.join(map(str, DIG))}.json", "w"), default=float)
print(f"done {time.time() - t0:.0f}s")
