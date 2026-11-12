"""R2 exploration E01 (CHARTER-2.md question 1), Qwen3-0.6B-Base: where do the tens-digit and last-digit signals go?
Usage: r2e01_qwen_path.py [--digits 2,3,4] [--per 96] [--model qwen3-06b]
For each digit count, 96 teen and 96 non-teen (ending 1-3) exploration items in round-1 and explore2 frames
(rankings, positions, anniversaries). Measures at the last token:
  clean summary; logit lens by layer; direct logit attribution (DLA) of every head and MLP to the teen effect
  and to the last-digit pull; full mean ablation of senders; path ablation of single senders (L0H2, L1H6, L0H3)
  into each attention layer, each MLP and the unembedding; per-head path ablation (digits >= 3) for L0H2 and L0H3.
Writes outputs/r2/e01_<model>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
DIG = [int(x) for x in arg("--digits", "2,3,4").split(",")]
PER = int(arg("--per", "96")); NAME = arg("--model", "qwen3-06b")
PERHEAD = arg("--perhead", "0.2,0.3")
M = Hooked(NAME); L, Hn, dh = M.L, M.H, M.dh
SENDERS_FULL = [[(0, 2)], [(1, 6)], [(0, 2), (1, 6)], [(0, 3)], [(11, 9)], [(0, 7)]]
SINGLE = [(0, 2), (1, 6), (0, 3)]
PH = [tuple(map(int, x.split("."))) for x in PERHEAD.split(",")] if PERHEAD else []
g = final_norm(M.model).weight.float().cpu()
eps = getattr(final_norm(M.model), "variance_epsilon", 1e-6)
Wd = M.wdir().cpu()
res = dict(model=NAME, per=PER, digits={})
t0 = time.time()
for d in DIG:
    items = make_items(d, PER, f"d4r2-e01-{d}")
    M.clear(); clean = M.run(items, cap=("oin", "mlpout", "resid", "mid", "emb")); C = M.c
    base = summary(clean, items)
    means = {l: v.mean(0) for l, v in C["oin"].items()}
    R = dict(base=base, n=len(items))
    R["lens"] = [summary(M.lens(C["resid"][l]), items) for l in range(L)]
    # DLA onto m's direction, final norm linearised at each item's own scale
    xf = C["resid"][L - 1]; rms = torch.sqrt((xf ** 2).mean(-1) + eps)
    idx = torch.tensor([it["n"] % 10 - 1 for it in items])
    u = (Wd[idx] - Wd[3]) * g / rms[:, None]
    cls = np.array([it["cls"] for it in items]); te, tt = cls == "end123", cls == "teen"
    def te_pull(v):
        v = v.numpy(); return [float(v[te].mean() - v[tt].mean()), float(v[te].mean())]
    dla_h = [[None] * Hn for _ in range(L)]; dla_m = []
    for l in range(L):
        Wo = blocks(M.model)[l].self_attn.o_proj.weight.float().cpu()
        for h in range(Hn):
            sl = slice(h * dh, (h + 1) * dh)
            dla_h[l][h] = te_pull(((C["oin"][l][:, sl] @ Wo[:, sl].T) * u).sum(-1))
        dla_m.append(te_pull((C["mlpout"][l] * u).sum(-1)))
    R["dla_head"], R["dla_mlp"], R["dla_emb"] = dla_h, dla_m, te_pull((C["emb"] * u).sum(-1))
    # full ablation of senders
    R["full"] = {}
    for S in SENDERS_FULL:
        M.clear(); M.ablate_heads(S, means)
        R["full"]["+".join(f"L{l}H{h}" for l, h in S)] = summary(M.run(items), items)
    # path ablation of single senders
    print(f"  [{d}d] clean, lens, DLA, full done {time.time() - t0:.0f}s", flush=True)
    R["path"] = {}
    for (l0, h0) in SINGLE:
        M.clear(); M.ablate_heads([(l0, h0)], means); M.run(items, cap=("mid",))
        delta = M.c["mid"][l0] - C["mid"][l0]; M.clear()
        P = {}
        for l in range(l0 + 1, L):
            M.add = {("attn", l): delta}; P[f"attn{l}"] = summary(M.run(items), items)
        for l in range(l0, L):
            M.add = {("mlp", l): delta}; P[f"mlp{l}"] = summary(M.run(items), items)
        M.add = {("final",): delta}; P["direct"] = summary(M.run(items), items)
        print(f"  [{d}d] path L{l0}H{h0} layers done {time.time() - t0:.0f}s", flush=True)
        if d >= 3 and (l0, h0) in PH:
            q = "teen" if (l0, h0) == (0, 2) else "pull"
            M.clear(); M.ablate_heads([(l0, h0)], means); fq = summary(M.run(items), items)[q]; M.clear()
            tot = base[q] - fq
            layers = [l for l in range(l0 + 1, L) if abs(tot) > 1e-6 and (base[q] - P[f"attn{l}"][q]) / tot >= 0.03]
            P["perhead_layers"] = layers
            print(f"  [{d}d] per-head scan of L{l0}H{h0} ({q}) in layers {layers}")
            for l in layers:
                for h in range(Hn):
                    M.add = {("attn", l): delta}; M.keep = {l: (h, C["oin"][l])}
                    P[f"L{l}H{h}"] = summary(M.run(items), items)
        M.clear()
        R["path"][f"L{l0}H{h0}"] = P
    res["digits"][str(d)] = R
    json.dump(res, open(OUT2 / f"e01_{NAME}_{'-'.join(map(str, DIG))}.json", "w"), default=float)
    # brief print
    print(f"== {d} digits ({time.time() - t0:.0f}s): base teen {base['teen']:+.2f} pull {base['pull']:+.2f} auroc {base['auroc']:.3f}")
    print("  lens teen by layer:", " ".join(f"{l}:{R['lens'][l]['teen']:+.1f}" for l in range(L)))
    print("  lens pull by layer:", " ".join(f"{l}:{R['lens'][l]['pull']:+.1f}" for l in range(L)))
    top = sorted(((dla_h[l][h][0], f"L{l}H{h}") for l in range(L) for h in range(Hn)), reverse=True)[:8]
    print("  DLA teen heads:", " ".join(f"{k}:{v:+.2f}" for v, k in top))
    print("  DLA teen MLPs:", " ".join(f"{l}:{v[0]:+.2f}" for l, v in enumerate(dla_m) if abs(v[0]) > 0.3))
    print("  full:", " ".join(f"{k}: teen {v['teen'] / base['teen']:.2f} pull {v['pull'] / base['pull']:.2f}" for k, v in R["full"].items()))
    for s, P in R["path"].items():
        full = R["full"][s]
        for key, q in (("teen", "teen"), ("pull", "pull")):
            tot = base[q] - full[q]
            if abs(tot) < 1e-6: continue
            sh = sorted(((base[q] - v[q]) / tot, k) for k, v in P.items() if isinstance(v, dict))[::-1][:14]
            print(f"  path {s} {key} (full loss {tot:+.2f}):", " ".join(f"{k}:{v:.2f}" for v, k in sh))
print(f"done {time.time() - t0:.0f}s")
