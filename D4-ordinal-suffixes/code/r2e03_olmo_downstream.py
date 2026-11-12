"""R2 exploration E03 (CHARTER-2.md question 3), OLMo-2-1B: what reads the tens digit after the layer-0 heads move
it, and is the 4/7-digit weakness a missing route or a weak one?
Items per length: 96 teen and 96 non-teen ending 1-3, plain form, exploration frames and numbers; lengths
3,4,5,7 (round 1) and 8,9,10 (explore2 numbers; tokens 3+3+2, 3+3+3, 3+3+3+1).
  clean summary; logit lens by layer; probes (5-fold CV, logistic) for teen status at the last token and at the
  previous token, by layer; the same last-token probe with the triple ablated; DLA of each attention and MLP
  block (post-norm block outputs) to the teen effect; triple (L0H5+L0H0+L0H9) full ablation; triple scaled
  by 2 and 3 (gain test); path ablation of the triple into each attention layer, MLP and the unembedding.
Writes outputs/r2/e03_<model>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PER = int(arg("--per", "96")); NAME = arg("--model", "olmo2-1b"); DIG = [int(x) for x in arg("--digits", "3,4,5,7,8,9,10").split(",")]
TRI = [(0, 5), (0, 0), (0, 9)]
M = Hooked(NAME); L = M.L
g = final_norm(M.model).weight.float().cpu(); eps = getattr(final_norm(M.model), "variance_epsilon", 1e-6)
Wd = M.wdir().cpu()
res = dict(model=NAME, digits={}); t0 = time.time()
for d in DIG:
    items = make_items(d, PER, f"d4r2-e03-{d}")
    pieces = number_pieces(M.tok, items[0]["prefix"], items[0]["n"])[1]
    M.clear(); M.offsets = (1,)
    clean = M.run(items, cap=("oin", "resid", "mid", "emb", "residoff")); C = M.c
    means = {l: v.mean(0) for l, v in C["oin"].items()}
    base = summary(clean, items)
    y = np.array([it["cls"] == "teen" for it in items])
    R = dict(base=base, pieces=pieces)
    R["lens"] = [summary(M.lens(C["resid"][l]), items) for l in range(L)]
    R["probe_last"] = [probe_cv(C["resid"][l], y) for l in range(L)]
    R["probe_prev"] = [probe_cv(C["resid-1"][l], y) for l in range(L)]
    # DLA of blocks (post-norm outputs): attn_l = mid_l - resid_{l-1}; mlp_l = resid_l - mid_l
    xf = C["resid"][L - 1]; rms = torch.sqrt((xf ** 2).mean(-1) + eps)
    idx = torch.tensor([it["n"] % 10 - 1 for it in items]); u = (Wd[idx] - Wd[3]) * g / rms[:, None]
    te, tt = ~y, y
    tp = lambda v: [float(v[te].mean() - v[tt].mean()), float(v[te].mean())]
    prev = C["emb"]; da, dm = [], []
    for l in range(L):
        da.append(tp(((C["mid"][l] - prev) * u).sum(-1).numpy())); dm.append(tp(((C["resid"][l] - C["mid"][l]) * u).sum(-1).numpy()))
        prev = C["resid"][l]
    R["dla_attn"], R["dla_mlp"], R["dla_emb"] = da, dm, tp((C["emb"] * u).sum(-1).numpy())
    # triple: full ablation, probes under ablation, gain
    M.clear(); M.ablate_heads(TRI, means)
    ab = M.run(items, cap=("resid", "mid")); A = M.c
    R["tri_full"] = summary(ab, items)
    R["probe_last_tri"] = [probe_cv(A["resid"][l], y) for l in range(L)]
    delta = A["mid"][0] - C["mid"][0]
    for k in (2.0, 3.0):
        M.clear(); M.scale = {0: [(h, k) for _, h in TRI]}
        R[f"tri_x{k:g}"] = summary(M.run(items), items)
    M.clear()
    P = {}
    for l in range(1, L):
        M.add = {("attn", l): delta}; P[f"attn{l}"] = summary(M.run(items), items)
    for l in range(0, L):
        M.add = {("mlp", l): delta}; P[f"mlp{l}"] = summary(M.run(items), items)
    M.add = {("final",): delta}; P["direct"] = summary(M.run(items), items); M.clear()
    R["tri_path"] = P
    res["digits"][str(d)] = R
    json.dump(res, open(OUT2 / f"e03_{NAME}_{'-'.join(map(str, DIG))}.json", "w"), default=float)
    print(f"== {d} digits {pieces} ({time.time() - t0:.0f}s): teen {base['teen']:+.2f} auroc {base['auroc']:.3f} acc teen {base['acc_teen']:.2f}; triple kept {R['tri_full']['teen'] / base['teen']:.2f}; x2 {R['tri_x2']['teen'] / base['teen']:.2f} auroc {R['tri_x2']['auroc']:.3f}; x3 {R['tri_x3']['teen'] / base['teen']:.2f} auroc {R['tri_x3']['auroc']:.3f}")
    print("  lens teen:", " ".join(f"{l}:{R['lens'][l]['teen']:+.1f}" for l in range(L)))
    print("  probe last:", " ".join(f"{v:.2f}" for v in R["probe_last"]))
    print("  probe last, triple ablated:", " ".join(f"{v:.2f}" for v in R["probe_last_tri"]))
    print("  probe prev:", " ".join(f"{v:.2f}" for v in R["probe_prev"]))
    print("  DLA attn:", " ".join(f"{l}:{v[0]:+.2f}" for l, v in enumerate(da) if abs(v[0]) > 0.2), "| mlp:", " ".join(f"{l}:{v[0]:+.2f}" for l, v in enumerate(dm) if abs(v[0]) > 0.2))
    tot = base["teen"] - R["tri_full"]["teen"]
    if abs(tot) > 0.05:
        sh = sorted(((base["teen"] - v["teen"]) / tot, k) for k, v in P.items())[::-1][:10]
        print(f"  path triple (full loss {tot:+.2f}):", " ".join(f"{k}:{v:.2f}" for v, k in sh))
print(f"done {time.time() - t0:.0f}s")
