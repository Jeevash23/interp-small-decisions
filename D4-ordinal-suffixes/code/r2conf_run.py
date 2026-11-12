"""Round-2 confirmation runs (PROTOCOL-2.md). Reads held2 frames and held-out numbers only, and only once
PROTOCOL-2.md is frozen (r2common refuses otherwise). Each task writes raw values to outputs/r2conf/; the verdicts
are computed separately by code/r2conf_evaluate.py.
Usage: r2conf_run.py qwen | qwen2 | olmo | other <model> [--dry]   (--dry: exploration items, to outputs/r2dry/)
  qwen  : Qwen3-0.6B-Base, 4 and 6 digits: base, logit lens, DLA (MLP24 share; suffix profiles of MLP24, MLP18,
          MLP20), L0H2 path ablation into every attention layer and MLP, mediation of L0H2 and L11H9 through
          every MLP input from layer 12, and the same for 20 random layer-0 heads' paths through MLP0 (control).
  qwen2 : Qwen3-0.6B-Base, 2, 3 and 4 digits: the backup route (A, A+B, B, A + 20 random pairs from layers 2-6).
  olmo  : OLMo-2-1B, 3, 4, 5, 7, 8, 9, 10 digits: base, probes (layers 2 and 15), the layer-0 triple ablated, 20
          random layer 0-2 triples, the triple x3, and at 4/7/10 digits the triple's path through every receiver.
  other : heads fixed in PROTOCOL-2.md for that model; mean ablation at the last token, 3 and 4 digits, 20 random
          same-size sets from layers 0-1 outside the named sets.
"""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

TASK = sys.argv[1]
DRY = "--dry" in sys.argv
PART = "explore" if DRY else "held"
OUTC = OUT / ("r2dry" if DRY else "r2conf"); OUTC.mkdir(exist_ok=True)
import os
PER, NR = int(os.environ.get("R2_PER", "96")), int(os.environ.get("R2_NR", "20"))
SEED = "d4r2-dry" if DRY else "d4r2-conf"
t0 = time.time()


def items_for(d):
    return make_items(d, PER, f"{SEED}-{TASK}-{d}", part=PART)


def save(obj, name):
    json.dump(obj, open(OUTC / name, "w"), default=float)


def dla_parts(M, C, items):
    """DLA (teen, pull) of MLPs and the suffix profiles of selected MLPs, final norm linearised per item."""
    L = M.L
    g = final_norm(M.model).weight.float().cpu(); eps = getattr(final_norm(M.model), "variance_epsilon", 1e-6)
    Wd = M.wdir().cpu()
    xf = C["resid"][L - 1]; rms = torch.sqrt((xf ** 2).mean(-1) + eps)
    idx = torch.tensor([it["n"] % 10 - 1 for it in items])
    u = (Wd[idx] - Wd[3]) * g / rms[:, None]
    cls = np.array([it["cls"] for it in items]); te, tt = cls == "end123", cls == "teen"
    last = np.array([it["n"] % 10 for it in items])
    tp = lambda v: [float(v[te].mean() - v[tt].mean()), float(v[te].mean())]
    mlp = [tp((C["mlpout"][l] * u).sum(-1).numpy()) for l in range(L)]
    G = (Wd * g)[None] / rms[:, None, None]
    prof = {}
    for l in (18, 20, 24):
        dl = torch.einsum("nd,nsd->ns", C["mlpout"][l], G).numpy(); dl = dl - dl.mean(1, keepdims=True)
        prof[f"mlp{l}"] = {f"{c}|{x}": dl[(cls == c) & (last == x)].mean(0).tolist() for c in ("teen", "end123") for x in (1, 2, 3)}
    return mlp, prof


if TASK == "qwen":
    M = Hooked("qwen3-06b"); L, Hn = M.L, M.H
    res = dict(task=TASK, part=PART, digits={})
    for d in (4, 6):
        items = items_for(d)
        M.clear(); clean = M.run(items, cap=("oin", "mlpout", "resid", "mid")); C = M.c
        means = {l: v.mean(0) for l, v in C["oin"].items()}
        R = dict(base=summary(clean, items), n=len(items))
        R["lens"] = [summary(M.lens(C["resid"][l]), items) for l in range(L)]
        R["dla_mlp"], R["profile"] = dla_parts(M, C, items)
        # L0H2: full, direct paths, mediation
        M.clear(); M.ablate_heads([(0, 2)], means); R["full_L0H2"] = summary(M.run(items, cap=("resid", "mid")), items); A = M.c
        delta = A["mid"][0] - C["mid"][0]
        P = {}
        for l in range(1, L):
            M.clear(); M.add = {("attn", l): delta}; P[f"attn{l}"] = summary(M.run(items), items)
        for l in range(L):
            M.clear(); M.add = {("mlp", l): delta}; P[f"mlp{l}"] = summary(M.run(items), items)
        R["path_L0H2"] = P
        R["med_L0H2"] = {}
        for l in range(12, L):
            M.clear(); M.add = {("mlp", l): A["mid"][l] - C["mid"][l]}; R["med_L0H2"][f"mlp{l}"] = summary(M.run(items), items)
        M.clear(); M.ablate_heads([(11, 9)], means); R["full_L11H9"] = summary(M.run(items, cap=("mid",)), items); A2 = M.c
        R["med_L11H9"] = {}
        for l in range(12, L):
            M.clear(); M.add = {("mlp", l): A2["mid"][l] - C["mid"][l]}; R["med_L11H9"][f"mlp{l}"] = summary(M.run(items), items)
        # control: 20 random layer-0 heads (not L0H2), each one's full ablation and its path through MLP0
        rng = random.Random(f"{SEED}-qwen-ctl-{d}")
        ctl = []
        for h in rng.sample([h for h in range(Hn) if h != 2], min(NR, Hn - 1)):
            M.clear(); M.ablate_heads([(0, h)], means); f = summary(M.run(items, cap=("mid",)), items); dd = M.c["mid"][0] - C["mid"][0]
            M.clear(); M.add = {("mlp", 0): dd}; ctl.append(dict(head=h, full=f, via_mlp0=summary(M.run(items), items)))
        R["ctl_L0"] = ctl
        M.clear()
        res["digits"][str(d)] = R
        save(res, f"qwen.json")
        print(f"qwen {d}d done {time.time() - t0:.0f}s; base {R['base']}")

elif TASK == "qwen2":
    M = Hooked("qwen3-06b"); L, Hn = M.L, M.H
    A = [(0, 2), (1, 6)]; B = [(2, 4), (5, 2)]
    pool = [(l, h) for l in range(2, 7) for h in range(Hn) if (l, h) not in B]
    res = dict(task=TASK, part=PART, digits={})
    for d in (2, 3, 4):
        items = items_for(d)
        M.clear(); clean = M.run(items, cap=("oin",)); means = {l: v.mean(0) for l, v in M.c["oin"].items()}
        R = dict(base=summary(clean, items), sets={}, random=[])
        for k, S in {"A": A, "A+B": A + B, "B": B}.items():
            M.clear(); M.ablate_heads(S, means); R["sets"][k] = summary(M.run(items), items)
        rng = random.Random(f"{SEED}-qwen2-{d}")
        for _ in range(NR):
            S = rng.sample(pool, 2); M.clear(); M.ablate_heads(A + S, means)
            R["random"].append(dict(heads=[f"L{l}H{h}" for l, h in S], **summary(M.run(items), items)))
        M.clear()
        res["digits"][str(d)] = R
        save(res, "qwen2.json")
        print(f"qwen2 {d}d done {time.time() - t0:.0f}s")

elif TASK == "olmo":
    M = Hooked("olmo2-1b"); L, Hn = M.L, M.H
    TRI = [(0, 5), (0, 0), (0, 9)]
    pool = [(l, h) for l in range(3) for h in range(Hn) if (l, h) not in TRI]
    res = dict(task=TASK, part=PART, digits={})
    for d in (3, 4, 5, 7, 8, 9, 10):
        items = items_for(d)
        M.clear(); clean = M.run(items, cap=("oin", "resid", "mid")); C = M.c
        means = {l: v.mean(0) for l, v in C["oin"].items()}
        y = np.array([it["cls"] == "teen" for it in items])
        R = dict(base=summary(clean, items), pieces=number_pieces(M.tok, items[0]["prefix"], items[0]["n"])[1],
                 probe={str(l): probe_cv(C["resid"][l], y) for l in (2, 15)})
        M.clear(); M.ablate_heads(TRI, means); R["tri"] = summary(M.run(items, cap=("mid",)), items); A = M.c
        delta = A["mid"][0] - C["mid"][0]
        M.clear(); M.scale = {0: [(h, 3.0) for _, h in TRI]}; R["tri_x3"] = summary(M.run(items), items)
        rng = random.Random(f"{SEED}-olmo-rand-{d}")
        R["random"] = []
        for _ in range(NR):
            S = rng.sample(pool, 3); M.clear(); M.ablate_heads(S, means)
            R["random"].append(dict(heads=[f"L{l}H{h}" for l, h in S], **summary(M.run(items), items)))
        # teen direction at the last token and its norm relative to the 3-digit one; steering with v3 at layer 8
        V = {l: C["resid"][l][y].mean(0) - C["resid"][l][~y].mean(0) for l in range(L)}
        if d == 3: V3 = V
        R["vnorm_ratio"] = [float(V[l].norm() / V3[l].norm()) for l in range(L)]
        R["cos_v3"] = [float(torch.nn.functional.cosine_similarity(V[l], V3[l], dim=0)) for l in range(L)]
        if d in (3, 4, 7, 10):
            non = [i for i, it in enumerate(items) if it["cls"] == "end123"]; nit = [items[i] for i in non]
            M.clear(); m0 = float(mvals(M.run(nit), nit).mean())
            gen = torch.Generator().manual_seed(8 + d)
            nrm = V3[8].norm(); dirs = {"v3": V3[8]}
            for j in range(5):
                r = torch.randn(V3[8].shape, generator=gen); dirs[f"rand{j}"] = r / r.norm() * nrm
            St = {}
            for k, v in dirs.items():
                M.clear(); M.rpatch = {8: (0, C["resid"][8][non] + v)}; St[k] = float(mvals(M.run(nit), nit).mean()) - m0
            R["steer_L8"] = dict(m0=m0, **St)
        if d in (4, 7, 10):
            P = {}
            for l in range(1, L):
                M.clear(); M.add = {("attn", l): delta}; P[f"attn{l}"] = summary(M.run(items), items)
            for l in range(L):
                M.clear(); M.add = {("mlp", l): delta}; P[f"mlp{l}"] = summary(M.run(items), items)
            R["path_tri"] = P
        M.clear()
        res["digits"][str(d)] = R
        save(res, "olmo.json")
        print(f"olmo {d}d {R['pieces']} done {time.time() - t0:.0f}s; auroc {R['base']['auroc']:.3f}")

elif TASK == "other":
    NAME = sys.argv[2]
    SETS = json.loads(open(Path(__file__).resolve().parent / "r2conf_sets.json").read())[NAME]
    M = Hooked(NAME); L, Hn = M.L, M.H
    named = {tuple(map(int, x[1:].split("H"))) for S in SETS.values() for x in S}
    pool = [(l, h) for l in range(2) for h in range(Hn) if (l, h) not in named]
    res = dict(task=TASK, model=NAME, part=PART, sets=SETS, digits={})
    for d in (3, 4):
        items = items_for(d)
        M.clear(); clean = M.run(items, cap=("oin",)); means = {l: v.mean(0) for l, v in M.c["oin"].items()}
        R = dict(base=summary(clean, items), sets={}, random={})
        for k, S in SETS.items():
            S = [tuple(map(int, x[1:].split("H"))) for x in S]
            M.clear(); M.ablate_heads(S, means); R["sets"][k] = summary(M.run(items), items)
            rng = random.Random(f"{SEED}-other-{NAME}-{d}-{k}")
            rr = []
            for _ in range(NR):
                Q = rng.sample(pool, len(S)); M.clear(); M.ablate_heads(Q, means)
                rr.append(dict(heads=[f"L{l}H{h}" for l, h in Q], **summary(M.run(items), items)))
            R["random"][k] = rr
        M.clear()
        res["digits"][str(d)] = R
        save(res, f"other_{NAME}.json")
        print(f"other {NAME} {d}d done {time.time() - t0:.0f}s")
print(f"done {time.time() - t0:.0f}s")
