"""R2 exploration E05 (CHARTER-2.md question 1), Qwen3-0.6B-Base: following the teen signal past its first hop.
Usage: r2e05_qwen_hops.py [--digits 2,3,4] [--per 96]
  1. Each MLP mean-ablated at the last token: teen effect and pull kept.
  2. Mediation: for senders L0H2, L0H3, L11H9 (full mean ablation at the last token), the change each sender
     causes at the input of every later attention layer and MLP is added to that receiver's input alone;
     share = loss through that receiver / full loss (teen for L0H2, pull for L0H3, both for L11H9).
  3. Hop 2: L0H2's change passed to MLP0's input only; the resulting change in MLP0's output is passed to each
     later receiver alone (share of the MLP0-only loss).
  4. Suffix profile: mean direct logit attribution of MLP18-27 and L11H9 to each of st/nd/rd/th (relative to
     their mean), by class (teen, non-teen) and last digit.
Writes outputs/r2/e05_<model>_<digits>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
DIG = [int(x) for x in arg("--digits", "2,3,4").split(",")]
PER = int(arg("--per", "96")); NAME = arg("--model", "qwen3-06b")
M = Hooked(NAME); L, Hn, dh = M.L, M.H, M.dh
g = final_norm(M.model).weight.float().cpu(); eps = getattr(final_norm(M.model), "variance_epsilon", 1e-6)
Wd = M.wdir().cpu()
res = dict(model=NAME, digits={}); t0 = time.time()


def receivers(lo):
    return [("attn", l) for l in range(lo + 1, L)] + [("mlp", l) for l in range(lo, L)]


for d in DIG:
    items = make_items(d, PER, f"d4r2-e01-{d}")      # same items as E01
    M.clear(); clean = M.run(items, cap=("oin", "mlpout", "resid", "mid")); C = M.c
    base = summary(clean, items)
    means = {l: v.mean(0) for l, v in C["oin"].items()}; mmeans = {l: v.mean(0) for l, v in C["mlpout"].items()}
    R = dict(base=base, mlp_abl={}, med={}, hop2={})
    for l in range(L):
        M.clear(); M.mabl = {l: mmeans[l]}; R["mlp_abl"][str(l)] = summary(M.run(items), items)
    print(f"  [{d}d] MLP scan {time.time() - t0:.0f}s")
    inp = lambda CC, kind, l: (CC["resid"][l - 1] if l > 0 else None) if kind == "attn" else CC["mid"][l]
    for (l0, h0) in [(0, 2), (0, 3), (11, 9)]:
        M.clear(); M.ablate_heads([(l0, h0)], means); full = summary(M.run(items, cap=("resid", "mid")), items); A = M.c
        P = dict(full=full)
        for kind, l in receivers(l0):
            M.clear(); M.add = {(kind, l): inp(A, kind, l) - inp(C, kind, l)}
            P[f"{kind}{l}"] = summary(M.run(items), items)
        M.clear(); M.add = {("final",): A["resid"][L - 1] - C["resid"][L - 1]}; P["total_check"] = summary(M.run(items), items)
        R["med"][f"L{l0}H{h0}"] = P
        print(f"  [{d}d] mediation L{l0}H{h0} {time.time() - t0:.0f}s")
    # hop 2: L0H2 -> MLP0 -> ?
    M.clear(); M.ablate_heads([(0, 2)], means); M.run(items, cap=("mid",)); d1 = M.c["mid"][0] - C["mid"][0]
    M.clear(); M.add = {("mlp", 0): d1}; only0 = summary(M.run(items, cap=("resid",)), items)
    d2 = M.c["resid"][0] - C["resid"][0]          # attention 0 unchanged, so this is MLP0's change
    R["hop2"]["mlp0_only"] = only0
    for kind, l in receivers(0):
        if kind == "mlp" and l == 0: continue
        M.clear(); M.add = {(kind, l): d2}; R["hop2"][f"{kind}{l}"] = summary(M.run(items), items)
    M.add = {("final",): d2}; R["hop2"]["direct"] = summary(M.run(items), items); M.clear()
    # suffix profile
    xf = C["resid"][L - 1]; rms = torch.sqrt((xf ** 2).mean(-1) + eps)
    G = (Wd * g)[None] / rms[:, None, None]            # [N, 4, d]
    prof = {}
    cls = np.array([it["cls"] for it in items]); last = np.array([it["n"] % 10 for it in items])
    comps = {f"mlp{l}": C["mlpout"][l] for l in range(18, L)}
    Wo = blocks(M.model)[11].self_attn.o_proj.weight.float().cpu()
    comps["L11H9"] = C["oin"][11][:, 9 * dh:10 * dh] @ Wo[:, 9 * dh:10 * dh].T
    for k, v in comps.items():
        dla = torch.einsum("nd,nsd->ns", v, G).numpy(); dla = dla - dla.mean(1, keepdims=True)
        prof[k] = {f"{c}|{x}": dla[(cls == c) & (last == x)].mean(0).tolist() for c in ("teen", "end123") for x in (1, 2, 3)}
    R["suffix_profile"] = prof
    res["digits"][str(d)] = R
    json.dump(res, open(OUT2 / f"e05_{NAME}_{'-'.join(map(str, DIG))}.json", "w"), default=float)
    print(f"== {d} digits ({time.time() - t0:.0f}s) teen {base['teen']:+.2f} pull {base['pull']:+.2f}")
    print("  MLP ablation, teen/pull kept:", " ".join(f"{l}:{v['teen'] / base['teen']:.2f}/{v['pull'] / base['pull']:.2f}" for l, v in R["mlp_abl"].items() if v['teen'] / base['teen'] < 0.9 or v['pull'] / base['pull'] < 0.9))
    for s, P in R["med"].items():
        for q in ("teen", "pull"):
            tot = base[q] - P["full"][q]
            if abs(tot) < 0.05: continue
            sh = sorted(((base[q] - v[q]) / tot, k) for k, v in P.items() if k not in ("full", "total_check"))[::-1][:10]
            print(f"  mediation {s} {q} (full loss {tot:+.2f}):", " ".join(f"{k}:{v:.2f}" for v, k in sh))
    tot = base["teen"] - only0["teen"]
    sh = sorted(((base["teen"] - v["teen"]) / tot, k) for k, v in R["hop2"].items() if k != "mlp0_only")[::-1][:10]
    print(f"  hop2 L0H2->MLP0->? (MLP0-only loss {tot:+.2f}):", " ".join(f"{k}:{v:.2f}" for v, k in sh))
    for k in ("mlp24", "mlp25", "L11H9"):
        print(f"  profile {k} (st nd rd th):", " | ".join(f"{c}{x}: " + " ".join(f"{z:+.1f}" for z in prof[k][f'{c}|{x}']) for c in ("teen", "end123") for x in (1, 2, 3)))
print(f"done {time.time() - t0:.0f}s")
