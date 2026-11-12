"""D5 exploration 5 (exploratory): head-level scan and combined patches.
Usage: exp05_heads.py <model> <layers a,b,c> [first=300] [n=50]
Same word starts and prefixes as exp03. For each head in the listed layers, its input to o_proj at the
last position is replaced by its value from the lower run (upper <- lower); fraction of
D = logp(U) - logp(Lc) removed. Then combined patches (all listed layers' attention, plus groups of
MLP outputs given by exp03) and their case-flip / same-word rates.
Writes outputs/exp05_<model>.json."""
import random, sys, time
from d5common import *  # noqa: F401,F403

short = sys.argv[1]; LAYS = [int(x) for x in sys.argv[2].split(",")]
FIRST = int(sys.argv[3]) if len(sys.argv) > 3 else 300; N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
rows = load_part("explore", seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; L, H = m.L, m.H
mlp_cap, mlp_patch, capturing = {}, {}, [False]


def mk(l):
    def f(mod, a, o):
        x = o[0] if isinstance(o, tuple) else o
        if capturing[0]: mlp_cap[l] = x[0, -1].detach().clone()
        rv = mlp_patch.get(l)
        if not rv: return None
        x = x.clone()
        for row, val in rv: x[row, -1] = val
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
    return f
for l in range(L): m.layers[l].mlp.register_forward_hook(mk(l))

# combined patches: (name, attention layers, mlp layers)
COMBOS = [("attn_listed", LAYS, []), ("attn_all", list(range(L)), []), ("mlp_all", [], list(range(L))),
          ("late_mlp", [], list(range(L - 4, L))), ("mid_attn_mlp", [l for l in LAYS if l < L - 4], [l for l in LAYS if l < L - 4]),
          ("listed_plus_late", LAYS, list(range(L - 4, L)))]


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


rng = random.Random(2); t0 = time.time(); out = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); V = versions(text)
    enc = {v: encode(tok, V[v]) for v in ("upper", "lower")}
    wsu, _ = positions(V["upper"], enc["upper"][1]); wsl, _ = positions(V["lower"], enc["lower"][1])
    both = sorted(set(wsu.values()) & set(wsl.values()))
    for i in sorted(rng.sample(both, min(4, len(both)))):
        kk = {"upper": [k for k, j in wsu.items() if j == i][0], "lower": [k for k, j in wsl.items() if j == i][0]}
        pre = {v: enc[v][0][:kk[v]] for v in kk}
        U, Lc = enc["upper"][0][kk["upper"]], enc["lower"][0][kk["lower"]]
        z, mc, base = {}, {}, {}
        for v in ("upper", "lower"):
            m.capture = True; capturing[0] = True
            with torch.no_grad(): lg = m.model(torch.tensor([pre[v]], device=DEV), logits_to_keep=1).logits[0, -1].float()
            m.capture = False; capturing[0] = False
            z[v] = {l: m.cap[l][-1].clone() for l in range(L)}; mc[v] = dict(mlp_cap); base[v] = torch.log_softmax(lg, -1)
        D0 = float(base["upper"][U] - base["upper"][Lc]); top0 = vs[int(base["upper"].argmax())]
        jobs = [("h", l, h) for l in LAYS for h in range(H)] + [("c",) + c for c in COMBOS]
        res = {}
        for s in range(0, len(jobs), 16):
            js = jobs[s:s + 16]; m.patch = {}; mlp_patch.clear()
            for j, jb in enumerate(js):
                if jb[0] == "h":
                    l, h = jb[1], jb[2]; m.patch.setdefault(l, []).append((h, -1, z["lower"][l][m.hs(l, h)], j))
                else:
                    for l in jb[2]:
                        for h in range(H): m.patch.setdefault(l, []).append((h, -1, z["lower"][l][m.hs(l, h)], j))
                    for l in jb[3]: mlp_patch.setdefault(l, []).append((j, mc["lower"][l]))
            x = torch.tensor([pre["upper"]] * len(js), device=DEV)
            with torch.no_grad(): lg = m.model(x, logits_to_keep=1).logits[:, -1].float()
            m.patch = {}; mlp_patch.clear()
            lp = torch.log_softmax(lg, -1)
            for j, jb in enumerate(js):
                key = f"L{jb[1]}H{jb[2]}" if jb[0] == "h" else jb[1]
                res[key] = [round(float(lp[j][U] - lp[j][Lc]), 3), vs[int(lp[j].argmax())]]
        out.append({"para": FIRST + pi, "run": i, "U": vs[U], "Lc": vs[Lc], "D0": round(D0, 3), "top0": top0, "patched": res})
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)

recs = [rc for rc in out if case_of(rc["top0"]) == "upper"]
Ds = np.array([rc["D0"] for rc in recs]); summ = {}
for key in out[0]["patched"]:
    Dp = np.array([rc["patched"][key][0] for rc in recs])
    flip = [rc for rc in recs if case_of(rc["patched"][key][1]) == "lower"]
    kept = [rc for rc in flip if same_word(rc["top0"], rc["patched"][key][1])]
    summ[key] = {"frac_D_moved": float((Ds.mean() - Dp.mean()) / Ds.mean()), "flip": len(flip) / len(recs), "flip_kept": len(kept) / len(recs)}
save({"model": short, "layers": LAYS, "rows": [FIRST, FIRST + N], "n": len(recs), "summary": summ, "records": out, "seconds": round(time.time() - t0)}, f"exp05_{short}.json")
heads = sorted([k for k in summ if k.startswith("L")], key=lambda k: -summ[k]["frac_D_moved"])
print("top heads", [(k, round(summ[k]["frac_D_moved"], 3)) for k in heads[:15]])
for c in COMBOS: print(c[0], {k: round(v, 3) for k, v in summ[c[0]].items()})
