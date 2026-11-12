"""D5 round 2, exploration 2 (exploratory; CHARTER-2.md Q1, Q2): total effect of each component.
Usage: r2_exp02_scan.py <model> <part> <first> <n> [per_para=2] [seed=12] [maxlayer=l]
At up to `per_para` word starts per paragraph (word 12 or later), prefixes U, L, A as in r2_exp01.
Each attention head (o_proj input slice) and each MLP output in layers 0..maxlayer is patched, at the
prediction position only and one at a time, from the L run into the U run and into the A run. Recorded
per patch: the projection of the residual after layer l on the unit capitals direction u, the gap
D = logp(U) - logp(Lc), and the top-1. Writes outputs/r2/exp02_<model>_<part>_<first>.pt"""
import sys, time
from d5common2 import *  # noqa: F401,F403
from d5common2 import LastBatch

short, part = sys.argv[1], sys.argv[2]; FIRST, N = int(sys.argv[3]), int(sys.argv[4])
PER = int(sys.argv[5]) if len(sys.argv) > 5 else 2; SEED = int(sys.argv[6]) if len(sys.argv) > 6 else 12
LAYER = {"gemma": 24, "qwen3-17b": 22}[short]; MAXL = int(sys.argv[7]) if len(sys.argv) > 7 else LAYER
BS = 24 if short == "qwen3-17b" else 32
rows = load_part2(part, seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; L, H = m.L, m.H
LB = LastBatch(M_)
md = torch.load(OUT / f"exp01_{short}_means.pt")
d = (md["means"]["upper"] - md["means"]["lower"]).float()[LAYER + 1].to(DEV); u = d / d.norm()
mcap, mpatch, st = {}, {}, {"cap": False}
def mk(l):
    def f(mod, a, o):
        if st["cap"]: mcap[l] = o[0, -1].detach().clone()
        lst = mpatch.get(l)
        if not lst: return None
        o = o.clone()
        for row, val in lst: o[row, -1] = val
        return o
    return f
for l in range(L): m.layers[l].mlp.register_forward_hook(mk(l))
lout = {}
m.layers[LAYER].register_forward_hook(lambda mod, a, o: lout.update(x=(o[0] if isinstance(o, tuple) else o)[:, -1].float()))
COMPS = [("h", l, h) for l in range(MAXL + 1) for h in range(H)] + [("m", l) for l in range(MAXL + 1)]
# readers: components after the direction layer, frozen at their unedited L-run outputs while d is added at l
COMPS2 = [("h", l, h) for l in range(LAYER + 1, L) for h in range(H)] + [("m", l) for l in range(LAYER + 1, L)]


def build(text, rs, i, form):
    pre = text[:rs[i][0] - 1]
    if form == "U": return pre.upper()
    if form == "L": return pre.lower()
    c0 = rs[i - 1][0]
    return pre[:c0].upper() + pre[c0:].lower()


rng = random.Random(SEED); t0 = time.time(); recs = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); rs = runs(text)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 12 and text[s - 1] == " "]
    for i in sorted(rng.sample(cand, min(PER, len(cand)))):
        word = text[rs[i][0]:rs[i][1]]
        Ut = tok(" " + word.upper(), add_special_tokens=False)["input_ids"][0]; Lt = tok(" " + word.lower(), add_special_tokens=False)["input_ids"][0]
        rec = {"para": FIRST + pi, "pageid": r["pageid"], "run": i, "word": word, "U": vs[Ut], "Lc": vs[Lt]}
        ids = {f: tok(build(text, rs, i, f), add_special_tokens=False)["input_ids"] for f in ("U", "L", "A")}
        zc = {}
        for f in ("U", "L", "A"):
            m.capture = True; st["cap"] = True
            with torch.no_grad(): lg = m.model(torch.tensor([ids[f]], device=DEV), logits_to_keep=1).logits[0, -1].float()
            m.capture = False; st["cap"] = False
            lp = torch.log_softmax(lg, -1)
            rec[f] = {"top": vs[int(lg.argmax())], "D": float(lp[Ut] - lp[Lt]), "proj": float(lout["x"][0] @ u)}
            if f == "L":
                zc = {"z": {l: m.cap[l][-1].clone() for l in range(L)}, "mlp": {l: mcap[l].clone() for l in range(L)}}
                hL = m.rcap[LAYER][-1].clone()
        LB.clear()
        for f in ("U", "L", "A"): LB.prime(ids[f])
        if pi == 0:
            for f in ("U", "A"):
                lg = LB(ids[f], 2); lp = torch.log_softmax(lg, -1)
                print("cache check", f, round(float(lp[0, Ut] - lp[0, Lt]), 3), round(rec[f]["D"], 3), round(float(lout["x"][0] @ u), 2), round(rec[f]["proj"], 2), flush=True)
        for f in ("U", "A"):
            P = np.zeros(len(COMPS), np.float32); Dd = np.zeros(len(COMPS), np.float32); tops = []
            for s in range(0, len(COMPS), BS):
                cs = COMPS[s:s + BS]; m.patch = {}; mpatch.clear()
                for j, c in enumerate(cs):
                    if c[0] == "h":
                        l, h = c[1], c[2]
                        m.patch.setdefault(l, []).append((h, -1, zc["z"][l][m.hs(l, h)], j))
                    else:
                        mpatch.setdefault(c[1], []).append((j, zc["mlp"][c[1]]))
                lg = LB(ids[f], len(cs))
                m.patch = {}; mpatch.clear()
                lp = torch.log_softmax(lg, -1)
                P[s:s + len(cs)] = (lout["x"] @ u).cpu().numpy(); Dd[s:s + len(cs)] = (lp[:, Ut] - lp[:, Lt]).cpu().numpy()
                tops += [vs[int(t)] for t in lg.argmax(-1)]
            rec[f]["patched_proj"] = P; rec[f]["patched_D"] = Dd; rec[f]["patched_top"] = tops
        # readers: L run + d at l, each later component frozen (row 0: nothing frozen)
        P = np.zeros(len(COMPS2) + 1, np.float32); Dd = np.zeros(len(COMPS2) + 1, np.float32); tops = []
        allc = [None] + COMPS2
        for s in range(0, len(allc), BS):
            cs = allc[s:s + BS]; m.patch = {}; mpatch.clear()
            m.rpatch = {LAYER: [(-1, hL + d.to(hL.dtype), j) for j in range(len(cs))]}
            for j, c in enumerate(cs):
                if c is None: continue
                if c[0] == "h": m.patch.setdefault(c[1], []).append((c[2], -1, zc["z"][c[1]][m.hs(c[1], c[2])], j))
                else: mpatch.setdefault(c[1], []).append((j, zc["mlp"][c[1]]))
            lg = LB(ids["L"], len(cs))
            m.patch = {}; mpatch.clear(); m.rpatch = {}
            lp = torch.log_softmax(lg, -1)
            Dd[s:s + len(cs)] = (lp[:, Ut] - lp[:, Lt]).cpu().numpy(); tops += [vs[int(t)] for t in lg.argmax(-1)]
        rec["Ld"] = {"patched_D": Dd, "patched_top": tops}
        recs.append(rec)
    print(pi, round(time.time() - t0), flush=True)

torch.save({"model": short, "part": part, "rows": [FIRST, FIRST + N], "layer": LAYER, "comps": COMPS, "comps2": COMPS2, "records": recs,
            "seconds": round(time.time() - t0)}, OUT / "r2" / f"exp02_{short}_{part}_{FIRST}.pt")
name = lambda c: f"L{c[1]}H{c[2]}" if c[0] == "h" else f"MLP{c[1]}"
for f in ("U", "A"):
    base = np.array([rc[f]["proj"] for rc in recs]); lo = np.array([rc["L"]["proj"] for rc in recs])
    Pm = np.stack([rc[f]["patched_proj"] for rc in recs])
    frac = (base[:, None] - Pm).mean(0) / (base - lo).mean()
    Db = np.array([rc[f]["D"] for rc in recs]); Dl = np.array([rc["L"]["D"] for rc in recs])
    Dm = np.stack([rc[f]["patched_D"] for rc in recs]); fD = (Db[:, None] - Dm).mean(0) / (Db - Dl).mean()
    order = np.argsort(-frac)
    print(f, "excess proj %.2f, D excess %.2f" % ((base - lo).mean(), (Db - Dl).mean()))
    print(" top by proj removed:", [(name(COMPS[k]), round(float(frac[k]), 3), round(float(fD[k]), 3)) for k in order[:25]])
    order = np.argsort(-fD)
    print(" top by D removed:", [(name(COMPS[k]), round(float(fD[k]), 3)) for k in order[:15]])
Db = np.array([rc["Ld"]["patched_D"][0] for rc in recs]); Dl = np.array([rc["L"]["D"] for rc in recs])
Dm = np.stack([rc["Ld"]["patched_D"][1:] for rc in recs]); fD = (Db[:, None] - Dm).mean(0) / (Db - Dl).mean()
fl0 = np.mean([case_of(rc["Ld"]["patched_top"][0]) == "upper" for rc in recs if case_of(rc["L"]["top"]) == "lower"])
print("Ld: flip without freezing %.3f; D gain %.2f" % (fl0, (Db - Dl).mean()))
order = np.argsort(-fD)
print(" readers by D gain removed:", [(name(COMPS2[k]), round(float(fD[k]), 3)) for k in order[:20]])
