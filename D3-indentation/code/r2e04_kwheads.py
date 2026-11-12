"""D3 round 2, r2e04 (exploratory, question 3): which heads read the terminal keyword from the decision newline.
gen2 exploration programs (terminal and, as control, plain). Target point and token roles as in r2e03.
 1. Attention from d to kw, per head (all layers).
 2. The edge d->kw removed in one layer only (all heads of it; exact, EdgeBlock), each layer 0-27.
 3. Per-head knock-out of kw for the top heads by attention (output recomputed from clean attention, Tap), alone
    and together (KWG); and KWG's layer-matched random heads (8 draws).
Measures: q(current level) and log q(true) at the target point; neutral programs for the terminal gap.
Usage: python r2e04_kwheads.py <n_programs> <kind> [top heads, comma-separated, else chosen from the first 60]"""
import sys, json, time
from r2common import *

n, kind = int(sys.argv[1]), sys.argv[2]
m = M("qwen3-06b-bf16", attn="eager"); eb = EdgeBlock(m); tap = Tap(m, list(range(m.L)))
progs = [g for g in r2_sources("gen2") if g["last"] == kind]; random.Random(24).shuffle(progs)


def setup(g, text):
    ids, pts = prepare(m, text)
    if not pts: return None
    pp = [p for p in pts if p["line"] == g["target_line"] and max(p["u"]) == 0]
    if not pp: return None
    p = pp[0]; offs = m.tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    lines = text.split("\n"); tl = g["target_line"]; ls = sum(len(x) + 1 for x in lines[:tl - 2])
    ind = len(lines[tl - 2]) - len(lines[tl - 2].lstrip(" "))
    kw = next(i for i, (a, b) in enumerate(offs) if a <= ls + ind < b)
    return ids, p, kw


def meas(lp, p):
    q = RootMasses(m, [p])(lp)[0]; K = p["K"]; cur = p["stack"][-1]
    return [float(q[K.index(cur)]), float(np.log(q[K.index(p["true"])]))]


items = []
for g in progs:
    if len(items) >= n: break
    s = setup(g, g["text"])
    if s is None: continue
    s2 = setup(g, g["text_neutral"]) if g.get("text_neutral") else None
    items.append((g, s, s2))
# 1. attention d->kw
att = np.zeros((m.L, m.H));
for g, (ids, p, kw), _ in items[:60]:
    tap.rows = [p["d"]]; tap.on = True; m.logprobs(ids, positions=[(0, p["d"])]); tap.on = False
    att += np.array([[float(tap.A[l][h, 0, kw]) for h in range(m.H)] for l in range(m.L)])
att /= min(60, len(items))
order = np.dstack(np.unravel_index(np.argsort(-att, axis=None), att.shape))[0]
if len(sys.argv) > 3: TOP = sys.argv[3].split(",")
else: TOP = [f"L{l}H{h}" for l, h in order[:8]]
rng = random.Random(77); RAND = []
for k in range(8):
    RAND.append([f"L{hn(x)[0]}H{rng.choice([y for y in range(m.H) if f'L{hn(x)[0]}H{y}' not in TOP])}" for x in TOP])
rec = []; t0 = time.time()
for g, (ids, p, kw), s2 in items:
    if time.time() - t0 > 1000: print("time limit"); break
    d = p["d"]; T = len(ids)
    tap.rows = [d]; tap.on = True; lp = m.logprobs(ids, positions=[(0, d)]); tap.on = False
    r = dict(kw=g["term_kw"], base=meas(lp, p))
    for l in range(m.L):
        eb.edges = [({l}, [d], [kw])]; r[f"lay{l}"] = meas(m.logprobs(ids, positions=[(0, d)]), p); eb.edges = []
    keep = torch.ones(T, device=DEV); keep[kw] = 0
    def ko(heads):
        m.zpatch = {}
        for hname in heads:
            l, h = hn(hname); m.zpatch.setdefault(l, []).append((h, torch.tensor([d]), tap.head_out(l, h, 0, keep)[None]))
        out = meas(m.logprobs(ids, positions=[(0, d)]), p); m.zpatch = {}; return out
    for hname in TOP: r[f"ko_{hname}"] = ko([hname])
    r["ko_KWG"] = ko(TOP)
    for k, rh in enumerate(RAND): r[f"ko_rand{k}"] = ko(rh)
    eb.edges = [(set(range(m.L)), [d], [kw])]; r["blk_all"] = meas(m.logprobs(ids, positions=[(0, d)]), p); eb.edges = []
    if s2 is not None and s2[1]["K"] == p["K"]:
        r["neutral"] = meas(m.logprobs(s2[0], positions=[(0, s2[1]["d"])]), s2[1])
    rec.append(r)
res = dict(n=len(rec), kind=kind, top=TOP, rand=RAND, att_top=[(f"L{l}H{h}", round(float(att[l, h]), 3)) for l, h in order[:20]], rec=rec)
(OUT2 / f"r2e04_{kind}_{n}.json").write_text(json.dumps(res))
print("top heads by attention d->kw:", res["att_top"][:12])
for grp in ["all", "return", "raise", "pass", "break", "continue", None]:
    rs = [r for r in rec if grp == "all" or r["kw"] == grp]
    if not rs: continue
    b = np.mean([r["base"][0] for r in rs]); nv = [r["neutral"][0] for r in rs if "neutral" in r]
    print(f"== {grp} n={len(rs)} base pcur={b:.3f} neutral pcur={np.mean(nv) if nv else float('nan'):.3f}")
    ks = ["blk_all", "ko_KWG"] + [f"ko_{h}" for h in TOP] + [f"lay{l}" for l in range(m.L)]
    for k in ks: print(f"   {k:12s} pcur={np.mean([r[k][0] for r in rs]):.3f} lq={np.mean([r[k][1] for r in rs]):.3f}")
    print("   rand KWG pcur:", [round(float(np.mean([r[f'ko_rand{k}'][0] for r in rs])), 3) for k in range(8)])
