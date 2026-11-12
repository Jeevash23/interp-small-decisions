"""D3 e10 (exploratory): do some components carry a fixed "four-space step" prior? Mean-ablate component groups
at the decision position (means over all decision positions of the material) and measure, by indent width of
generated exploration programs (2, 3, 4, mixed) and for natural code, the change in mean log q(true) and
top-1 for terminal dedents, other dedents and headers. A step prior predicts: ablation hurts 4-space code
and natural code, and helps 2-space, 3-space and mixed code.
Groups: NAME=L25H9,mlp26,... (heads LxHy, MLPs mlpN). Random head groups matched by layer for head-only groups.
Usage: python e10_width_ablate.py <model> <n_gen> <n_nat> NAME=... [NAME=...] [--draws 8]"""
import sys, json, time
from common import *

name, ngen, nnat = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
groups = {}
for a in sys.argv[4:]:
    if "=" in a:
        g, cs = a.split("="); groups[g] = cs.split(",")
draws = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 8
m = M(name, attn="sdpa"); t0 = time.time()


def load(source, n, seed):
    ch = load_chunks(source, "explore"); random.Random(seed).shuffle(ch); items = []
    for c in ch:
        ids, pts = prepare(m, c["text"])
        if not pts: continue
        pts = [p for p in pts if max(p["u"]) == 0]
        for p in pts: p["w"] = str(c.get("width", "nat"))
        if pts: items.append((ids, pts))
        if len(items) >= n: break
    return items


items = load("gen", ngen, 11) + load("newcode", nnat, 12)
B = 8; batches = []; pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
for i in range(0, len(items), B):
    grp = items[i:i + B]; T = max(len(x[0]) for x in grp)
    x = torch.tensor([ids + [pad] * (T - len(ids)) for ids, _ in grp])
    pos = [(r, p["d"]) for r, (_, pts) in enumerate(grp) for p in pts]
    bp = [p for _, pts in grp for p in pts]
    batches.append((x, pos, bp, RootMasses(m, bp)))
allp = [p for b in batches for p in b[2]]
key = np.array([("header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent") + "|" + p["w"] for p in allp])
zs = {l: 0 for l in range(m.L)}; ms = {l: 0 for l in range(m.L)}; nn = 0
with torch.no_grad():
    for x, pos, pts, rm in batches:
        for r in range(x.shape[0]):
            rp = [p for rr, p in pos if rr == r]
            if not rp: continue
            m.capture = True; m.logprobs(x[r:r + 1], positions=[(0, rp[0])]); m.capture = False
            for l in range(m.L): zs[l] = zs[l] + m.zcap[l][rp].sum(0); ms[l] = ms[l] + m.mcap[l][rp].sum(0)
            nn += len(rp)
zm = {l: zs[l] / nn for l in zs}; mm = {l: ms[l] / nn for l in ms}


def run(comps):
    lq, t1 = [], []
    for x, pos, pts, rm in batches:
        rows = torch.tensor([r for r, _ in pos]); cols = torch.tensor([p for _, p in pos]); m.zpatch, m.mpatch = {}, {}
        for c in comps:
            if c.startswith("mlp"):
                l = int(c[3:]); m.mpatch[l] = [(cols, mm[l].expand(len(cols), -1), rows)]
            else:
                l, h = (int(v) for v in c[1:].split("H")); w = m.dhl[l]
                m.zpatch.setdefault(l, []).append((h, cols, zm[l][h * w:(h + 1) * w].expand(len(cols), -1), rows))
        lp = m.logprobs(x, positions=pos); m.zpatch, m.mpatch = {}, {}
        for q, p in zip(rm(lp), pts):
            i = p["K"].index(p["true"]); lq.append(float(np.log(q[i]))); t1.append(float(np.argmax(q) == i))
    lq, t1 = np.array(lq), np.array(t1)
    return {k: dict(lq=float(lq[key == k].mean()), top1=float(t1[key == k].mean()), n=int((key == k).sum())) for k in sorted(set(key))}


res = dict(base=run([]), groups={})
for g, comps in groups.items():
    o = run(comps); r = []
    if all(c.startswith("L") for c in comps):
        rng = random.Random(len(g) * 13)
        for _ in range(draws):
            rc = []
            for c in comps:
                l, h = (int(v) for v in c[1:].split("H"))
                rc.append(f"L{l}H{rng.choice([x for x in range(m.H) if f'L{l}H{x}' not in comps])}")
            r.append(run(rc))
    res["groups"][g] = dict(comps=comps, obs=o, random=r)
    for k in sorted(o):
        if k.split("|")[0] in ("term", "dedent", "header"):
            b = res["base"][k]; rr = [x[k]["lq"] - res["base"][k]["lq"] for x in r]
            print(f"{g} {k:16s} n={b['n']:5d} dlq {o[k]['lq'] - b['lq']:+.3f} dtop1 {o[k]['top1'] - b['top1']:+.3f}" + (f" random dlq {np.mean(rr):+.3f} [{min(rr):+.3f},{max(rr):+.3f}]" if rr else ""), flush=True)
(OUT / f"e10_{name}_{'_'.join(groups)}.json").write_text(json.dumps(res))
print("secs", round(time.time() - t0))
