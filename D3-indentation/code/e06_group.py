"""D3 e06 (exploratory): mean-ablation of named head groups at the decision position, against random head
groups matched by layer (for each head, a random other head in the same layer; 20 draws).
Measures by class (header / same / term / dedent): mean log q(true), mass on the current level (p_same),
mass on shallower open levels (p_open), mass deeper (p_deep).
Usage: python e06_group.py <model> <source> <n_chunks> NAME=L19H3,L20H14,... [NAME=...] [--draws 20]"""
import sys, json, time
from common import *

name, source, nch = sys.argv[1], sys.argv[2], int(sys.argv[3])
groups = {}
for a in sys.argv[4:]:
    if "=" in a:
        g, hs = a.split("=")
        groups[g] = [tuple(int(x) for x in h[1:].split("H")) for h in hs.split(",")]
draws = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 20
m = M(name, attn="sdpa"); t0 = time.time()
chunks = load_chunks(source, "explore"); random.Random(1).shuffle(chunks)
items = []
for c in chunks:
    ids, pts = prepare(m, c["text"])
    if not pts: continue
    pts = [p for p in pts if max(p["u"]) == 0]
    if pts: items.append((ids, pts))
    if len(items) >= nch: break
B = 8; batches = []
pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
for i in range(0, len(items), B):
    grp = items[i:i + B]; T = max(len(x[0]) for x in grp)
    x = torch.tensor([ids + [pad] * (T - len(ids)) for ids, _ in grp])
    pos = [(r, p["d"]) for r, (_, pts) in enumerate(grp) for p in pts]
    bp = [p for _, pts in grp for p in pts]
    batches.append((x, pos, bp, RootMasses(m, bp)))
allpts = [p for b in batches for p in b[2]]
cls = np.array([("header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent") for p in allpts])
zsum = {l: 0 for l in range(m.L)}; nn = 0
with torch.no_grad():
    for x, pos, pts, rm in batches:
        for r in range(x.shape[0]):
            rp = [p for rr, p in pos if rr == r]
            if not rp: continue
            m.capture = True; m.logprobs(x[r:r + 1], positions=[(0, rp[0])]); m.capture = False
            for l in range(m.L): zsum[l] = zsum[l] + m.zcap[l][rp].sum(0)
            nn += len(rp)
zmean = {l: zsum[l] / nn for l in zsum}


def run(heads):
    lq, ps, po, pd = [], [], [], []
    for x, pos, pts, rm in batches:
        rows = torch.tensor([r for r, _ in pos]); cols = torch.tensor([p for _, p in pos])
        m.zpatch = {}
        for l, h in heads:
            w = m.dhl[l]; m.zpatch.setdefault(l, []).append((h, cols, zmean[l][h * w:(h + 1) * w].expand(len(cols), -1), rows))
        lp = m.logprobs(x, positions=pos); m.zpatch = {}
        for q, p in zip(rm(lp), pts):
            K = p["K"]; st = p["stack"]; cur = st[-1]
            lq.append(float(np.log(q[K.index(p["true"])]))); ps.append(float(q[K.index(cur)]) if cur in K else 0.0)
            po.append(float(sum(q[i] for i, k in enumerate(K) if k in st[:-1]))); pd.append(float(sum(q[i] for i, k in enumerate(K) if k > cur)))
    a = lambda v: {c: float(np.mean(np.array(v)[cls == c])) for c in ["header", "same", "term", "dedent"]}
    return dict(lq=a(lq), p_same=a(ps), p_open=a(po), p_deep=a(pd))


res = dict(n={c: int((cls == c).sum()) for c in ["header", "same", "term", "dedent"]}, base=run([]), groups={})
print("base", json.dumps(res["base"]), flush=True)
for g, heads in groups.items():
    obs = run(heads); rnd = []
    rng = random.Random(hash(g) % 1000)
    for _ in range(draws):
        rh = [(l, rng.choice([h for h in range(m.H) if (l, h) not in heads])) for l, _ in heads]
        rnd.append(run(rh))
    res["groups"][g] = dict(heads=[f"L{l}H{h}" for l, h in heads], obs=obs, random=rnd)
    for meas in ["lq", "p_same", "p_open"]:
        for c in ["same", "term", "dedent", "header"]:
            r = np.array([x[meas][c] for x in rnd]); o = obs[meas][c]; b = res["base"][meas][c]
            print(f"{g} {meas} {c}: base {b:.3f} obs {o:.3f} random {r.mean():.3f} [{r.min():.3f},{r.max():.3f}]", flush=True)
(OUT / f"e06_{name}_{source}_{nch}_{'_'.join(groups)}.json").write_text(json.dumps(res))
print("secs", round(time.time() - t0))
