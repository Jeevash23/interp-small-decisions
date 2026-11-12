"""D3 e02 (exploratory): mean-ablation scan of every attention head and MLP at the decision position.
For each component, its output at every decision position is replaced by its mean over decision positions;
effect = change in log q(true indent) by class (header / same / terminal dedent / other dedent), and in
shallower-open mass at terminal dedents. Only points whose candidates are all identified at the root token
(true of single-token-indent tokenisers) are used.
Usage: python e02_ablate.py <model> <source> <n_chunks> [--time-limit S]"""
import sys, json, time
from common import *

name, source, nch = sys.argv[1], sys.argv[2], int(sys.argv[3])
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
outp = OUT / f"e02_{name}_{source}_{nch}.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
m = M(name, attn="sdpa"); t0 = time.time()
chunks = load_chunks(source, "explore")
rng = random.Random(1); rng.shuffle(chunks)
items = []
for c in chunks:
    ids, pts = prepare(m, c["text"])
    if not pts: continue
    pts = [p for p in pts if max(p["u"]) == 0]
    if pts: items.append((ids, pts))
    if len(items) >= nch: break
B = 8
batches = []
for i in range(0, len(items), B):
    grp = items[i:i + B]; T = max(len(x[0]) for x in grp)
    pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
    x = torch.tensor([ids + [pad] * (T - len(ids)) for ids, _ in grp])
    pos = [(r, p["d"]) for r, (_, pts) in enumerate(grp) for p in pts]
    bp = [p for _, pts in grp for p in pts]
    batches.append((x, pos, bp, RootMasses(m, bp)))
allpts = [p for _, _, ps, _ in batches for p in ps]
cls = np.array([("header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent") for p in allpts])


def masses_slow(lp, pts):
    out = []
    for q, p in zip(lp, pts):
        v = []
        for el in p["paths"]:
            kind, t = el[0]
            v.append(float(q[t]) if kind == "t" else float(torch.logsumexp(q[m.set_ids(t)], 0)))
        v = np.exp(np.array(v)); out.append(v / v.sum())
    return out


def metrics():
    lq, sh = [], []
    for x, pos, pts, rm in batches:
        lp = m.logprobs(x, positions=pos)
        for q, p in zip(rm(lp), pts):
            lq.append(float(np.log(q[p["K"].index(p["true"])])))
            st = p["stack"]; sh.append(float(sum(q[i] for i, k in enumerate(p["K"]) if k in st[:-1])))
    return np.array(lq), np.array(sh)


# means of each head's output and each MLP's output at decision positions
zsum = {l: 0 for l in range(m.L)}; msum = {l: 0 for l in range(m.L)}; nn = 0
with torch.no_grad():
    for x, pos, pts, rm in batches:
        for r in range(x.shape[0]):
            rp = [p for rr, p in pos if rr == r]
            if not rp: continue
            m.capture = True; m.logprobs(x[r:r + 1], positions=[(0, rp[0])]); m.capture = False
            for l in range(m.L):
                zsum[l] = zsum[l] + m.zcap[l][rp].sum(0); msum[l] = msum[l] + m.mcap[l][rp].sum(0)
            nn += len(rp)
zmean = {l: zsum[l] / nn for l in zsum}; mmean = {l: msum[l] / nn for l in msum}
base_lq, base_sh = metrics()
summ = lambda lq, sh: {c: float(lq[cls == c].mean()) for c in ["header", "same", "term", "dedent"]} | {"term_shopen": float(sh[cls == "term"].mean())}
res["base"] = summ(base_lq, base_sh); res["n"] = {c: int((cls == c).sum()) for c in ["header", "same", "term", "dedent"]}
comps = [("mlp", l, -1) for l in range(m.L)] + [("head", l, h) for l in range(m.L) for h in range(m.H)]
for kind, l, h in comps:
    key = f"{kind}{l}" if kind == "mlp" else f"L{l}H{h}"
    if key in res: continue
    if time.time() - t0 > tlim: print("time limit"); break
    lq, sh = [], []
    for x, pos, pts, rm in batches:
        rows = torch.tensor([r for r, _ in pos]); cols = torch.tensor([p for _, p in pos])
        if kind == "mlp":
            m.mpatch = {l: [(cols, mmean[l].expand(len(cols), -1), rows)]}
        else:
            w = m.dhl[l]; m.zpatch = {l: [(h, cols, zmean[l][h * w:(h + 1) * w].expand(len(cols), -1), rows)]}
        lp = m.logprobs(x, positions=pos)
        for q, p in zip(rm(lp), pts):
            lq.append(float(np.log(q[p["K"].index(p["true"])])))
            st = p["stack"]; sh.append(float(sum(q[i] for i, k in enumerate(p["K"]) if k in st[:-1])))
    m.mpatch, m.zpatch = {}, {}
    s = summ(np.array(lq), np.array(sh))
    res[key] = {k: s[k] - res["base"][k] for k in s}
    outp.write_text(json.dumps(res))
print("done", len(res), round(time.time() - t0))
