"""D3 confirmation c01 (PROTOCOL.md): group mean-ablation at the decision token on a SEALED part.
Same computation as e06_group.py, but: all chunks of the part; fixed seeds; each run (base, a named group, or a
random draw) saved to outputs/conf/c01_<source>.json as it finishes, so the job resumes across lock holds.
Random groups: for each head of the named group, a head drawn uniformly from the other heads of the same layer
(random.Random(1000 + draw)); 20 draws per group.
Usage: python c01_ablate.py <source: newcode|gen> [--time-limit S] [--smoke]   (--smoke: exploration part, 16 chunks)"""
import sys, json, time
from common import *

MODEL = "qwen3-06b-bf16"
DED = ["L23H2", "L17H3", "L17H4", "L17H5", "L18H6"]
TERM = ["L11H9", "L5H6", "L17H6", "L16H2", "L23H6", "L22H8", "L23H3", "L25H9", "L19H6"]
GROUPS = {"DED": DED, "TERM": TERM, "TERMDED": TERM + DED}
DRAWS = {"DED": 20, "TERM": 20, "TERMDED": 0}
source = sys.argv[1]; smoke = "--smoke" in sys.argv
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
part = "explore" if smoke else "conf"
od = OUT / ("conf_smoke" if smoke else "conf"); od.mkdir(exist_ok=True)
outp = od / f"c01_{source}.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
m = M(MODEL, attn="sdpa"); t0 = time.time()
chunks = load_chunks(source, part)
if smoke: chunks = chunks[:16]
items = []
for ci, c in enumerate(chunks):
    ids, pts = prepare(m, c["text"])
    if not pts: continue
    pts = [p for p in pts if max(p["u"]) == 0]
    for p in pts: p["_cluster"] = c.get("repo", c.get("seed", ci))
    if pts: items.append((ids, pts))
B = 8; batches = []; pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
for i in range(0, len(items), B):
    grp = items[i:i + B]; T = max(len(x[0]) for x in grp)
    x = torch.tensor([ids + [pad] * (T - len(ids)) for ids, _ in grp])
    pos = [(r, p["d"]) for r, (_, pts) in enumerate(grp) for p in pts]
    bp = [p for _, pts in grp for p in pts]
    batches.append((x, pos, bp, RootMasses(m, bp)))
allp = [p for b in batches for p in b[2]]
cls = np.array([("header" if p["header"] else "same" if p["cls"] == "same" else "term" if p["terminal"] else "dedent") for p in allp])
res["n_points"] = {c: int((cls == c).sum()) for c in ["header", "same", "term", "dedent"]}; res["n_chunks_used"] = len(items)
zs = {l: 0 for l in range(m.L)}; nn = 0
with torch.no_grad():
    for x, pos, pts, rm in batches:
        for r in range(x.shape[0]):
            rp = [p for rr, p in pos if rr == r]
            if not rp: continue
            m.capture = True; m.logprobs(x[r:r + 1], positions=[(0, rp[0])]); m.capture = False
            for l in range(m.L): zs[l] = zs[l] + m.zcap[l][rp].sum(0)
            nn += len(rp)
zm = {l: zs[l] / nn for l in zs}


clusters = [str(p["_cluster"]) for p in allp]
np.save(od / f"c01_{source}_meta.npy", np.array([cls, np.array(clusters)]))


def run(heads, save=None):
    lq, ps, po = [], [], []
    for x, pos, pts, rm in batches:
        rows = torch.tensor([r for r, _ in pos]); cols = torch.tensor([p for _, p in pos]); m.zpatch = {}
        for hname in heads:
            l, h = (int(v) for v in hname[1:].split("H")); w = m.dhl[l]
            m.zpatch.setdefault(l, []).append((h, cols, zm[l][h * w:(h + 1) * w].expand(len(cols), -1), rows))
        lp = m.logprobs(x, positions=pos); m.zpatch = {}
        for q, p in zip(rm(lp), pts):
            K = p["K"]; st = p["stack"]; cur = st[-1]
            lq.append(float(np.log(q[K.index(p["true"])]))); ps.append(float(q[K.index(cur)]) if cur in K else 0.0)
            po.append(float(sum(q[i] for i, k in enumerate(K) if k in st[:-1])))
    if save: np.save(od / f"c01_{source}_points_{save}.npy", np.array([lq, ps, po]))
    a = lambda v: {c: float(np.mean(np.array(v)[cls == c])) for c in ["header", "same", "term", "dedent"]}
    return dict(lq=a(lq), p_same=a(ps), p_open=a(po))


todo = [("base", [])]
for g, hs in GROUPS.items():
    todo.append((g, hs))
    for k in range(DRAWS[g]):
        rng = random.Random(1000 + k)
        rh = []
        for hname in hs:
            l = int(hname[1:].split("H")[0])
            rh.append(f"L{l}H{rng.choice([x for x in range(m.H) if f'L{l}H{x}' not in hs])}")
        todo.append((f"{g}_rand{k}", rh))
for key, heads in todo:
    if key in res: continue
    if time.time() - t0 > tlim: print("time limit; resume later"); break
    res[key] = dict(heads=heads, **run(heads, save=None if "_rand" in key else key)); outp.write_text(json.dumps(res))
done = all(k in res for k, _ in todo)
print(json.dumps(dict(source=source, part=part, done=done, runs=sum(1 for k, _ in todo if k in res), of=len(todo), secs=round(time.time() - t0))))
