"""D3 round 2 confirmation r2c03 (PROTOCOL-2.md): group mean-ablation of the second model's block-end heads on SEALED
natural code (data/r2code/part-held.jsonl). Same computation as round 1's e06_group.py / c01_ablate.py: the named
heads' outputs at every decision token are replaced by their mean over all decision tokens of the part. Classes as
round 1 (header / same / term / dedent); measures per class: mean log q(true), mass on the current level, mass on
shallower open levels. Random control groups: for each head of the group, a head drawn uniformly from the other
heads of the same layer, random.Random(3000 + k), k = 0..19. Each run is saved as it finishes (resumable).
Refuses to read the sealed part unless PROTOCOL-2.md is read-only. --calib <n>: round 1's newcode exploration part
(n chunks), to set thresholds before freezing. Chunks longer than 1400 tokens are skipped (as r2c01).
Usage: python r2c03_group.py <model> NAME=L13H1,L10H12,... [--time-limit S] [--calib n]
Output: outputs/r2/conf/r2c03_<model>_<NAME>.json (or conf_calib_newcode/)."""
import sys, json, time, stat
from r2common import *

MODEL = sys.argv[1]; g, hs = sys.argv[2].split("="); heads = [hn(h) for h in hs.split(",")]
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
if "--calib" in sys.argv:
    chunks = r2_sources("r1code")[:int(sys.argv[sys.argv.index("--calib") + 1])]; od = OUT2 / "conf_calib_newcode"
else:
    pf = R2 / "PROTOCOL-2.md"
    if not pf.exists() or (pf.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
        raise SystemExit("refusing to read the sealed part: PROTOCOL-2.md is not frozen")
    chunks = [json.loads(l) for l in open(DATA / "r2code" / "part-held.jsonl")]; od = OUT2 / "conf"
od.mkdir(parents=True, exist_ok=True); outp = od / f"r2c03_{MODEL}_{g}.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
m = M(MODEL, attn="sdpa"); t0 = time.time()
items = []
for ci, c in enumerate(chunks):
    ids, pts = prepare(m, c["text"])
    if not pts or len(ids) > 1400: continue
    pts = [p for p in pts if max(p["u"]) == 0]
    for p in pts: p["_cl"] = c.get("repo", ci)
    if pts: items.append((ids, pts))
B = 8; batches = []
pad = m.tok.pad_token_id if m.tok.pad_token_id is not None else 0
for i in range(0, len(items), B):
    grp = items[i:i + B]; T = max(len(x[0]) for x in grp)
    x = torch.tensor([ids + [pad] * (T - len(ids)) for ids, _ in grp])
    pos = [(r, p["d"]) for r, (_, pts) in enumerate(grp) for p in pts]
    bp = [p for _, pts in grp for p in pts]
    batches.append((x, pos, bp, RootMasses(m, bp)))
allpts = [p for b in batches for p in b[2]]
cls = np.array([point_class(p) for p in allpts]); CL = ["header", "same", "term", "dedent"]
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


def run(hl):
    lq, ps, po = [], [], []
    for x, pos, pts, rm in batches:
        rows = torch.tensor([r for r, _ in pos]); cols = torch.tensor([p for _, p in pos])
        m.zpatch = {}
        for l, h in hl:
            w = m.dhl[l]; m.zpatch.setdefault(l, []).append((h, cols, zmean[l][h * w:(h + 1) * w].expand(len(cols), -1), rows))
        lp = m.logprobs(x, positions=pos); m.zpatch = {}
        for q, p in zip(rm(lp), pts):
            K = p["K"]; st = p["stack"]; cur = st[-1]
            lq.append(float(np.log(q[K.index(p["true"])]))); ps.append(float(q[K.index(cur)]) if cur in K else 0.0)
            po.append(float(sum(q[i] for i, k in enumerate(K) if k in st[:-1])))
    a = lambda v: {c: float(np.mean(np.array(v)[cls == c])) for c in CL}
    return dict(lq=a(lq), p_same=a(ps), p_open=a(po), per_point_lq=lq)


res.setdefault("n", {c: int((cls == c).sum()) for c in CL}); res.setdefault("cls", cls.tolist())
res.setdefault("cluster", [str(p["_cl"]) for p in allpts]); res.setdefault("heads", [f"L{l}H{h}" for l, h in heads])
def draw(k):
    rng = random.Random(3000 + k); return [(l, rng.choice([h for h in range(m.H) if (l, h) not in heads])) for l, _ in heads]


todo = [("base", [])] + [(g, heads)] + [(f"rand{k}", draw(k)) for k in range(20)]
with torch.no_grad():
    for key, hl in todo:
        if key in res: continue
        if time.time() - t0 > tlim: print("time limit; resume later", flush=True); break
        r = run(hl); r["heads"] = [f"L{l}H{h}" for l, h in hl]; res[key] = r
        outp.write_text(json.dumps(res)); print(key, json.dumps(r["lq"]), flush=True)
print("secs", round(time.time() - t0), "runs", sum(1 for k, _ in todo if k in res), "of", len(todo))
