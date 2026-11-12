"""D3 round 2, r2e06 (exploratory, question 1): what makes the block-end head look at the enclosing header when a
block ends? Every attention head and MLP of layers 0..L-1 is mean-ablated in turn at the decision tokens only (means
over decision tokens of the first 40 items), and the block-end head's attention to the enclosing header's ':\\n'
token is read, at unmarked dedents and at same-level lines, together with log q(true) at unmarked dedents.
Usage: python r2e06_qcomp.py <source> <n_chunks> [model head]"""
import sys, json, time
from r2common import *

src, nch = sys.argv[1], int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3-06b-bf16"; HEAD = sys.argv[4] if len(sys.argv) > 4 else "L23H2"
LH, HH = hn(HEAD)
m = M(MODEL, attn="eager"); tap = Tap(m, [LH])
chunks = r2_sources(src); random.Random(26).shuffle(chunks)
items = []
for c in chunks:
    if len(items) >= nch: break
    ids, pts = prepare(m, c["text"])
    if not pts or len(ids) > 900: continue
    offs = m.tok(c["text"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    pts = [p for p in pts if max(p["u"]) == 0 and point_class(p) in ("dedent", "same")]
    keep = []
    for p in pts:
        oh = [(ind, ti) for ind, ti in open_headers(c["text"], offs, p) if ti is not None and ti < p["d"]]
        if oh: p["E"] = oh[-1][1]; p["k"] = point_class(p); keep.append(p)
    if keep: items.append((ids, keep))
zs, ms, nz = {}, {}, 0
for ids, pts in items[:40]:
    m.capture = True; m.logprobs(ids, positions=[(0, pts[0]["d"])]); m.capture = False
    d = torch.tensor([p["d"] for p in pts])
    for l in range(LH):
        zs[l] = zs.get(l, 0) + m.zcap[l][d].float().sum(0); ms[l] = ms.get(l, 0) + m.mcap[l][d].float().sum(0)
    nz += len(pts)
zm = {l: zs[l] / nz for l in zs}; mm = {l: ms[l] / nz for l in ms}
comps = ["base"] + [f"L{l}H{h}" for l in range(LH) for h in range(m.H)] + [f"MLP{l}" for l in range(LH)]
acc = {c: dict(a_ded=0.0, a_same=0.0, lq=0.0) for c in comps}; nd = ns = 0; t0 = time.time()
for ids, pts in items:
    d = torch.tensor([p["d"] for p in pts]); rm = RootMasses(m, pts)
    isd = [p["k"] == "dedent" for p in pts]; nd += sum(isd); ns += len(pts) - sum(isd)
    for c in comps:
        m.zpatch, m.mpatch = {}, {}
        if c.startswith("MLP"):
            l = int(c[3:]); m.mpatch[l] = [(d, mm[l].expand(len(pts), -1))]
        elif c != "base":
            l, h = hn(c); w = m.dhl[l]; m.zpatch[l] = [(h, d, zm[l][h * w:(h + 1) * w].expand(len(pts), -1))]
        tap.rows = [p["d"] for p in pts]; tap.on = True
        lp = m.logprobs(ids, positions=[(0, p["d"]) for p in pts]); tap.on = False; m.zpatch, m.mpatch = {}, {}
        qs = rm(lp)
        for qi, p in enumerate(pts):
            a = float(tap.A[LH][HH, qi, p["E"]])
            if isd[qi]:
                acc[c]["a_ded"] += a; acc[c]["lq"] += float(np.log(qs[qi][p["K"].index(p["true"])]))
            else: acc[c]["a_same"] += a
    if time.time() - t0 > 1100: print("time limit", flush=True); break
res = {c: dict(a_ded=v["a_ded"] / max(nd, 1), a_same=v["a_same"] / max(ns, 1), lq=v["lq"] / max(nd, 1)) for c, v in acc.items()}
b = res["base"]
out = dict(model=MODEL, head=HEAD, source=src, n_dedent=nd, n_same=ns, base=b, comps=res)
(OUT2 / f"r2e06_{MODEL}_{HEAD}_{src}_{nch}.json").write_text(json.dumps(out))
print(json.dumps(dict(n_dedent=nd, n_same=ns, base=b)))
rank = sorted(((v["a_ded"] - b["a_ded"], c) for c, v in res.items() if c != "base"))
print("largest drops in attention to the enclosing header end at unmarked dedents:")
for dv, c in rank[:15]: print(f"  {c:8s} a_ded {dv:+.3f}  a_same {res[c]['a_same'] - b['a_same']:+.3f}  lq {res[c]['lq'] - b['lq']:+.3f}")
print("largest rises:")
for dv, c in rank[-5:]: print(f"  {c:8s} a_ded {dv:+.3f}  a_same {res[c]['a_same'] - b['a_same']:+.3f}  lq {res[c]['lq'] - b['lq']:+.3f}")
