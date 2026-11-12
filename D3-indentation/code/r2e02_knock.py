"""D3 round 2, r2e02 (exploratory, questions 1-2): attention knock-out and OV readout for the block-end heads.
For every decision point (root-identified) of a source:
 conditions (head outputs at the decision token recomputed from clean attention with some sources removed):
   base;  ko23_encl: L23H2 loses the enclosing header's last token (its ':\\n');  ko23_encl_line: L23H2 loses the
   whole enclosing header line;  ko23_other_hdr: L23H2 loses the last token of the nearest earlier header line
   that is not the enclosing one (control);  koDED_encl_line: all five DED heads lose the enclosing header
   line;  ko23_rand_encl: a random other layer-23 head loses the enclosing header line (8 draws, control);
   ablate17: L17H3, L17H4, L17H5 and L18H6 mean-ablated (for the query-composition read-out below).
 per point: log q(true), q(current level), open-shallower mass; L23H2 attention to the enclosing header end
 in base and in ablate17; L23H2 OV read-out from the enclosing header end: which candidate indent its output
 favours (direct logit through the final norm scale at that point).
Usage: python r2e02_knock.py <source> <n_chunks>"""
import sys, json, time
from r2common import *

src, nch = sys.argv[1], int(sys.argv[2]); name = "qwen3-06b-bf16"
m = M(name, attn="eager"); tap = Tap(m, [17, 18, 23])
chunks = r2_sources(src); random.Random(22).shuffle(chunks)
DED = ["L23H2", "L17H3", "L17H4", "L17H5", "L18H6"]; A17 = ["L17H3", "L17H4", "L17H5", "L18H6"]
W_U = m.model.get_output_embeddings().weight; normw = m.model.model.norm.weight
items = []
for c in chunks:
    if len(items) >= nch: break
    ids, pts = prepare(m, c["text"])
    if not pts or len(ids) > 900: continue
    pts = [p for p in pts if max(p["u"]) == 0]
    if not pts: continue
    offs = m.tok(c["text"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    for p in pts: p["R"] = line_roles(c["text"], offs, p, p["d"]); p["kcls"] = point_class(p); p["OH"] = open_headers(c["text"], offs, p)
    items.append((ids, pts, c))
# mean head outputs at decision tokens (first 60 items) for ablate17
zs = {}; nz = 0
for ids, pts, c in items[:60]:
    m.capture = True; m.logprobs(ids, positions=[(0, pts[0]["d"])]); m.capture = False
    d = torch.tensor([p["d"] for p in pts])
    for l in (17, 18): zs[l] = zs.get(l, 0) + m.zcap[l][d].float().sum(0)
    nz += len(pts)
zm = {l: zs[l] / nz for l in zs}
rec = []; t0 = time.time()


def qstats(lp, pts):
    rm = RootMasses(m, pts); out = []
    for q, p in zip(rm(lp), pts):
        K = p["K"]; st = p["stack"]; cur = st[-1]
        out.append((float(np.log(q[K.index(p["true"])])), float(q[K.index(cur)]) if cur in K else 0.0,
                    float(sum(q[i] for i, k in enumerate(K) if k in st[:-1]))))
    return out


def last_tok(R, role):
    xs = R.get(role, []); return xs[-1:] if xs else []


for ii, (ids, pts, c) in enumerate(items):
    T = len(ids); dpos = [p["d"] for p in pts]
    tap.rows = dpos; tap.on = True; m.capture = True
    base_lp = m.logprobs(ids, positions=[(0, d) for d in dpos]); tap.on = False; m.capture = False
    A23 = tap.A[23][2].clone()
    hdr_ends = []  # earlier header-line ends (':\n' tokens) per point: tokens whose string ends with ':\n' or ':'
    conds = {}
    def kovals(hname, getpos):
        l, h = hn(hname); vals = []
        for qi, p in enumerate(pts):
            keep = torch.ones(T, device=DEV); pos = getpos(p)
            if pos: keep[torch.tensor(pos, device=DEV)] = 0
            vals.append(tap.head_out(l, h, qi, keep[:tap.A[l].shape[-1]]))
        return torch.stack(vals)
    def other_hdr(p):
        enc = set(p["R"]["encl"] + p["R"]["encl_start"])
        cand = [i for i in range(1, p["d"]) if m.vs[ids[i]].rstrip(" ").endswith(":\n") and i not in enc]
        return cand[-1:] if cand else []
    encl_end = lambda p: last_tok(p["R"], "encl")
    encl_line = lambda p: p["R"]["encl"] + p["R"]["encl_start"]
    conds["ko23_encl"] = [("L23H2", kovals("L23H2", encl_end))]
    conds["ko23_encl_line"] = [("L23H2", kovals("L23H2", encl_line))]
    conds["ko23_other_hdr"] = [("L23H2", kovals("L23H2", other_hdr))]
    conds["koDED_encl_line"] = [(h, kovals(h, encl_line)) for h in DED]
    rng = random.Random(500 + ii)
    for k in range(8):
        rh = f"L23H{rng.choice([x for x in range(m.H) if x != 2])}"
        conds[f"ko23_rand_encl_{k}"] = [(rh, kovals(rh, encl_line))]
    # OV read-out of L23H2 from the enclosing header end
    l, h = 23, 2; w = m.dhl[l]; Wo = m.oproj[l].weight[:, h * w:(h + 1) * w].float()
    ov = []
    for qi, p in enumerate(pts):
        e = encl_end(p)
        if not e: ov.append(None); continue
        v = tap.V[l][e[0], h // (m.H // tap.KVH)]
        out = (Wo @ v) * normw.float()  # ignores the final norm's 1/rms (a positive scale, same for all candidates)
        sc = []
        for el in p["paths"]:
            kind, t = el[0]
            sc.append(float(W_U[t].float() @ out) if kind == "t" else float((W_U[m.set_ids(t)].float() @ out).mean()))
        ov.append(sc)
    def ov_at(qi, ti, p):
        # L23H2's direct logit, from source token ti, for the candidate whose indent equals the header's indent at ti
        v = tap.V[l][ti, h // (m.H // tap.KVH)]; out = (Wo @ v) * normw.float()
        sc = [float(W_U[t].float() @ out) if kind == "t" else float((W_U[m.set_ids(t)].float() @ out).mean()) for kind, t in (el[0] for el in p["paths"])]
        return p["K"][int(np.argmax(sc))]  # the candidate indent this source token favours
    stats = {"base": qstats(base_lp, pts)}
    d_t = torch.tensor(dpos)
    for cname, plist in conds.items():
        m.zpatch = {}
        for hname, vals in plist:
            l2, h2 = hn(hname); m.zpatch.setdefault(l2, []).append((h2, d_t, vals))
        stats[cname] = qstats(m.logprobs(ids, positions=[(0, d) for d in dpos]), pts); m.zpatch = {}
    # ablate17: mean-ablate the four layer-17/18 DED heads at decision tokens, read L23H2's attention
    m.zpatch = {}
    for hname in A17:
        l2, h2 = hn(hname); w2 = m.dhl[l2]
        m.zpatch.setdefault(l2, []).append((h2, d_t, zm[l2][h2 * w2:(h2 + 1) * w2].expand(len(dpos), -1)))
    tap.on = True; lp17 = m.logprobs(ids, positions=[(0, d) for d in dpos]); tap.on = False; m.zpatch = {}
    stats["ablate17"] = qstats(lp17, pts); A23b = tap.A[23][2]
    for qi, p in enumerate(pts):
        e = encl_end(p); el = encl_line(p)
        r = dict(cls=p["kcls"], nd=p.get("ndedent"), blank=p["blank"], true=p["true"], cur=p["stack"][-1], stack=p["stack"],
                 K=p["K"], src=str(c.get("repo", c.get("seed", ""))),
                 a23_encl=float(A23[qi, e].sum()) if e else None, a23_encl_line=float(A23[qi, el].sum()) if el else None,
                 a23_encl_abl17=float(A23b[qi, e].sum()) if e else None, ov=ov[qi],
                 **{k: v[qi] for k, v in stats.items()})
        oh = [(ind, ti) for ind, ti in p["OH"] if ti is not None and ti < p["d"]]
        r["open_hdr"] = [(ind, float(A23[qi, ti]), float(ov_at(qi, ti, p))) for ind, ti in oh]
        rec.append(r)
    if time.time() - t0 > 1000: print("time limit"); break
(OUT2 / f"r2e02_{src}_{nch}.json").write_text(json.dumps(rec))
cls = np.array([r["cls"] for r in rec])
def mean(key, j, c): return float(np.mean([r[key][j] for r in rec if r["cls"] == c]))
summ = {}
for c in ["dedent", "term", "same", "header"]:
    if not (cls == c).any(): continue
    b = mean("base", 0, c)
    summ[c] = dict(n=int((cls == c).sum()), base_lq=b, base_psame=mean("base", 1, c),
                   **{k: round(mean(k, 0, c) - b, 4) for k in stats if k != "base" and "_rand_" not in k},
                   rand23=[round(mean(f"ko23_rand_encl_{k}", 0, c) - b, 4) for k in range(8)],
                   psame={k: round(mean(k, 1, c), 4) for k in ["base", "ko23_encl", "koDED_encl_line", "ablate17"]},
                   a23_encl=float(np.mean([r["a23_encl"] for r in rec if r["cls"] == c and r["a23_encl"] is not None] or [np.nan])),
                   a23_encl_abl17=float(np.mean([r["a23_encl_abl17"] for r in rec if r["cls"] == c and r["a23_encl_abl17"] is not None] or [np.nan])))
print(json.dumps(summ, indent=1))
