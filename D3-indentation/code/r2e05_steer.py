"""D3 round 2, r2e05 (exploratory, question 2): does the open header that L23H2 attends to choose the dedent level?
Dedent points (unmarked and terminal, root-identified) with at least two open headers before them.
For each open header x (indent ind_x, its ':\\n' token t_x):
  steer: L23H2's attention at the decision token is replaced by all-on-t_x (its output = its value vector at t_x);
  the same for a control head (a random other layer-23 head, fixed per point) steered to t_x.
Also: ko_true (L23H2 loses the true-level header's ':\\n'), ko_allhdr (loses every open header's ':\\n').
Measures: log q(ind_x) change when steering to x; how often ind_x becomes the top candidate; log q(true).
Usage: python r2e05_steer.py <source> <n_chunks> [model head]   (default qwen3-06b-bf16 L23H2)"""
import sys, json, time
from r2common import *

src, nch = sys.argv[1], int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3-06b-bf16"; HEAD = sys.argv[4] if len(sys.argv) > 4 else "L23H2"
L, Hh = hn(HEAD)
m = M(MODEL, attn="eager"); tap = Tap(m, [L])
chunks = r2_sources(src); random.Random(25).shuffle(chunks)
g = m.H // tap.KVH
rec = []; t0 = time.time(); used = 0
for c in chunks:
    if used >= nch or time.time() - t0 > 1000: break
    ids, pts = prepare(m, c["text"])
    if not pts or len(ids) > 900: continue
    offs = m.tok(c["text"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    pts = [p for p in pts if max(p["u"]) == 0 and p["cls"] == "shallower"]
    sel = []
    for p in pts:
        oh = [(ind, ti) for ind, ti in open_headers(c["text"], offs, p) if ti is not None and ti < p["d"]]
        if len(oh) >= 2 and all(ind in p["K"] for ind, _ in oh): p["OH"] = oh; sel.append(p)
    if not sel: continue
    used += 1; d_t = torch.tensor([p["d"] for p in sel]); T = len(ids)
    tap.rows = [p["d"] for p in sel]; tap.on = True
    lp0 = m.logprobs(ids, positions=[(0, p["d"]) for p in sel]); tap.on = False
    rm = RootMasses(m, sel); q0 = rm(lp0)
    rng = random.Random(900 + used); ctrl = [rng.choice([h for h in range(m.H) if h != Hh]) for _ in sel]

    def run(vals_by_head):
        m.zpatch = {L: [(h, d_t, v) for h, v in vals_by_head]}
        out = rm(m.logprobs(ids, positions=[(0, p["d"]) for p in sel])); m.zpatch = {}; return out
    # knock-outs for L23H2 (all points at once)
    def ko(get):
        vals = []
        for qi, p in enumerate(sel):
            keep = torch.ones(T, device=DEV); pos = get(p)
            if pos: keep[torch.tensor(pos, device=DEV)] = 0
            vals.append(tap.head_out(L, Hh, qi, keep))
        return run([(Hh, torch.stack(vals))])
    q_kotrue = ko(lambda p: [ti for ind, ti in p["OH"] if ind == p["true"]])
    q_koall = ko(lambda p: [ti for ind, ti in p["OH"]])
    # steering: one pass per header rank j (points lacking rank j keep their clean value)
    maxj = max(len(p["OH"]) for p in sel)
    steer = {qi: {} for qi in range(len(sel))}
    for j in range(maxj):
        for who in ("L23H2", "ctrl"):
            vals = []; cvals = []
            for qi, p in enumerate(sel):
                h = Hh if who == "L23H2" else ctrl[qi]
                if j < len(p["OH"]): v = tap.V[L][p["OH"][j][1], h // g]
                else: v = tap.head_out(L, h, qi)
                vals.append((h, v))
            # group by head index (ctrl heads differ per point): patch per point via row-specific positions
            m.zpatch = {L: [(h, torch.tensor([sel[qi]["d"]]), v[None]) for qi, (h, v) in enumerate(vals)]}
            qs = rm(m.logprobs(ids, positions=[(0, p["d"]) for p in sel])); m.zpatch = {}
            for qi, p in enumerate(sel):
                if j < len(p["OH"]): steer[qi][(who, j)] = qs[qi]
    for qi, p in enumerate(sel):
        K = p["K"]; ix = lambda k: K.index(k)
        r = dict(cls="term" if p["terminal"] else "dedent", true=p["true"], cur=p["stack"][-1], oh=[ind for ind, _ in p["OH"]],
                 lq_base=float(np.log(q0[qi][ix(p["true"])])), lq_kotrue=float(np.log(q_kotrue[qi][ix(p["true"])])),
                 lq_koall=float(np.log(q_koall[qi][ix(p["true"])])), top_base=K[int(np.argmax(q0[qi]))],
                 q_base={ind: float(q0[qi][ix(ind)]) for ind, _ in p["OH"]}, steer=[])
        for j, (ind, ti) in enumerate(p["OH"]):
            for who in ("L23H2", "ctrl"):
                qs = steer[qi][(who, j)]
                r["steer"].append(dict(who=who, j=j, ind=ind, dlq=float(np.log(qs[ix(ind)]) - np.log(q0[qi][ix(ind)])),
                                       top=K[int(np.argmax(qs))], lq_true=float(np.log(qs[ix(p["true"])]))))
        rec.append(r)
(OUT2 / (f"r2e05_{src}_{nch}.json" if MODEL == "qwen3-06b-bf16" and HEAD == "L23H2" else f"r2e05_{MODEL}_{HEAD}_{src}_{nch}.json")).write_text(json.dumps(rec))
print(src, "points", len(rec))
for cl in ["dedent", "term"]:
    rs = [r for r in rec if r["cls"] == cl]
    if not rs: continue
    print(f"== {cl} n={len(rs)} lq base {np.mean([r['lq_base'] for r in rs]):.3f} ko_true {np.mean([r['lq_kotrue'] for r in rs]):.3f} ko_allhdr {np.mean([r['lq_koall'] for r in rs]):.3f}")
    for who in ("L23H2", "ctrl"):  # "L23H2" labels the steered head
        st = [s for r in rs for s in r["steer"] if s["who"] == who]
        nb = [s for r in rs for s in r["steer"] if s["who"] == who and s["ind"] != r["top_base"]]
        print(f"   steer {who}: mean dlq(x) {np.mean([s['dlq'] for s in st]):.3f}; x becomes top {np.mean([s['top'] == s['ind'] for s in st]):.3f}; "
              f"for x not already top: dlq {np.mean([s['dlq'] for s in nb]):.3f}, becomes top {np.mean([s['top'] == s['ind'] for s in nb]):.3f} (n={len(nb)})")
