"""D3 round 2 confirmation r2c01 (PROTOCOL-2.md): the block-end head's reading of open headers, on SEALED natural code
(data/r2code/part-held.jsonl). Refuses to run unless PROTOCOL-2.md exists and is read-only (except --smoke, which
uses 12 chunks of X1's code as a pipeline test).
Per chunk, for every root-identified point (classes as round 1), with H = the named head (layer LH) and OH = the
open headers before the point (r2common.open_headers; innermost last):
  base: log q(true), q(current level), attention of H to each open header's ':\\n' token;
  ko_inner: H loses the innermost open header's ':\\n' (output recomputed from clean attention, renormalised);
  ko_inner for every other head of layer LH (layer-matched controls, all of them);
  ko_closed: H loses the most recent earlier header ':\\n' that is not an open header (control), if any;
  ko_true: H loses the true-level open header's ':\\n' (dedents with the true level among OH);
  steer_j: for dedents (unmarked and terminal) with at least two open headers, H's attention set all-on OH[j]'s
    ':\\n' (output = its value vector there), for each j; the same for a control head of layer LH drawn with
    random.Random(4000 + point counter).
Records go to outputs/r2/conf/r2c01_<model>_<head>.jsonl, one line per chunk, so the job resumes across lock holds.
Usage: python r2c01_header.py <model> <head> [--time-limit S] [--smoke | --calib <exploration source> <n>]"""
import sys, json, time, stat
from r2common import *

MODEL, HEAD = sys.argv[1], sys.argv[2]; smoke = "--smoke" in sys.argv
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
calib = "--calib" in sys.argv
if smoke:
    chunks = r2_sources("x1code")[:12]; od = OUT2 / "conf_smoke"
elif calib:  # exploration material, to set PROTOCOL-2 thresholds before freezing (added 9 Nov)
    csrc, cn = sys.argv[sys.argv.index("--calib") + 1], int(sys.argv[sys.argv.index("--calib") + 2])
    chunks = r2_sources(csrc); random.Random(31).shuffle(chunks); chunks = chunks[:cn]; od = OUT2 / f"conf_calib_{csrc}"
else:
    pf = R2 / "PROTOCOL-2.md"
    if not pf.exists() or (pf.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
        raise SystemExit("refusing to read the sealed part: PROTOCOL-2.md is not frozen")
    chunks = [json.loads(l) for l in open(DATA / "r2code" / "part-held.jsonl")]; od = OUT2 / "conf"
od.mkdir(parents=True, exist_ok=True); outp = od / f"r2c01_{MODEL}_{HEAD}.jsonl"
done = {json.loads(l)["chunk"] for l in open(outp)} if outp.exists() else set()
LH, HH = hn(HEAD); m = M(MODEL, attn="eager"); tap = Tap(m, [LH]); g = m.H // tap.KVH
OTHERS = [h for h in range(m.H) if h != HH]; t0 = time.time(); pc = 0
fh = open(outp, "a")
for ci, c in enumerate(chunks):
    if ci in done: continue
    if time.time() - t0 > tlim: print("time limit; resume later", flush=True); break
    ids, pts = prepare(m, c["text"])
    rec = dict(chunk=ci, repo=c.get("repo"), points=[])
    if pts and len(ids) <= 1400:
        offs = m.tok(c["text"], return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        pts = [p for p in pts if max(p["u"]) == 0]
        hdr_ends = [i for i in range(len(ids)) if m.vs[ids[i]].rstrip(" ").endswith(":\n")]
        for p in pts:
            p["OH"] = [(ind, ti) for ind, ti in open_headers(c["text"], offs, p) if ti is not None and ti < p["d"]]
            ohs = {ti for _, ti in p["OH"]}
            cl = [i for i in hdr_ends if i < p["d"] and i not in ohs]; p["CL"] = cl[-1] if cl else None
    if pts and len(ids) <= 1400:
        T = len(ids); dpos = [p["d"] for p in pts]; d_t = torch.tensor(dpos); rm = RootMasses(m, pts)
        tap.rows = dpos; tap.on = True; q0 = rm(m.logprobs(ids, positions=[(0, d) for d in dpos])); tap.on = False
        A = tap.A[LH]

        def run(entries):
            m.zpatch = {LH: entries}; q = rm(m.logprobs(ids, positions=[(0, d) for d in dpos])); m.zpatch = {}; return q

        def ko(h, get):
            vals = []
            for qi, p in enumerate(pts):
                keep = torch.ones(T, device=DEV); pos = get(p)
                if pos: keep[torch.tensor(pos, device=DEV)] = 0
                vals.append(tap.head_out(LH, h, qi, keep))
            return run([(h, d_t, torch.stack(vals))])
        inner = lambda p: [p["OH"][-1][1]] if p["OH"] else []
        Q = {"base": q0, "ko_inner": ko(HH, inner), "ko_closed": ko(HH, lambda p: [p["CL"]] if p["CL"] is not None else []),
             "ko_true": ko(HH, lambda p: [ti for ind, ti in p["OH"] if ind == p["true"]])}
        for h in OTHERS: Q[f"ko_inner_h{h}"] = ko(h, inner)
        steerable = [qi for qi, p in enumerate(pts) if p["cls"] == "shallower" and len(p["OH"]) >= 2]
        ctrl = {}
        for qi in steerable:
            ctrl[qi] = random.Random(4000 + pc).choice(OTHERS); pc += 1
        steer = {}
        maxj = max([len(pts[qi]["OH"]) for qi in steerable], default=0)
        for j in range(maxj):
            for who in ("head", "ctrl"):
                ent = []
                for qi in steerable:
                    if j >= len(pts[qi]["OH"]): continue
                    h = HH if who == "head" else ctrl[qi]
                    ent.append((h, torch.tensor([dpos[qi]]), tap.V[LH][pts[qi]["OH"][j][1], h // g][None]))
                if not ent: continue
                qs = run(ent)
                for qi in steerable:
                    if j < len(pts[qi]["OH"]): steer[(qi, j, who)] = qs[qi]
        for qi, p in enumerate(pts):
            K = p["K"]; cur = p["stack"][-1]; ix = K.index
            def mz(q): return [float(np.log(q[ix(p["true"])])), float(q[ix(cur)]) if cur in K else 0.0,
                               float(sum(q[i] for i, k in enumerate(K) if k in p["stack"][:-1]))]
            r = dict(cls=point_class(p), true=p["true"], cur=cur, oh=[ind for ind, _ in p["OH"]],
                     att=[float(A[HH, qi, ti]) for _, ti in p["OH"]], has_closed=p["CL"] is not None,
                     **{k: mz(v[qi]) for k, v in Q.items()})
            if qi in steerable:
                r["ctrl"] = ctrl[qi]; r["top_base"] = K[int(np.argmax(q0[qi]))]; r["steer"] = []
                for j, (ind, _) in enumerate(p["OH"]):
                    for who in ("head", "ctrl"):
                        qs = steer[(qi, j, who)]
                        r["steer"].append([who, j, ind, float(np.log(qs[ix(ind)]) - np.log(q0[qi][ix(ind)])), K[int(np.argmax(qs))]])
            rec["points"].append(r)
    fh.write(json.dumps(rec) + "\n"); fh.flush()
fh.close()
n_done = sum(1 for _ in open(outp))
print(json.dumps(dict(model=MODEL, head=HEAD, chunks_done=n_done, of=len(chunks), secs=round(time.time() - t0))))
