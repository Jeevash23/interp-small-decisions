"""D3 round 2 confirmation r2c02 (PROTOCOL-2.md): the terminal route and level steering on SEALED targeted programs
(data/r2gen2/part-held.jsonl). Refuses to run unless PROTOCOL-2.md exists and is read-only (except --smoke: 16 gen2
exploration programs). Target point = the dedent at the target line (r2e03 roles: kw, after, line, d).
Per program (terminal and plain):
  base; L0H2 mean-ablated (mean over all positions of the first 40 programs of the part, seed-free order) on the
  terminal line and d / everywhere; edges removed in all layers (EdgeBlock): d->kw, d->kw plus after->kw;
  d->kw in each single layer 0..L-1; neutral twin (terminal programs only).
  Steering at the target point: the named head's attention set all-on each open header's ':\\n'; the same for a
  control head of its layer drawn with random.Random(5000 + seed).
Measures: q(current level), log q(true); steering: change in log q(header's indent) and the new top candidate.
Records: outputs/r2/conf/r2c02_<model>_<head>.jsonl, one line per program (resumable).
Usage: python r2c02_gen2.py <model> <head> [--time-limit S] [--smoke | --calib <n>]"""
import sys, json, time, stat
from r2common import *

MODEL, HEAD = sys.argv[1], sys.argv[2]; smoke = "--smoke" in sys.argv
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
calib = "--calib" in sys.argv
if smoke:
    progs = r2_sources("gen2")[:16]; od = OUT2 / "conf_smoke"
elif calib:  # gen2 exploration programs, to set PROTOCOL-2 thresholds before freezing (added 9 Nov)
    progs = r2_sources("gen2")[:int(sys.argv[sys.argv.index("--calib") + 1])]; od = OUT2 / "conf_calib_gen2"
else:
    pf = R2 / "PROTOCOL-2.md"
    if not pf.exists() or (pf.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)):
        raise SystemExit("refusing to read the sealed part: PROTOCOL-2.md is not frozen")
    progs = [json.loads(l) for l in open(DATA / "r2gen2" / "part-held.jsonl")]; od = OUT2 / "conf"
od.mkdir(parents=True, exist_ok=True); outp = od / f"r2c02_{MODEL}_{HEAD}.jsonl"
done = {json.loads(l)["seed"] for l in open(outp)} if outp.exists() else set()
LH, HH = hn(HEAD); m = M(MODEL, attn="eager"); eb = EdgeBlock(m); tap = Tap(m, [LH]); g = m.H // tap.KVH
L0, H0 = (0, 2) if m.H > 2 else (0, 0)
acc = 0; cnt = 0
for pr in progs[:40]:
    ids = m.tok(pr["text"], add_special_tokens=False)["input_ids"]
    m.capture = True; m.logprobs(ids, positions=[(0, 0)]); m.capture = False
    acc = acc + m.zcap[L0].float().sum(0); cnt += len(ids)
w0 = m.dhl[L0]; mu = (acc / cnt)[H0 * w0:(H0 + 1) * w0]


def setup(text, tl):
    ids, pts = prepare(m, text)
    if not pts: return None
    pp = [p for p in pts if p["line"] == tl and max(p["u"]) == 0]
    if not pp: return None
    p = pp[0]; offs = m.tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    lines = text.split("\n"); ls = sum(len(x) + 1 for x in lines[:tl - 2]); ind = len(lines[tl - 2]) - len(lines[tl - 2].lstrip(" "))
    kw = next(i for i, (a, b) in enumerate(offs) if a <= ls + ind < b)
    line = [i for i in range(p["d"]) if offs[i][1] > ls]
    p["OH"] = [(i2, ti) for i2, ti in open_headers(text, offs, p) if ti is not None and ti < p["d"]]
    return ids, p, kw, line


def meas(ids, p):
    q = RootMasses(m, [p])(m.logprobs(ids, positions=[(0, p["d"])]))[0]; K = p["K"]; cur = p["stack"][-1]
    return q, [float(q[K.index(cur)]), float(np.log(q[K.index(p["true"])]))]


t0 = time.time(); fh = open(outp, "a")
for pr in progs:
    if pr["seed"] in done: continue
    if time.time() - t0 > tlim: print("time limit; resume later", flush=True); break
    s = setup(pr["text"], pr["target_line"]); rec = dict(seed=pr["seed"], last=pr["last"], kw=pr["term_kw"], D=pr["D"], width=pr["width"])
    if s is None:
        rec["skip"] = True; fh.write(json.dumps(rec) + "\n"); continue
    ids, p, kw, line = s; d = p["d"]; after = list(range(kw + 1, d)); T = len(ids)
    tap.rows = [d]; tap.on = True; q0, rec["base"] = meas(ids, p); tap.on = False
    rec.update(true=p["true"], cur=p["stack"][-1], n_after=len(after), oh=[ind for ind, _ in p["OH"]])
    for cname, pos in [("l0h2_line", line + [d]), ("l0h2_all", list(range(T)))]:
        m.zpatch = {L0: [(H0, torch.tensor(pos), mu.expand(len(pos), -1))]}; rec[cname] = meas(ids, p)[1]; m.zpatch = {}
    AL = set(range(m.L))
    for cname, edges in [("blk_kw", [(AL, [d], [kw])]), ("blk_kw_after", [(AL, [d], [kw])] + ([(AL, after, [kw])] if after else []))] + \
            [(f"lay{l}", [({l}, [d], [kw])]) for l in range(m.L)]:
        eb.edges = edges; rec[cname] = meas(ids, p)[1]; eb.edges = []
    if pr.get("text_neutral"):
        s2 = setup(pr["text_neutral"], pr["target_line"])
        rec["neutral"] = meas(s2[0], s2[1])[1] if s2 is not None and s2[1]["K"] == p["K"] else None
    if len(p["OH"]) >= 2:
        ch = random.Random(5000 + pr["seed"]).choice([h for h in range(m.H) if h != HH]); rec["ctrl"] = ch; rec["steer"] = []
        K = p["K"]; rec["top_base"] = K[int(np.argmax(q0))]
        for j, (ind, ti) in enumerate(p["OH"]):
            for who, h in (("head", HH), ("ctrl", ch)):
                m.zpatch = {LH: [(h, torch.tensor([d]), tap.V[LH][ti, h // g][None])]}
                qs = meas(ids, p)[0]; m.zpatch = {}
                rec["steer"].append([who, j, ind, float(np.log(qs[K.index(ind)]) - np.log(q0[K.index(ind)])), K[int(np.argmax(qs))]])
    fh.write(json.dumps(rec) + "\n"); fh.flush()
fh.close()
print(json.dumps(dict(model=MODEL, head=HEAD, programs_done=sum(1 for _ in open(outp)), of=len(progs), secs=round(time.time() - t0))))
