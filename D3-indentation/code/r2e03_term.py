"""D3 round 2, r2e03 (exploratory, question 3): how the terminal keyword reaches the decision newline.
gen2 exploration programs. Target point = the dedent at the target line; its previous line is the innermost
block's last line (terminal keyword, or a plain statement in the unmarked programs). Tokens of that line:
kw (the token holding its first word), after (tokens after kw, before the newline), line (all its tokens), d (the
decision token, the newline).
Conditions:
 L0H2 mean-ablated (mean over all positions of the first 40 programs) at: all positions / line+d / kw / after / d /
   every position except line and d;
 attention edges removed in all layers (exact, EdgeBlock): d->kw, d->after, after->kw, d->kw plus after->kw,
   d->line; and d->line only in layers < L or only in layers >= L, L in 4, 8, ..., 24;
 neutral: the terminal line replaced by "value = value" (terminal programs only), for the gap.
Measures at the target point: s = log q(true) - log q(current level), log q(true), q(current level).
Usage: python r2e03_term.py <n_programs> [kind: terminal|plain]"""
import sys, json, time
from r2common import *

n = int(sys.argv[1]); kind = sys.argv[2] if len(sys.argv) > 2 else "terminal"
m = M("qwen3-06b-bf16", attn="eager"); eb = EdgeBlock(m)
progs = [g for g in r2_sources("gen2") if g["last"] == kind]; random.Random(23).shuffle(progs)


def target_point(text, tl):
    ids, pts = prepare(m, text)
    if not pts: return None, None
    pp = [p for p in pts if p["line"] == tl and max(p["u"]) == 0]
    return (ids, pp[0]) if pp else (ids, None)


def meas(ids, p):
    lp = m.logprobs(ids, positions=[(0, p["d"])])
    q = RootMasses(m, [p])(lp)[0]; K = p["K"]; cur = p["stack"][-1]
    lt, lc = np.log(q[K.index(p["true"])]), np.log(q[K.index(cur)])
    return [float(lt - lc), float(lt), float(np.exp(lc))]


# L0H2 mean over all positions
acc = 0; cnt = 0
for g in progs[:40]:
    ids = m.tok(g["text"], add_special_tokens=False)["input_ids"]
    m.capture = True; m.logprobs(ids, positions=[(0, 0)]); m.capture = False
    acc = acc + m.zcap[0].float().sum(0); cnt += len(ids)
w0 = m.dhl[0]; mu = (acc / cnt)[2 * w0:3 * w0]
rec = []; t0 = time.time()
for g in progs:
    if len(rec) >= n or time.time() - t0 > 1000: break
    text, tl = g["text"], g["target_line"]
    ids, p = target_point(text, tl)
    if p is None: continue
    offs = m.tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    lines = text.split("\n"); ls = sum(len(x) + 1 for x in lines[:tl - 2]); ind = len(lines[tl - 2]) - len(lines[tl - 2].lstrip(" "))
    d = p["d"]; kwc = ls + ind
    kw = next(i for i, (a, b) in enumerate(offs) if a <= kwc < b)
    line = [i for i in range(d) if offs[i][1] > ls]; after = list(range(kw + 1, d))
    r = dict(seed=g["seed"], kw=g["term_kw"], D=g["D"], width=g["width"], true=p["true"], cur=p["stack"][-1],
             n_after=len(after), kw_tok=m.vs[ids[kw]], d_tok=m.vs[ids[d]])
    r["base"] = meas(ids, p)
    T = len(ids)
    for cname, pos in [("l0h2_all", list(range(T))), ("l0h2_line", line + [d]), ("l0h2_kw", [kw]), ("l0h2_after", after),
                       ("l0h2_d", [d]), ("l0h2_notline", [i for i in range(T) if i not in set(line + [d])])]:
        if not pos: r[cname] = None; continue
        pt = torch.tensor(pos); m.zpatch = {0: [(2, pt, mu.expand(len(pos), -1))]}
        r[cname] = meas(ids, p); m.zpatch = {}
    AL = set(range(m.L))
    for cname, edges in [("blk_d_kw", [(AL, [d], [kw])]), ("blk_d_after", [(AL, [d], after)] if after else None),
                         ("blk_after_kw", [(AL, after, [kw])] if after else None),
                         ("blk_d_kw_after_kw", [(AL, [d], [kw])] + ([(AL, after, [kw])] if after else [])),
                         ("blk_d_line", [(AL, [d], line)])] + \
                        [(f"blk_d_line_lt{L}", [(set(range(L)), [d], line)]) for L in (4, 8, 12, 16, 20, 24)] + \
                        [(f"blk_d_line_ge{L}", [(set(range(L, m.L)), [d], line)]) for L in (4, 8, 12, 16, 20, 24)]:
        if edges is None: r[cname] = None; continue
        eb.edges = edges; r[cname] = meas(ids, p); eb.edges = []
    if g.get("text_neutral"):
        ids2, p2 = target_point(g["text_neutral"], tl)
        r["neutral"] = meas(ids2, p2) if p2 is not None and p2["K"] == p["K"] else None
    rec.append(r)
(OUT2 / f"r2e03_{kind}_{n}.json").write_text(json.dumps(rec))
keys = [k for k in rec[0] if isinstance(rec[0][k], list)]
print(len(rec), "programs", kind)
for grp in ["all", "return", "raise", "pass", "break", "continue"]:
    rs = [r for r in rec if grp == "all" or r["kw"] == grp]
    if not rs: continue
    b = np.mean([r["base"][0] for r in rs]); neu = [r["neutral"][0] for r in rs if r.get("neutral")]
    gap = b - np.mean(neu) if neu else float("nan")
    print(f"== {grp} n={len(rs)} base s={b:.3f} neutral s={np.mean(neu) if neu else float('nan'):.3f} gap={gap:.3f}")
    for k in keys:
        if k in ("base", "neutral"): continue
        v = [r[k][0] for r in rs if r.get(k)]
        if v: print(f"   {k:22s} s={np.mean(v):7.3f}  share of gap removed={(b - np.mean(v)) / gap if gap == gap else float('nan'):6.3f}  lq={np.mean([r[k][1] for r in rs if r.get(k)]):7.3f}")
