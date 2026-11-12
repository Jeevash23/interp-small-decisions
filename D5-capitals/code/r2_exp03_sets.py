"""D5 round 2, exploration 3 (exploratory; CHARTER-2.md Q1, Q2): named component sets against matched
random sets.
Usage: r2_exp03_sets.py <model> <part> <first> <n> <sets.json> [per_para=2] [seed=13] [nrand=20] [tag]
sets.json: {"writers": [...], "carriers": [...], "readers": [...]}, components as "L19H0" or "MLP21".
At up to `per_para` word starts per paragraph (word 12 or later), prefixes U, L, A as in r2_exp01:
  W (writers, U runs): directional ablation - each member's write along the unit capitals direction u
    (d5common2.Writes terms, before later layer scalars) is moved to its value in the paired L run;
    the rest of each output is unchanged.
  C (carriers, A runs and U runs): each member head's output (o_proj input slice) at the prediction
    position replaced by its L-run value.
  R (readers, L runs + d at l): each member's output frozen at its unedited L-run value while d is added
    to the residual after layer l.
Each condition: row 0 = no set, row 1 = the named set, rows 2.. = random sets matched member by member
(see matched(); seed 'r2-sets'). Records per row: top-1, D = logp(U) - logp(Lc), and the
residual's projection on u after layer l. Writes outputs/r2/exp03_<model>_<part>_<first>_<tag>.pt"""
import json as _json, re, sys, time
from d5common2 import *  # noqa: F401,F403
from d5common2 import LastBatch, Writes

short, part = sys.argv[1], sys.argv[2]; FIRST, N = int(sys.argv[3]), int(sys.argv[4])
SETS = _json.load(open(sys.argv[5])); PER = int(sys.argv[6]) if len(sys.argv) > 6 else 2
SEED = int(sys.argv[7]) if len(sys.argv) > 7 else 13; NR = int(sys.argv[8]) if len(sys.argv) > 8 else 20
TAG = sys.argv[9] if len(sys.argv) > 9 else "sets"
LAYER = {"gemma": 24, "qwen3-17b": 22}[short]
rows = load_part2(part, seed=0)[FIRST:FIRST + N]
M_ = Model(short); m = M_.m; vs = M_.vs; tok = M_.tok; L, H = m.L, m.H; LB = LastBatch(M_); W = Writes(M_)
md = torch.load(OUT / f"exp01_{short}_means.pt")
d = (md["means"]["upper"] - md["means"]["lower"]).float()[LAYER + 1].to(DEV); u = d / d.norm()


def parse(s):
    a = re.fullmatch(r"L(\d+)H(\d+)", s)
    return ("h", int(a.group(1)), int(a.group(2))) if a else ("m", int(re.fullmatch(r"MLP(\d+)", s).group(1)))


def matched(named, rng, kind):
    """Random set matched member by member: a head by another head of the same layer; an MLP by another
    MLP not in the named set (any layer before the last), or, for readers, by a random head of the same
    layer (all MLPs after the direction layer may be named, and earlier ones cannot react to d)."""
    nm = set(named); out = []
    for c in named:
        if c[0] == "h" or kind == "readers":
            pool = [("h", c[1], h) for h in range(H) if ("h", c[1], h) not in nm and ("h", c[1], h) not in out]
        else:
            pool = [("m", l) for l in range(L - 1) if ("m", l) not in nm and ("m", l) not in out]
        out.append(rng.choice(pool))
    return out


rng = random.Random("r2-sets")
NAMED = {k: [parse(s) for s in v] for k, v in SETS.items()}
RAND = {k: [matched(v, rng, k) for _ in range(NR)] for k, v in NAMED.items()}
GROUPS = {k: [[]] + [NAMED[k]] + RAND[k] for k in NAMED}

# hooks: full-output patches for MLPs, additions along u at the residual-add points
mpatch, dadd = {}, {}
def mk_mlp(l):
    def f(mod, a, o):
        lst = mpatch.get(l)
        if not lst: return None
        o = o.clone()
        for row, val in lst: o[row, -1] = val
        return o
    return f
def mk_add(key):
    def f(mod, a, o):
        lst = dadd.get(key)
        if not lst: return None
        o = o.clone()
        for row, val in lst: o[row, -1] += val.to(o.dtype)
        return o
    return f
for l, lay in enumerate(m.layers):
    lay.mlp.register_forward_hook(mk_mlp(l))
    if W.gemma:
        lay.post_attention_layernorm.register_forward_hook(mk_add(("a", l)))
        lay.post_feedforward_layernorm.register_forward_hook(mk_add(("m", l)))
    else:
        lay.self_attn.o_proj.register_forward_hook(mk_add(("a", l)))
        lay.mlp.register_forward_hook(mk_add(("m", l)))
lout = {}
m.layers[LAYER].register_forward_hook(lambda mod, a, o: lout.update(x=(o[0] if isinstance(o, tuple) else o)[:, -1].float()))


def build(text, rs, i, form):
    pre = text[:rs[i][0] - 1]
    if form == "U": return pre.upper()
    if form == "L": return pre.lower()
    c0 = rs[i - 1][0]
    return pre[:c0].upper() + pre[c0:].lower()


def full(ids):
    W.t = {}; W.on = True; m.capture = True
    with torch.no_grad(): lg = m.model(torch.tensor([ids], device=DEV), logits_to_keep=1).logits[0, -1].float()
    W.on = False; m.capture = False
    T = W.terms()
    return {"lg": lg, "proj": {k: float(v @ u) for k, v in T.items()}, "z": {l: m.cap[l][-1].clone() for l in range(L)},
            "mlp": {l: W.t[("mlp", l)].clone() for l in range(L)} if not W.gemma else None, "h": m.rcap[LAYER][-1].clone()}


mlpraw = {}
for l, lay in enumerate(m.layers):
    lay.mlp.register_forward_hook((lambda l: (lambda mod, a, o: mlpraw.__setitem__(l, o[0, -1].detach().clone()) if W.on else None))(l))


def run(ids, groups, setup):
    res = []
    for s in range(0, len(groups), 22):
        gs = groups[s:s + 22]; m.patch = {}; mpatch.clear(); dadd.clear(); m.rpatch = {}
        for j, g in enumerate(gs): setup(j, g)
        lg = LB(ids, len(gs)); m.patch = {}; mpatch.clear(); dadd.clear(); m.rpatch = {}
        lp = torch.log_softmax(lg, -1)
        for j in range(len(gs)):
            res.append((vs[int(lg[j].argmax())], float(lp[j, Ut] - lp[j, Lt]), float(lout["x"][j] @ u)))
    return res


rng2 = random.Random(SEED); t0 = time.time(); recs = []
for pi, r in enumerate(rows):
    text = cut(r["text"]); rs = runs(text)
    cand = [i for i, (s, e) in enumerate(rs) if i >= 12 and text[s - 1] == " "]
    for i in sorted(rng2.sample(cand, min(PER, len(cand)))):
        word = text[rs[i][0]:rs[i][1]]
        Ut = tok(" " + word.upper(), add_special_tokens=False)["input_ids"][0]; Lt = tok(" " + word.lower(), add_special_tokens=False)["input_ids"][0]
        ids = {f: tok(build(text, rs, i, f), add_special_tokens=False)["input_ids"] for f in ("U", "L", "A")}
        F = {}
        for f in ("U", "L", "A"):
            mlpraw.clear(); F[f] = full(ids[f]); F[f]["mlpraw"] = dict(mlpraw)
        rec = {"para": FIRST + pi, "pageid": r["pageid"], "run": i, "word": word}
        LB.clear()
        for f in ("U", "L", "A"): LB.prime(ids[f])
        for f in ("U", "L", "A"):
            lp = torch.log_softmax(F[f]["lg"], -1)
            rec[f] = {"top": vs[int(F[f]["lg"].argmax())], "D": float(lp[Ut] - lp[Lt]), "proj": float(F[f]["h"].float() @ u)}

        def wsetup(j, g):
            for c in g:
                if c[0] == "h":
                    key, dl = ("a", c[1]), F["L"]["proj"][("attn", c[1], c[2])] - F["U"]["proj"][("attn", c[1], c[2])]
                else:
                    key, dl = ("m", c[1]), F["L"]["proj"][("mlp", c[1])] - F["U"]["proj"][("mlp", c[1])]
                dadd.setdefault(key, []).append((j, dl * u))
        def csetup(j, g):
            for c in g:
                if c[0] == "h": m.patch.setdefault(c[1], []).append((c[2], -1, F["L"]["z"][c[1]][m.hs(c[1], c[2])], j))
                else: mpatch.setdefault(c[1], []).append((j, F["L"]["mlpraw"][c[1]]))
        def rsetup(j, g):
            m.rpatch.setdefault(LAYER, []).append((-1, F["L"]["h"] + d.to(F["L"]["h"].dtype), j)); csetup(j, g)
        if "writers" in GROUPS: rec["W"] = run(ids["U"], GROUPS["writers"], wsetup)
        if "carriers" in GROUPS:
            rec["CA"] = run(ids["A"], GROUPS["carriers"], csetup); rec["CU"] = run(ids["U"], GROUPS["carriers"], csetup)
        if "readers" in GROUPS: rec["R"] = run(ids["L"], GROUPS["readers"], rsetup)
        recs.append(rec)
    if pi % 10 == 0: print(pi, round(time.time() - t0), flush=True)

nm = lambda c: f"L{c[1]}H{c[2]}" if c[0] == "h" else f"MLP{c[1]}"
torch.save({"model": short, "part": part, "rows": [FIRST, FIRST + N], "layer": LAYER, "sets": SETS,
            "random": {k: [[nm(c) for c in g] for g in v] for k, v in RAND.items()}, "records": recs,
            "seconds": round(time.time() - t0)}, OUT / "r2" / f"exp03_{short}_{part}_{FIRST}_{TAG}.pt")


def same_word(a, b):
    a, b = a.strip().lower(), b.strip().lower()
    return bool(a) and bool(b) and a.isalpha() and b.isalpha() and (a.startswith(b) or b.startswith(a))


def report(cond, run_key, base_key, ref_key, want):
    el = [rc for rc in recs if case_of(rc[base_key]["top"]) == ("upper" if want == "lower" else "lower")] if want else recs
    n = len(GROUPS[cond]); out = []
    for j in range(n):
        up = np.mean([case_of(rc[run_key][j][0]) == "upper" for rc in el])
        sw = np.mean([same_word(rc[base_key]["top"], rc[run_key][j][0]) for rc in el])
        exD = np.mean([rc[run_key][0][1] - rc[ref_key]["D"] for rc in recs]); exP = np.mean([rc[run_key][0][2] - rc[ref_key]["proj"] for rc in recs])
        fD = np.mean([rc[run_key][0][1] - rc[run_key][j][1] for rc in recs]) / exD
        fP = np.mean([rc[run_key][0][2] - rc[run_key][j][2] for rc in recs]) / exP
        out.append((round(float(up), 3), round(float(sw), 3), round(float(fD), 3), round(float(fP), 3)))
    print(f"{cond}/{run_key} (n {len(el)}): none {out[0]}, named {out[1]}; random upper share {min(o[0] for o in out[2:])}-{max(o[0] for o in out[2:])},"
          f" D removed {min(o[2] for o in out[2:])}-{max(o[2] for o in out[2:])}, proj removed {min(o[3] for o in out[2:])}-{max(o[3] for o in out[2:])}")


print("(upper share, same word any case, fraction of D excess removed, fraction of proj excess removed)")
if "writers" in GROUPS: report("writers", "W", "U", "L", "lower")
if "carriers" in GROUPS: report("carriers", "CA", "A", "L", None); report("carriers", "CU", "U", "L", "lower")
if "readers" in GROUPS: report("readers", "R", "L", "L", "upper")
