"""D3 e04 (exploratory): minimal pairs at terminal-statement dedents in generated exploration programs.
Clean: the program as written, where line L-1 is a terminal statement (return/pass/raise/break/continue) and
line L dedents. Corrupt: line L-1 replaced by a neutral assignment at the same indent ("<name> = <name>"),
everything else unchanged. Signal at the decision position: dedent score = log q(true level) - log q(current
level) over the candidate indents.
For every head and MLP: (a) sufficiency, patch its clean output at the decision position into the corrupt
run, fraction of the clean-corrupt gap restored; (b) necessity, patch its corrupt output into the clean run,
fraction of the gap removed. Also the residual stream at the decision position per layer, and the residual at
the previous line's tokens per layer (where the terminal information is first carried).
Usage: python e04_pairs.py <model> <n_pairs> [--time-limit S]"""
import sys, json, time, re
from common import *

name, npairs = sys.argv[1], int(sys.argv[2])
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
m = M(name, attn="sdpa"); t0 = time.time()
progs = load_chunks("gen", "explore"); random.Random(4).shuffle(progs)
pairs = []
for pr in progs:
    text = pr["text"]; lines = text.split("\n")
    for p in P.decision_points(text):
        if not (p["terminal"] and p["cls"] == "shallower" and p["blank"] == 0): continue
        L = p["line"]; prev = lines[L - 2]; ind = len(prev) - len(prev.lstrip(" "))
        if prev.strip().split()[0] not in P.TERMINAL: continue
        neutral = " " * ind + "value = value"
        ctext = "\n".join(lines[:L - 2] + [neutral] + lines[L - 1:])
        a_ids, a_pts = prepare(m, text); b_ids, b_pts = prepare(m, ctext)
        if not a_pts or not b_pts: continue
        pa = [q for q in a_pts if q["line"] == L]; pb = [q for q in b_pts if q["line"] == L]
        if not pa or not pb or max(pa[0]["u"]) or max(pb[0]["u"]) or pa[0]["K"] != pb[0]["K"]: continue
        # tokens of the edited line in each version (for previous-line residual patching)
        pairs.append(dict(a=a_ids, b=b_ids, pa=pa[0], pb=pb[0]))
        break
    if len(pairs) >= npairs: break
print("pairs", len(pairs), flush=True)


def score(lp, p):
    if "_rm" not in p: p["_rm"] = RootMasses(m, [p])
    q = p["_rm"](lp[None])[0]
    K = p["K"]; return float(np.log(q[K.index(p["true"])]) - np.log(q[K.index(p["stack"][-1])]))


def run(ids, p, cap=False):
    m.capture = cap; lp = m.logprobs(ids, positions=[(0, p["d"])])[0]; m.capture = False
    return score(lp, p)


clean, corr, caps = [], [], []
for pr in pairs:
    sa = run(pr["a"], pr["pa"], True)
    ca = ({l: m.zcap[l][pr["pa"]["d"]].clone() for l in range(m.L)}, {l: m.mcap[l][pr["pa"]["d"]].clone() for l in range(m.L)},
          {l: m.rcap[l][pr["pa"]["d"]].clone() for l in range(m.L)})
    sb = run(pr["b"], pr["pb"], True)
    cb = ({l: m.zcap[l][pr["pb"]["d"]].clone() for l in range(m.L)}, {l: m.mcap[l][pr["pb"]["d"]].clone() for l in range(m.L)},
          {l: m.rcap[l][pr["pb"]["d"]].clone() for l in range(m.L)})
    clean.append(sa); corr.append(sb); caps.append((ca, cb))
clean, corr = np.array(clean), np.array(corr); gap = clean - corr
outp = OUT / f"e04_{name}_{npairs}.json"
res = dict(n=len(pairs), clean=float(clean.mean()), corrupt=float(corr.mean()), gap=float(gap.mean()), comps={})
if outp.exists(): res["comps"] = json.loads(outp.read_text())["comps"]
print(res, flush=True)
comps = [("resid", l, -1) for l in range(m.L)] + [("mlp", l, -1) for l in range(m.L)] + [("head", l, h) for l in range(m.L) for h in range(m.H)]
for kind, l, h in comps:
    if time.time() - t0 > tlim: print("time limit"); break
    key = f"resid{l}" if kind == "resid" else f"mlp{l}" if kind == "mlp" else f"L{l}H{h}"
    if key in res["comps"]: continue
    suf, nec = [], []
    for i, pr in enumerate(pairs):
        (za, ma, ra), (zb, mb, rb) = caps[i]
        for direction in ("suf", "nec"):
            ids, p, src = (pr["b"], pr["pb"], (za, ma, ra)) if direction == "suf" else (pr["a"], pr["pa"], (zb, mb, rb))
            d = p["d"]
            if kind == "head":
                w = m.dhl[l]; m.zpatch = {l: [(h, d, src[0][l][h * w:(h + 1) * w])]}
            elif kind == "mlp": m.mpatch = {l: [(d, src[1][l])]}
            else: m.rpatch = {l: [(d, src[2][l])]}
            s = run(ids, p); m.zpatch, m.mpatch, m.rpatch = {}, {}, {}
            (suf if direction == "suf" else nec).append(s)
    suf, nec = np.array(suf), np.array(nec)
    res["comps"][key] = dict(suf=float(((suf - corr) / gap.mean()).mean()), nec=float(((clean - nec) / gap.mean()).mean()))
    outp.write_text(json.dumps(res))
outp.write_text(json.dumps(res))
top = sorted(((k, v) for k, v in res["comps"].items() if not k.startswith("resid")), key=lambda kv: -kv[1]["nec"])[:15]
print("necessity top", [(k, round(v["nec"], 3), round(v["suf"], 3)) for k, v in top])
top = sorted(((k, v) for k, v in res["comps"].items() if not k.startswith("resid")), key=lambda kv: -kv[1]["suf"])[:15]
print("sufficiency top", [(k, round(v["suf"], 3), round(v["nec"], 3)) for k, v in top])
print("resid", [(k, round(v["suf"], 2)) for k, v in res["comps"].items() if k.startswith("resid")])
print("secs", round(time.time() - t0))
