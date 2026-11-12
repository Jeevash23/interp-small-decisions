"""D3 confirmation c02 (PROTOCOL.md): group patching on return/assignment pairs built from the SEALED generated
confirmation part (same construction, score and patching as e04/e08). Groups and 20 random layer-matched draws
per group (random.Random(2000 + draw)) are fixed here; each run is saved to outputs/conf/c02_gen.json as it
finishes. Usage: python c02_pairs.py [--smoke] [--time-limit S]
Derived from: D3 e08 (exploratory; group version of e04, same pairs and score). Patches whole head groups at the
decision position (clean into corrupt = sufficiency; corrupt into clean = necessity), against random head
groups matched by layer (12 draws). Usage: python e08_pairs_group.py <model> <n_pairs> NAME=L17H4,... ...
Original e04 docstring follows.
D3 e04 (exploratory): minimal pairs at terminal-statement dedents in generated exploration programs.
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

name, npairs = "qwen3-06b-bf16", 300
smoke = "--smoke" in sys.argv
if smoke: npairs = 12
GROUPS = {"CARRY": "L17H4,L11H9,L5H6,L17H6,L16H2,L6H5", "DED": "L23H2,L17H3,L17H4,L17H5,L18H6"}
groups = {g: [tuple(int(x) for x in h[1:].split("H")) for h in v.split(",")] for g, v in GROUPS.items()}
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
m = M(name, attn="sdpa"); t0 = time.time()
progs = load_chunks("gen", "explore" if smoke else "conf"); random.Random(4).shuffle(progs)
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
od = OUT / ("conf_smoke" if smoke else "conf"); od.mkdir(exist_ok=True)
outp = od / "c02_gen.json"
res = dict(n=len(pairs), clean=float(clean.mean()), corrupt=float(corr.mean()), gap=float(gap.mean()), comps={})
if outp.exists(): res["comps"] = json.loads(outp.read_text())["comps"]

print(res, flush=True)


def patch_run(heads, direction):
    out = []
    for i, pr in enumerate(pairs):
        (za, ma, ra), (zb, mb, rb) = caps[i]
        ids, p, src = (pr["b"], pr["pb"], za) if direction == "suf" else (pr["a"], pr["pa"], zb)
        d = p["d"]; m.zpatch = {}
        for l, h in heads:
            w = m.dhl[l]; m.zpatch.setdefault(l, []).append((h, d, src[l][h * w:(h + 1) * w]))
        out.append(run(ids, p)); m.zpatch = {}
    return np.array(out)


def fr(heads):
    suf, nec = patch_run(heads, "suf"), patch_run(heads, "nec")
    return float(((suf - corr) / gap.mean()).mean()), float(((clean - nec) / gap.mean()).mean())


for g, heads in groups.items():
    todo = [(g, heads)]
    for k in range(20):
        rng = random.Random(2000 + k)
        todo.append((f"{g}_rand{k}", [(l, rng.choice([h for h in range(m.H) if (l, h) not in heads])) for l, _ in heads]))
    for key, hs in todo:
        if key in res["comps"]: continue
        if time.time() - t0 > tlim: print("time limit; resume later"); break
        su, ne = fr(hs)
        res["comps"][key] = dict(heads=[f"L{l}H{h}" for l, h in hs], suf=su, nec=ne); outp.write_text(json.dumps(res))
print(json.dumps(dict(pairs=len(pairs), runs=len(res["comps"]), of=42, secs=round(time.time() - t0))))
