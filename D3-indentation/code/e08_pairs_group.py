"""D3 e08 (exploratory; group version of e04, same pairs and score). Patches whole head groups at the
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

name, npairs = sys.argv[1], int(sys.argv[2])
groups = {a.split("=")[0]: [tuple(int(x) for x in h[1:].split("H")) for h in a.split("=")[1].split(",")] for a in sys.argv[3:] if "=" in a}
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
outp = OUT / f"e08_{name}_{npairs}_{'_'.join(groups)}.json"
res = dict(n=len(pairs), clean=float(clean.mean()), corrupt=float(corr.mean()), gap=float(gap.mean()), comps={})

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
    o = fr(heads); rng = random.Random(len(g) * 7 + 1); rnd = []
    for _ in range(12):
        rh = [(l, rng.choice([h for h in range(m.H) if (l, h) not in heads])) for l, _ in heads]
        rnd.append(fr(rh))
    rs, rn = np.array([x[0] for x in rnd]), np.array([x[1] for x in rnd])
    res["comps"][g] = dict(heads=[f"L{l}H{h}" for l, h in heads], suf=o[0], nec=o[1], rand_suf=rs.tolist(), rand_nec=rn.tolist())
    print(f"{g}: suf {o[0]:.3f} (random {rs.mean():.3f}, max {rs.max():.3f}); nec {o[1]:.3f} (random {rn.mean():.3f}, max {rn.max():.3f})", flush=True)
    outp.write_text(json.dumps(res))
print("secs", round(time.time() - t0))
