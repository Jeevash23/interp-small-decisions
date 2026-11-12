"""Independent check of every number, verdict, text rule and reference in the D3 write-up.

  python3 -B scripts/check_numbers.py [--records auto|conf|calib] [--numbers PATH]

Shares no code with make_numbers.py, build_report.py or the study's producing code. Round 1 is checked by the code
in this file. Round 2 is checked with code/r2_independent.py, which is checker code written without reading the
round-2 producers (make_numbers.py's round-2 part, r2c04_verdicts.py, outputs/r2/measures_*.json); this file
checks that it imports nothing from the study code. --records picks the round-2 confirmation records: auto (the
default) uses outputs/r2/conf/ when every run there is complete, otherwise the calibration records, which
numbers.tex holds under c2.* as a draft; the verdict keys (c2.v.*, c2.ntests, c2.npass) are then "pending conf".
--numbers checks another copy of numbers.tex (for mutation tests); the check files are then not written.
It recomputes:
  - the confirmation values from the per-point arrays (outputs/conf/c01_*_points_*.npy with their class arrays),
    random-set extremes from the raw run files, intervals with its own clustered bootstrap (tolerance 0.015),
    and every pass or fail from the thresholds, which it also finds in PROTOCOL.md;
  - the gate from the raw per-point records (outputs/e01_*.jsonl; pre-fix: outputs/v1-before-2230-fix/);
  - the material counts from the data files, and the freeze times from FROZEN.md;
  - the exploratory values from their raw output files (arrays and per-component JSON), and the figure data.
  - round 2: every c2.*, cal.*, x2.*, ck.*, gate.post.ol.*, x1.*, mat2.* and freeze.*2 value from raw records and
    files, its bootstrap intervals (tolerance 0.015), every verdict Q1-Q11, O1-O10, G1-G4 and the five question
    verdicts from the thresholds (each found in its PROTOCOL-2.md row), and the round-2 digests in FROZEN.md.
It then checks the text: every \\nv key defined, no hand-typed decimals except thresholds found in the frozen
charter or protocol, British spelling, no product or company names, no repository paths in captions, and the
references (every key cited, each arXiv entry matching its saved abstract page).
Writes checks/NUMBER-CHECK.md and checks/number-check.json; exits 1 if anything fails.
"""
import json, re, sys
from pathlib import Path
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
D3 = PAPER.parents[1]
OUT = D3 / "outputs"; CONF = OUT / "conf"
RES = []
ARGV = sys.argv[1:]
R2_MODE = ARGV[ARGV.index("--records") + 1] if "--records" in ARGV else "auto"
NUMBERS = Path(ARGV[ARGV.index("--numbers") + 1]).resolve() if "--numbers" in ARGV else PAPER / "numbers.tex"
WRITE = NUMBERS == PAPER / "numbers.tex"
PENDING = []  # round-2 keys that wait for the confirmation run


def check(name, ok, detail=""):
    RES.append(dict(name=name, ok=bool(ok), detail=str(detail)))


NV = {}
for line in NUMBERS.read_text().splitlines():
    m = re.match(r"\\expandafter\\def\\csname nv@(.+?)\\endcsname\{(.*)\}$", line)
    if m: NV[m.group(1)] = m.group(2)
SEEN = set()


def tonum(s): return float(s.replace("\u2212", "-").replace(",", "").replace("+", ""))


def same(key, x, dec=3):
    SEEN.add(key)
    if key not in NV: check(f"value {key}", False, "missing from numbers.tex"); return
    got = tonum(NV[key]); ok = abs(got - x) <= 0.5 * 10 ** -dec + 1e-9
    check(f"value {key}", ok, f"text {NV[key]} vs recomputed {x:.6f}")


def same_int(key, x):
    SEEN.add(key)
    check(f"count {key}", key in NV and int(tonum(NV[key])) == int(x), f"text {NV.get(key)} vs {x}")


def same_str(key, x):
    SEEN.add(key)
    check(f"text value {key}", NV.get(key) == x, f"text {NV.get(key)} vs {x}")


def rj(p): return json.loads(Path(p).read_text())


def lines(p): return [json.loads(l) for l in open(p) if l.strip()]


# ------------------------------------------------------------------ material and freezes
for part in ("explore", "conf"):
    same_int(f"mat.gen.{part}", len(lines(D3 / "data" / "gen" / f"part-{part}.jsonl")))
fetched = lines(D3 / "data" / "newcode" / "fetched.jsonl")
same_int("mat.nc.files", len(fetched)); same_int("mat.nc.repos", len({r["repo"] for r in fetched}))
same("mat.nc.mb", sum(r["bytes"] for r in fetched) / 1e6, 1)
same_int("mat.nc.listed", rj(D3 / "data" / "newcode" / "fetch_summary.json")["repos_listed"])
same_int("mat.nc.norm", rj(D3 / "data" / "newcode" / "build_summary.json")["normalised"])
tot = 0
for part in ("explore", "conf"):
    rows = lines(D3 / "data" / "newcode" / f"part-{part}.chunks.jsonl"); tot += len(rows)
    same_int(f"mat.nc.{part}.chunks", len(rows)); same_int(f"mat.nc.{part}.repos", len({r["repo"] for r in rows}))
same_int("mat.nc.chunks", tot)
ex_repos = {r["repo"] for r in lines(D3 / "data" / "newcode" / "part-explore.chunks.jsonl")}
cf_repos = {r["repo"] for r in lines(D3 / "data" / "newcode" / "part-conf.chunks.jsonl")}
check("no repository in both natural-code parts", not (ex_repos & cf_repos), len(ex_repos & cf_repos))
def _frozen_sha(p):
    """Release: the SHA-256 of the frozen original of a file in this release (tools/frozen_digest.py). A file changed
    for the release counts only if release-changes.json records exactly that change from the frozen digest; a file
    not shipped counts only if release-changes.json lists it with the frozen digest."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
    import frozen_digest
    return frozen_digest.sha256_as_frozen(p)


frozen = (D3 / "FROZEN.md").read_text()
mc = re.findall(r"^\| CHARTER\.md \| ([0-9a-f]{64}) \| \d{4}-\d\d-\d\d (\d\d:\d\d)", frozen, re.M)
mp = re.findall(r"^\| PROTOCOL\.md \| ([0-9a-f]{64}) \| \d{4}-\d\d-\d\d (\d\d:\d\d)", frozen, re.M)
same_str("freeze.charter", mc[0][1]); same_str("freeze.protocol", mp[0][1])
import hashlib
check("PROTOCOL.md digest matches FROZEN.md", _frozen_sha(D3 / "PROTOCOL.md") == mp[0][0])
check("CHARTER.md digest matches FROZEN.md", _frozen_sha(D3 / "CHARTER.md") == mc[0][0])

# ------------------------------------------------------------------ gate from raw per-point records
TAG = {"pythia-410m": "py", "qwen3-06b": "qw", "gemma4-e2b": "ge", "gpt2": "g2"}


def mean_where(xs, key, cond):
    v = [x[key] for x in xs if cond(x)]
    return sum(v) / len(v)


for tag, base in (("post", OUT), ("pre", OUT / "v1-before-2230-fix")):
    for m, k in TAG.items():
        f = base / f"e01_{m}_newcode.jsonl"
        if not f.exists(): continue
        xs = [x for x in lines(f) if "mass" in x]; p = f"gate.{tag}.{k}"
        same_int(f"{p}.n", len(xs))
        same(f"{p}.head", mean_where(xs, "top1", lambda x: x["header"]))
        same(f"{p}.termt", mean_where(xs, "top1", lambda x: x["terminal"]))
        same(f"{p}.terms", mean_where(xs, "p_shallow_open", lambda x: x["terminal"]))
        t1, cp = mean_where(xs, "top1", lambda x: True), mean_where(xs, "copy_ok", lambda x: True)
        same(f"{p}.top1", t1); same(f"{p}.copy", cp); same(f"{p}.marg", t1 - cp)
        same(f"{p}.other", mean_where(xs, "other", lambda x: True))
        gf = base / f"e01_{m}_gen.jsonl"
        if gf.exists():
            gs = [x for x in lines(gf) if "mass" in x]
            worst = max(mean_where(gs, "p_shallow_notopen", lambda x, w=w: str(x["width"]) == w and x["cls"] == "shallower") for w in ("2", "3", "mixed"))
            same(f"{p}.open", worst)
        if tag == "post" and k in ("py", "qw"):
            same(f"{p}.same", mean_where(xs, "top1", lambda x: x["cls"] == "same"))
        if tag == "post" and k == "qw":
            same_int(f"{p}.ntermpts", sum(x["terminal"] for x in xs)); same_int(f"{p}.nheadpts", sum(x["header"] for x in xs))
            same(f"{p}.ded", mean_where(xs, "top1", lambda x: x["cls"] == "shallower"))
            same(f"{p}.ntded", mean_where(xs, "top1", lambda x: x["cls"] == "shallower" and not x["terminal"]))
        if tag == "pre" and k == "py":
            same_int("gate.v1.py.termpts", sum(x["terminal"] for x in xs))
# Pythia: first indented line of a chunk, by a separate pass over chunk order
xs = [x for x in lines(OUT / "e01_pythia-410m_newcode.jsonl") if "mass" in x]
by_chunk = {}
for x in xs: by_chunk.setdefault(x["chunk"], []).append(x)
first, rest = [], []
for ch, pts in by_chunk.items():
    indented_seen = False
    for x in sorted(pts, key=lambda y: y["line"]):
        if x["header"]:
            (first if (x["prev"] == 0 and not indented_seen) else rest).append(x)
        if x["true"] > 0: indented_seen = True
same_int("py.first.n", len(first)); same_int("py.rest.n", len(rest))
same("py.first.top1", sum(x["top1"] for x in first) / len(first)); same("py.rest.top1", sum(x["top1"] for x in rest) / len(rest))
for kk in (4, 8):
    same(f"py.first.q{kk}", sum(x["mass"][x["K"].index(kk)] / sum(x["mass"]) for x in first) / len(first), 2)

# ------------------------------------------------------------------ exploration, from raw outputs
def att(cls, cat, l, h): return float(np.load(OUT / f"e03_qwen3-06b_newcode_{cls}_{cat}.npy")[l][h])


same("x.att.L20H14.term.open", att("term", "openstarts", 20, 14), 2); same("x.att.L19H3.term.open", att("term", "openstarts", 19, 3), 2)
same("x.att.L22H9.term.open", att("term", "openstarts", 22, 9), 2); same("x.att.L22H9.same.prev", att("same", "prevstart", 22, 9), 2)
same("x.att.L1H0.head.word", att("header", "prevword", 1, 0), 2); same("x.att.L5H6.term.word", att("term", "prevword", 5, 6), 2)
e2 = rj(OUT / "e02_qwen3-06b-bf16_newcode_160.json"); comps = [k for k in e2 if re.match(r"^(L\d+H\d+|mlp\d+)$", k)]
same_int("x.e2.ncomp", len(comps)); same("x.e2.maxterm", -min(e2[k]["term"] for k in comps)); same("x.e2.L23H2.ded", e2["L23H2"]["dedent"])
e2f = rj(OUT / "e02_qwen3-06b_newcode_160.json"); both = [k for k in comps if k in e2f]
same_int("x.bf16.ncomp", len(both))
rs = []
for meas in ("header", "same", "term", "dedent", "term_shopen"):
    a = np.array([e2f[k][meas] for k in both]); b = np.array([e2[k][meas] for k in both])
    rs.append(float(((a - a.mean()) * (b - b.mean())).sum() / np.sqrt(((a - a.mean()) ** 2).sum() * ((b - b.mean()) ** 2).sum())))
same("x.bf16.r", min(rs)); same("x.bf16.base", max(abs(e2f["base"][q] - e2["base"][q]) for q in e2["base"]))
e6 = rj(OUT / "e06_qwen3-06b-bf16_newcode_120_OPEN_PREV_KW_DED.json")
same("x.e6.base.ded", e6["base"]["lq"]["dedent"]); same("x.e6.ded.ded", e6["groups"]["DED"]["obs"]["lq"]["dedent"])
rr = sorted(r["lq"]["dedent"] for r in e6["groups"]["DED"]["random"]); same("x.e6.rand.lo", rr[0]); same("x.e6.rand.hi", rr[-1])
same("x.e6.base.psame", e6["base"]["p_same"]["dedent"]); same("x.e6.ded.psame", e6["groups"]["DED"]["obs"]["p_same"]["dedent"])
same("x.e6.ded.termopen", e6["groups"]["DED"]["obs"]["p_open"]["term"]); same_int("x.e6.n.ded", e6["n"]["dedent"])
check("e06 DED group is the paper's DED set", e6["groups"]["DED"]["heads"] == ["L23H2", "L17H3", "L17H4", "L17H5", "L18H6"])
lo = rj(OUT / "e06_qwen3-06b-bf16_newcode_120_DEDno23H2_DEDno17H3_DEDno17H4_DEDno17H5_DEDno18H6.json")["groups"]
same("x.loo.no23", lo["DEDno23H2"]["obs"]["lq"]["dedent"])
ov = sorted(lo[g]["obs"]["lq"]["dedent"] for g in lo if g != "DEDno23H2"); same("x.loo.lo", ov[0]); same("x.loo.hi", ov[-1])
e4 = rj(OUT / "e04_qwen3-06b-bf16_120.json")
same("x.e4.gap", e4["gap"], 2); same_int("x.e4.n", e4["n"])
for l in (4, 5, 16, 17, 20):
    same(f"x.e4.r{l}.suf", e4["comps"][f"resid{l}"]["suf"], 2); same(f"x.e4.r{l}.nec", e4["comps"][f"resid{l}"]["nec"], 2)
same("x.e4.L17H4.suf", e4["comps"]["L17H4"]["suf"]); same("x.e4.L17H4.nec", e4["comps"]["L17H4"]["nec"])
heads_only = {k: v for k, v in e4["comps"].items() if k.startswith("L")}
check("L17H4 is the largest single head by necessity in e04", max(heads_only, key=lambda k: heads_only[k]["nec"]) == "L17H4")
fig = [l.split() for l in (PAPER / "fig-resid.dat").read_text().splitlines()[1:]]
check("figure data matches e04 residual patching", len(fig) == 28 and all(abs(float(s) - e4["comps"][f"resid{int(l)}"]["suf"]) < 1e-4 and abs(float(nn) - e4["comps"][f"resid{int(l)}"]["nec"]) < 1e-4 for l, s, nn in fig))
e8 = rj(OUT / "e08_qwen3-06b-bf16_120_TOP6_TOP12_EARLY_OPEN.json")["comps"]["TOP6"]
check("e08 TOP6 is the paper's CARRY set", e8["heads"] == ["L17H4", "L11H9", "L5H6", "L17H6", "L16H2", "L6H5"])
same("x.e8.top6.nec", e8["nec"]); same("x.e8.top6.suf", e8["suf"]); same("x.e8.top6.rnec", max(e8["rand_nec"])); same("x.e8.top6.rsuf", max(e8["rand_suf"]))
e7 = rj(OUT / "e07b_qwen3-06b-bf16_newcode_250.json")
for cls in ("term", "dedent"):
    tab = {r[0]: r for r in e7[cls]["top"] + e7[cls]["bottom"]}
    for h in ("L23H2", "L22H8"):
        same(f"x.e7.{cls}.{h}.t", tab[h][2], 2); same(f"x.e7.{cls}.{h}.c", tab[h][3], 2)
    same(f"x.e7.{cls}.mlp27", tab["mlp27"][1], 2)
dla = np.load(OUT / "e07b_qwen3-06b-bf16_newcode_term_dla.npy")
check("e07b split for L23H2 matches its saved array", abs(dla[23, 2, 0] - e7["term"]["top"][[r[0] for r in e7["term"]["top"]].index("L23H2")][2]) < 1e-3)
e5 = rj(OUT / "e05_qwen3-06b_newcode_5005.json")
same_int("x.e5.n", e5["n"]); same("x.e5.prev3", e5["3"]["prev"], 2); same("x.e5.cls0", e5["0"]["cls_acc"], 2); same("x.e5.true28", e5["28"]["true"], 2)
nd = [e5[str(l)]["nded"] for l in range(29)]; same("x.e5.ndedmax", max(nd), 2); same_str("x.e5.ndedarg", str(int(np.argmax(nd))))
e9 = rj(OUT / "e09_qwen3-06b-bf16_250_READ.json")
rows9 = e9["rows"]; same_int("x.e9.n", sum(1 for r in rows9 if "base" in r))
for w in ("return", "pass"):
    for kind, key in (("kw", "kw"), ("ctl", "ctl"), ("real", "real")):
        v = [r[f"{key}:{w}"] for r in rows9 if f"{key}:{w}" in r]; same(f"x.e9.{w}.{kind}", sum(v) / len(v))
sd = [r["str:kw:return"] - r["str:ctl:return"] for r in rows9 if "str:kw:return" in r]; same("x.e9.strret", sum(sd) / len(sd))
worst = 0
for f in ("e09_qwen3-06b-bf16_250_READ.json", "e09_pythia-410m_250.json", "e09_gemma4-e2b_250.json"):
    rws = rj(OUT / f)["rows"]
    for w in ("return", "pass", "break", "continue", "raise"):
        dd = [r[f"kw:{w}"] - r[f"ctl:{w}"] for r in rws if f"kw:{w}" in r]; re_ = [r[f"real:{w}"] for r in rws if f"real:{w}" in r]
        worst = max(worst, (sum(dd) / len(dd)) / (sum(re_) / len(re_)))
same("x.e9.maxratio", worst, 2)
e10 = rj(OUT / "e10_qwen3-06b-bf16_H25_STEP_WRITE_MLP26_MLPS.json")
for w, key in (("2", "w2"), ("4", "w4"), ("nat", "nat")):
    same(f"x.e10.h25.ded.{key}", e10["groups"]["H25"]["obs"][f"dedent|{w}"]["lq"] - e10["base"][f"dedent|{w}"]["lq"])
check("e10 H25 group is L25H9 alone", e10["groups"]["H25"]["comps"] == ["L25H9"])
e6g = rj(OUT / "e06_qwen3-06b-bf16_gen_200_DED_TERM.json")
same("x.e6g.ded.term", e6g["groups"]["DED"]["obs"]["lq"]["term"] - e6g["base"]["lq"]["term"])
same("x.e6g.ded.termr", min(r["lq"]["term"] for r in e6g["groups"]["DED"]["random"]) - e6g["base"]["lq"]["term"])

# ------------------------------------------------------------------ confirmation from per-point arrays
def arr(src, name): return np.load(CONF / f"c01_{src}_points_{name}.npy")


META = {s: np.load(CONF / f"c01_{s}_meta.npy", allow_pickle=True) for s in ("newcode", "gen")}
ROW = {"lq": 0, "p_same": 1, "p_open": 2}


def cmean(src, name, meas, cls):
    c = META[src][0]; return float(arr(src, name)[ROW[meas]][c == cls].mean())


def cdelta(src, name, meas, cls): return cmean(src, name, meas, cls) - cmean(src, "base", meas, cls)


nraw = rj(CONF / "c01_newcode.json"); graw = rj(CONF / "c01_gen.json"); praw = rj(CONF / "c02_gen.json")
for src, tag, raw in (("newcode", "nc", nraw), ("gen", "gen", graw)):
    c = META[src][0]
    for cls in ("header", "same", "term", "dedent"):
        same_int(f"c.n.{tag}.{cls}", int((c == cls).sum())); same(f"c.base.{tag}.{cls}", cmean(src, "base", "lq", cls))
        check(f"run-file base agrees with per-point base ({src}, {cls})", abs(raw["base"]["lq"][cls] - cmean(src, "base", "lq", cls)) < 1e-9)
    same(f"c.base.{tag}.psame", cmean(src, "base", "p_same", "dedent"))
same_int("c.nchunks.nc", len(lines(D3 / "data" / "newcode" / "part-conf.chunks.jsonl")))
same_int("c.nchunks.gen", len(lines(D3 / "data" / "gen" / "part-conf.jsonl")))
check("every held-out natural chunk was used", nraw["n_chunks_used"] == len(lines(D3 / "data" / "newcode" / "part-conf.chunks.jsonl")))
check("every held-out generated program was used", graw["n_chunks_used"] == len(lines(D3 / "data" / "gen" / "part-conf.jsonl")))


def rand_deltas(raw, grp, meas, cls):
    v = [raw[k][meas][cls] - raw["base"][meas][cls] for k in raw if k.startswith(grp + "_rand")]
    return v


H1 = cdelta("newcode", "DED", "lq", "dedent"); r1 = rand_deltas(nraw, "DED", "lq", "dedent")
H2 = cdelta("newcode", "DED", "p_same", "dedent"); r2 = rand_deltas(nraw, "DED", "p_same", "dedent")
H3o = cmean("newcode", "DED", "p_open", "term"); H3d = cdelta("newcode", "DED", "lq", "term")
H4 = cdelta("newcode", "DED", "lq", "same")
H5 = cdelta("newcode", "TERM", "lq", "term"); r5 = rand_deltas(nraw, "TERM", "lq", "term")
H6 = cdelta("newcode", "TERM", "lq", "dedent")
H7 = cdelta("newcode", "TERMDED", "lq", "term")
H8 = cdelta("gen", "DED", "lq", "dedent"); r8 = rand_deltas(graw, "DED", "lq", "dedent")
check("20 random sets each for DED (natural), TERM (natural), DED (generated)", len(r1) == len(r5) == len(r8) == 20)
same("c.H1", H1); same("c.H1.abs", -H1, 2); same("c.H1.r", min(r1))
same("c.H2", H2); same("c.H2.r", max(r2)); same("c.H2.after", cmean("newcode", "DED", "p_same", "dedent"))
same("c.H3.open", H3o); same("c.H3.d", H3d); same("c.H3.ratio", abs(H3d) / abs(H1))
same("c.H4", H4); same("c.H5", H5); same("c.H5.r", min(r5)); same("c.H6", H6)
same("c.H7", H7); same("c.H7.sum", H5 + H3d); same("c.H7.psame", cmean("newcode", "TERMDED", "p_same", "term"))
same("c.H8", H8); same("c.H8.r", min(r8))
cm = praw["comps"]; rn = [v["nec"] for k, v in cm.items() if k.startswith("CARRY_rand")]; rsf = [v["suf"] for k, v in cm.items() if k.startswith("CARRY_rand")]
check("20 random sets for CARRY", len(rn) == 20)
same("c.H9.nec", cm["CARRY"]["nec"]); same("c.H9.suf", cm["CARRY"]["suf"]); same("c.H9.rnec", max(rn)); same("c.H9.rsuf", max(rsf))
same_int("c.pairs", praw["n"]); same("c.gap", praw["gap"], 2)
same("c.dedpairs.nec", cm["DED"]["nec"]); same("c.dedpairs.suf", cm["DED"]["suf"])
check("named sets in the run files match the protocol", nraw["DED"]["heads"] == ["L23H2", "L17H3", "L17H4", "L17H5", "L18H6"]
      and nraw["TERM"]["heads"] == ["L11H9", "L5H6", "L17H6", "L16H2", "L23H6", "L22H8", "L23H3", "L25H9", "L19H6"]
      and cm["CARRY"]["heads"] == ["L17H4", "L11H9", "L5H6", "L17H6", "L16H2", "L6H5"])
# verdicts from the thresholds, and the thresholds in PROTOCOL.md
proto = (D3 / "PROTOCOL.md").read_text()
rules = {"H1": (H1 <= -0.40 and H1 < min(r1), ["−0.40"]), "H2": (H2 >= 0.10 and H2 > max(r2), ["+0.10"]),
         "H3": (H3o >= 0.95 and abs(H3d) <= 0.25 * abs(H1), ["0.95", "0.25"]), "H4": (H4 >= -0.01, ["−0.01"]),
         "H5": (H5 <= -0.03 and H5 < min(r5), ["−0.03"]), "H6": (abs(H6) <= 0.10, ["0.10"]), "H7": (H7 < H5 + H3d, []),
         "H8": (H8 <= -0.25 and H8 < min(r8), ["−0.25"]),
         "H9": (cm["CARRY"]["nec"] >= 0.40 and cm["CARRY"]["suf"] >= 0.20 and cm["CARRY"]["nec"] > max(rn) and cm["CARRY"]["suf"] > max(rsf), ["0.40", "0.20"])}
ver = {v["id"]: v["verdict"] for v in rj(CONF / "verdicts.json")["verdicts"]}
for h, (ok, ths) in rules.items():
    check(f"{h} verdict re-derived ({'pass' if ok else 'fail'}) equals the evaluation file", ("pass" if ok else "fail") == ver[h])
    line = next((l for l in proto.splitlines() if l.startswith(f"| **{h}**")), "")
    for t in ths: check(f"{h} threshold {t} appears in its PROTOCOL.md row", t in line, line[:80])
same_str("c.npass", str(sum(r[0] for r in rules.values()))); same_str("c.ntests", str(len(rules)))
check("every confirmation verdict printed in Table 2 is 'pass' and matches", all(v == "pass" for v in ver.values()))


# intervals: own clustered bootstrap (different random numbers), tolerance 0.015
def boot(src, name, meas, cls, B=2000, seed=12345):
    c, cl = META[src][0], META[src][1]
    dlt = (arr(src, name)[ROW[meas]] - arr(src, "base")[ROW[meas]])[c == cls]; g = cl[c == cls]
    keys, inv = np.unique(g, return_inverse=True)
    sums = np.bincount(inv, weights=dlt, minlength=len(keys)); cnts = np.bincount(inv, minlength=len(keys))
    rng = np.random.default_rng(seed); out = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)); out[b] = (w * sums).sum() / (w * cnts).sum()
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


for h, (src, name, meas, cls) in {"H1": ("newcode", "DED", "lq", "dedent"), "H2": ("newcode", "DED", "p_same", "dedent"), "H3": ("newcode", "DED", "lq", "term"),
                                  "H4": ("newcode", "DED", "lq", "same"), "H5": ("newcode", "TERM", "lq", "term"), "H6": ("newcode", "TERM", "lq", "dedent"),
                                  "H7": ("newcode", "TERMDED", "lq", "term"), "H8": ("gen", "DED", "lq", "dedent")}.items():
    a, b = boot(src, name, meas, cls); SEEN.update({f"c.{h}.lo", f"c.{h}.hi"})
    check(f"interval {h} within 0.015 of an independent bootstrap", abs(tonum(NV[f"c.{h}.lo"]) - a) <= 0.015 and abs(tonum(NV[f"c.{h}.hi"]) - b) <= 0.015,
          f"text [{NV[f'c.{h}.lo']}, {NV[f'c.{h}.hi']}] vs [{a:.3f}, {b:.3f}]")

# ------------------------------------------------------------------ round 2 (PROTOCOL-2.md)
R2I_PATH = D3 / "code" / "r2_independent.py"
_imports = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", R2I_PATH.read_text(), re.M)
check("round-2 recomputation imports only the standard library and numpy (no study code)",
      set(_imports) <= {"argparse", "json", "re", "sys", "pathlib", "numpy"}, sorted(set(_imports)))
sys.path.insert(0, str(R2I_PATH.parent))
import r2_independent as R2I  # noqa: E402  (checker code; see the docstring)

_st = R2I.completeness("conf", R2I.records_block("conf"))
conf_complete = all(a == b for a, b in _st.values())
if R2_MODE == "auto": R2_MODE = "conf" if conf_complete else "calib"
check(f"round-2 records used: {R2_MODE}" + ("" if R2_MODE == "conf" else " (draft: conf not complete; c2.* holds calibration values)"),
      R2_MODE != "conf" or conf_complete, ", ".join(f"{k} {a}/{b}" for k, (a, b) in _st.items()))
R2V, R2META, R2B = R2I.compute(R2_MODE)
for k, v in R2META["completeness"].items():
    check(f"round-2 {R2_MODE} records complete: {k}", v["complete"], f"{v['done']} of {v['expected']}")
for t_ in R2I.MODELS:
    if R2B.get(t_): check(f"r2c01 {t_}: knock-out of every other head of the layer recorded (15)", R2B[t_][0]["n.others"] == 15, R2B[t_][0]["n.others"])
    if R2B.get(t_ + "g"): check(f"r2c02 {t_}: no program skipped", R2B[t_ + "g"][0]["n.skip"] == 0, R2B[t_ + "g"][0]["n.skip"])
if R2B.get("g"):
    check("r2c03: the named group is DED-O as in the protocol", R2B["g"][0]["heads"] == ["L13H1", "L10H12", "L11H0", "L15H4"], R2B["g"][0]["heads"])
    check("r2c03: 20 random groups", R2B["g"][0]["n.rand"] == 20, R2B["g"][0]["n.rand"])
    check("r2c03: run-file class means equal the per-point means", R2META["consistency"]["r2c03 run-file means vs per-point means (max abs diff)"] < 1e-9)
check("X1: all 22 tests re-derived from the raw run files agree with X1's verdicts.json",
      all(v["matches_verdicts_json"] for v in R2META["x1_verdicts"].values()), [k for k, v in R2META["x1_verdicts"].items() if not v["matches_verdicts_json"]])
NUMRE = re.compile(r"^[+−-]?[\d,]*\d(\.\d+)?$")
R2VERD = {k for k in R2V if k.startswith("c2.v.")} | {"c2.ntests", "c2.npass"}
for k, mine in R2V.items():
    if k in R2VERD: continue
    if k not in NV: continue  # a value the paper does not print; any printed round-2 key left unchecked fails below
    if ".ci." in k or re.match(r"^c2\.g\.ci\.", k):
        SEEN.add(k)
        check(f"interval {k} within 0.015 of an independent bootstrap", abs(tonum(NV[k]) - tonum(mine)) <= 0.015 + 1e-9, f"text {NV[k]} vs {mine}")
    elif not NUMRE.match(mine.replace(" ", "")) or re.match(r"^L\d+H\d+$|^MLP\d+$", mine):
        same_str(k, mine)
    elif "." not in mine.lstrip("+−-").replace(",", "") or k.endswith("layer_return_arg"):
        same_int(k, tonum(mine))
    else:
        dec = len(mine.split(".")[-1]); same(k, tonum(mine), dec)


def _norm_v(x): return re.sub(r"[^a-z ]", "", re.sub(r"\\[a-zA-Z]+", "", str(x)).lower()).strip()


R2DER = R2META["verdict_details"]
for i in R2I.PRED_IDS + list(R2I.QUESTIONS):
    k = f"c2.v.{i}"; der = R2DER[i]["verdict"]
    if k in NV:
        SEEN.add(k); check(f"{i} verdict re-derived ({der}) equals numbers.tex", _norm_v(NV[k]) == der, f"text {NV[k]} | {R2DER[i]['detail']}")
    elif R2_MODE != "conf" or i in R2I.QUESTIONS:
        PENDING.append(k) if i not in R2I.QUESTIONS else None
    else:
        check(f"{i} verdict printed in numbers.tex", False, "missing")
for k in ("c2.ntests", "c2.npass"):
    if k in NV: same_str(k, R2V[k])
    elif R2_MODE != "conf": PENDING.append(k)
unknown_v = sorted(k for k in NV if k.startswith("c2.v.") and k not in SEEN)
check("every c2.v.* key in numbers.tex is a known verdict", not unknown_v, unknown_v)
vf = OUT / "r2" / "conf" / "verdicts.json"
if R2_MODE == "conf" and vf.exists():
    found = {}

    def _walk(o):
        if isinstance(o, dict):
            ident = o.get("id") or o.get("name")
            if isinstance(ident, str) and re.fullmatch(r"[QOG]\d+", ident):
                vv = o.get("verdict", o.get("result", o.get("passed")))
                found[ident] = ("pass" if vv else "fail") if isinstance(vv, bool) else _norm_v(vv)
            for kk, vv in o.items():
                if re.fullmatch(r"[QOG]\d+", str(kk)) and not isinstance(vv, (dict, list)):
                    found[kk] = ("pass" if vv else "fail") if isinstance(vv, bool) else _norm_v(vv)
                elif re.fullmatch(r"[QOG]\d+", str(kk)) and isinstance(vv, dict) and ("passed" in vv or "verdict" in vv):
                    x = vv.get("verdict", vv.get("passed")); found[kk] = ("pass" if x else "fail") if isinstance(x, bool) else _norm_v(x)
                _walk(vv)
        elif isinstance(o, list):
            for x in o: _walk(x)
    _walk(rj(vf))
    check("outputs/r2/conf/verdicts.json holds a verdict for each of the 25 predictions", set(found) >= set(R2I.PRED_IDS), sorted(set(R2I.PRED_IDS) - set(found)))
    for i in R2I.PRED_IDS:
        if i in found: check(f"{i} verdict re-derived ({R2DER[i]['verdict']}) equals the evaluation file", found[i] == R2DER[i]["verdict"], found[i])
# thresholds, each in its PROTOCOL-2.md row (the prediction column, not the calibration column)
proto2 = (D3 / "PROTOCOL-2.md").read_text()
TH2 = {"Q1": ["≥ 0.30", "≥ 2.5 ×"], "Q2": ["≤ −0.05", "every other layer-23 head", "2 ×"], "Q3": ["≥ −0.02", "≤ 0.25 ×"],
       "Q4": ["≤ 0.5 ×"], "Q5": ["≥ 0.60", "≥ 0.02"], "Q6": ["≥ +1.0", "≥ 5 ×", "floored at 0.02", "+ 0.05"], "Q7": ["≤ −0.10"],
       "Q8": ["≥ +1.0", "≥ 5 ×", "floored at 0.02", "≥ 0.20", "+ 0.10"], "Q9": ["≥ 0.40"], "Q10": ["≤ 0.35 ×"], "Q11": ["at most 0.05"],
       "O1": ["As Q1", "≥ 0.20", "≥ 1.4 ×"], "O2": ["As Q2", "every other layer-13 head"], "O3": ["As Q3"], "O4": ["As Q4"],
       "O5": ["≥ 0.45", "≥ 0.15"], "O6": ["As Q6", "≥ +0.8", "≥ 4 ×", "+ 0.05"], "O7": ["≤ −0.08"],
       "O8": ["As Q8", "≥ +0.8", "≥ 4 ×", "≥ 0.15", "+ 0.10"], "O9": ["As Q9", "≥ 0.40"], "O10": ["As Q10", "≤ 0.35 ×"],
       "G1": ["≤ −0.30", "20 random"], "G2": ["≤ 0.25 ×", "≥ 0.90"], "G3": ["≥ −0.02"], "G4": ["≥ 0.08"]}
check("a threshold list for each of the 25 predictions", set(TH2) == set(R2I.PRED_IDS))
for i, ths in TH2.items():
    row = next((l for l in proto2.splitlines() if l.startswith(f"| **{i}** |")), "")
    col = row.split(" | ")[1] if row.count(" | ") >= 2 else ""
    for t_ in ths: check(f"{i} threshold '{t_}' appears in its PROTOCOL-2.md row", t_ in col, col[:90])
for qn, ids in R2I.QUESTIONS.items():
    lab = qn.replace("question", "")
    row = next((l for l in proto2.splitlines() if l.startswith(f"- **Question {lab}**")), "")
    named = re.findall(r"[QOG]\d+", row.split("):", 1)[-1].split("Qualifier")[0])
    check(f"question {lab}: named predictions match PROTOCOL-2.md", named == ids, f"{named} vs {ids}")
# round-2 digests
_fz = (D3 / "FROZEN.md").read_text()


def _digest_rows(section):
    sec = _fz.split(section, 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| (\S+)(?: \([^)]*\))? \| ([0-9a-f]{64}) \|", sec, re.M)


for f_, dg in [(f_, d_) for f_, d_ in _digest_rows("## Round 2 charter")] + _digest_rows("## Round 2 confirmation protocol") + _digest_rows("## Round 2 held-out material"):
    check(f"{f_} digest matches FROZEN.md", _frozen_sha(D3 / f_) == dg)
check("PROTOCOL-2.md and CHARTER-2.md both digested in FROZEN.md",
      {"PROTOCOL-2.md", "CHARTER-2.md"} <= {f_ for f_, _ in _digest_rows("## Round 2 charter") + _digest_rows("## Round 2 confirmation protocol")})
check("PROTOCOL-2.md is read-only on disk", not ((D3 / "PROTOCOL-2.md").stat().st_mode & 0o222))

# ------------------------------------------------------------------ text
tex = (PAPER / "main.tex").read_text()
body = tex[tex.index("\\begin{document}"):]
used = set(re.findall(r"\\nv\{([^}]+)\}", body))
pend = set(PENDING) if R2_MODE != "conf" else set()
check("every \\nv key used in the text is defined" + (" (round-2 verdict keys pending conf)" if pend else ""), used - pend <= set(NV), sorted(used - pend - set(NV)))
check("every \\nv key used in the text is checked here", used - pend <= SEEN, sorted(used - pend - SEEN))
check("every number in numbers.tex is checked here", set(NV) <= SEEN, sorted(set(NV) - SEEN))
plain = re.sub(r"\\nv\{[^}]+\}", "", body)
plain = re.sub(r"\\(begin|end)\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})?", "", plain)
plain = re.sub(r"\\begin\{axis\}.*?\\end\{axis\}", "", body, flags=re.S) if False else plain
axis = re.search(r"\\begin\{axis\}\[(.*?)\]", body, re.S).group(1)
plain = plain.replace(axis, "")
decs = re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", plain)
allowed = {"0.90", "0.7", "0.5", "0.10", "0.95", "0.40", "0.25", "0.03", "0.01", "0.20"}
charter = (D3 / "CHARTER.md").read_text()
allowed2 = {"0.02", "0.05", "0.08", "0.15", "0.30", "0.35", "0.45", "0.60", "0.8", "1.0", "1.4", "2.5"}  # round-2 thresholds
check("no hand-typed decimals other than frozen thresholds", set(decs) <= allowed | allowed2, sorted(set(decs) - allowed - allowed2))
for t in sorted(allowed2):
    check(f"round-2 threshold {t} appears in the frozen PROTOCOL-2.md", t in proto2)
AS_PCT = {"0.90": "90%", "0.5": "50%"}  # the charter states these two as percentages
for t in sorted(set(decs) & allowed):
    check(f"threshold {t} appears in the frozen charter or protocol", t in charter or t in proto or AS_PCT.get(t, "@@") in charter)
US = ["behavior", "analyze", "analyzed", "modeling", "color", "center", "normalize", "normalized", "recognize", "organize", "favor", "labeled", "generalize"]
low = re.sub(r"\\[a-zA-Z]+", " ", body).lower()  # LaTeX command names (e.g. \\centering) are not prose
check("British spelling (no US forms)", not [w for w in US if re.search(rf"\b{w}\w*", low)], [w for w in US if re.search(rf"\b{w}\w*", low)])
check("no product or company names in the text", not re.search(r"\b(claude|codex|anthropic|openai|chatgpt)\b", low))
caps = re.findall(r"\\caption\{(.*?)\}\n", body, re.S)
check("no repository paths in captions", not any(re.search(r"(outputs/|code/|data/|\.py|\.json|\.md)", c) for c in caps))
check("AI-use statement says language-model assistants", "Language-model assistants" in body)
check("held-out is used only for the sealed parts", "post-cut-off" in body and not re.search(r"held[- ]out (exploration|post-cut-off)", low))

# ------------------------------------------------------------------ references
bib = (PAPER / "refs.bib").read_text()
keys = re.findall(r"@\w+\{([^,]+),", bib)
cited = set(k.strip() for c in re.findall(r"\\cite[tp]?\{([^}]+)\}", body) for k in c.split(","))
check("every reference is cited and every citation has an entry", set(keys) == cited, (sorted(set(keys) - cited), sorted(cited - set(keys))))
abs_ = rj(PAPER / "sources" / "arxiv-abs-pages.json")
for ent in re.split(r"\n(?=@)", bib):
    m = re.match(r"\s*@\w+\{([^,]+),", ent)
    if not m: continue
    k = m.group(1); ep = re.search(r"(?:eprint\s*=\s*\{|arXiv:)(\d{4}\.\d{5})", ent)
    if not ep:
        check(f"reference {k}: non-arXiv entry (checked in REFERENCE-CHECK.md)", k == "radford2019language"); continue
    s = abs_.get(ep.group(1))
    if s is None: check(f"reference {k}: saved abstract page", False, ep.group(1)); continue
    t = re.search(r"title\s*=\s*\{(.*)\},", ent).group(1); norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    check(f"reference {k}: title matches arXiv {ep.group(1)}", norm(t) == norm(s["title"]), f"{t} | {s['title']}")
    fa = re.search(r"author\s*=\s*\{(.*?)\}?,\n", ent).group(1).split(" and ")[0].strip("{}")
    surname = fa.split(",")[0].strip("{} ") if "," in fa else fa.split()[-1]
    check(f"reference {k}: first author matches", surname.lower() in s["authors"][0].lower() or surname.lower() in " ".join(s["authors"]).lower() or "team" in surname.lower(), f"{surname} | {s['authors'][0]}")
    yr = re.search(r"year\s*=\s*\{(\d{4})\}", ent).group(1)
    venue = re.search(r"booktitle\s*=", ent) is not None  # conference entries carry the venue year
    check(f"reference {k}: year matches first version" + (" or venue year" if venue else ""), yr == s["date"][:4] or (venue and int(yr) == int(s["date"][:4]) + 1), f"{yr} | {s['date']}")
src_b = (PAPER / "sources" / "html-2607.04167.html").read_text(errors="replace")
check("Bassi and Tomar: indent level 0-5 in units of four spaces (as stated)", "0–5 in units of 4 spaces" in re.sub(r"<[^>]+>", "", src_b))
check("Bassi and Tomar: three models as stated", all(x in src_b for x in ("Gemma-2-2B", "Gemma-2-9B", "Qwen3-4B")))
check("Bassi and Tomar: tests informational concentration, not behavioural causality", "not behavioral causality" in re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", src_b)))
check("Gao et al.: an indentation task is among their Python tasks", "indent_for" in (PAPER / "sources" / "html-2511.13653.html").read_text(errors="replace"))

# ------------------------------------------------------------------ write
nf = sum(not r["ok"] for r in RES)
pend_note = f" Round-2 records: {R2_MODE}." + (f" Pending conf (not yet in numbers.tex): {', '.join(sorted(set(PENDING)))}." if PENDING and R2_MODE != "conf" else "")
if WRITE:
    (PAPER / "checks" / "number-check.json").write_text(json.dumps(dict(records=R2_MODE, pending=sorted(set(PENDING)), checks=RES), indent=1, ensure_ascii=False))
    md = [f"# Number check: D3 write-up, v2 (9 November 2026)", "", f"{len(RES)} checks, {nf} failed.{pend_note} Produced by `scripts/check_numbers.py`, which shares no code with `scripts/make_numbers.py` or the study's producing code.", "",
          "| check | result | detail |", "|---|---|---|"] + [f"| {r['name']} | {'ok' if r['ok'] else 'FAIL'} | {r['detail'][:120].replace('|', '/')} |" for r in RES]
    (PAPER / "checks" / "NUMBER-CHECK.md").write_text("\n".join(md) + "\n")
print(f"{len(RES)} checks, {nf} failed (numbers file: {NUMBERS}).{pend_note}")
for r in RES:
    if not r["ok"]: print("FAIL", r["name"], r["detail"][:200])
sys.exit(1 if nf else 0)
