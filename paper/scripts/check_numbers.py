"""Independent check of every number, text rule and reference in the combined D3-D4-D5-X1 paper (version 6).

  python -B scripts/check_numbers.py [--numbers PATH]

Shares no code with scripts/make_numbers.py, and does not import it or read checks/numbers.json.
  1. d3.*, d4.*, d5.*: each source paper's independent checker (D3 v2, D4 v3, D5 v2, file scripts/check_numbers.py,
     written without the source builders' code; each recomputes its values from per-point, per-item or per-position
     records) is run unchanged except for where it reads, on a mirror of its paper folder (build/check-d3/ etc.) in
     which numbers.tex and the figure data are replaced by this paper's values with the prefix removed. A single
     wrong value fails that checker. The source paper folders are only read. The key set must also equal the source
     paper's own key set, so that nothing is added or lost on the way.
  2. x1m.*: recomputed here from X1's raw run files (c01_*.json, c02_attn_*.json), and every pass or fail recomputed
     from the raw files by X1's own independent checker (X1-shared-parts/code/check_report.py, which does not use
     the code that wrote verdicts.json); the pass counts are then recounted here from the 22 test values with the
     thresholds of X1's PROTOCOL.md.
  3. dx.*: the totals, recounted from the checked per-study values.
  4. Text of main.tex: every \\nv key defined; no hand-typed decimals except thresholds found in a source paper or in
     X1's PROTOCOL.md; no product or company names; British spelling; no repository paths in captions; the author
     line; the companion preprint's title against the newest D1 paper version; and (versions 4 and 5) what the text claims
     about test counts, gate counts, head-group members and values outside their random ranges.
  5. References: every key cited and defined; every arXiv entry against the saved arXiv record (sources/arxiv-api.xml):
     identifier, title, first author and year (first version, or venue year); every non-arXiv entry against
     sources/web-records.json.
--numbers checks another copy of numbers.tex (for the planted-error test); nothing is then written.
Writes checks/NUMBER-CHECK.md and checks/number-check.json; exits 1 if anything fails.
"""
import io, json, re, shutil, subprocess, sys, contextlib
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
AREA = HERE.parent
ARGV = sys.argv[1:]
NUMFILE = Path(ARGV[ARGV.index("--numbers") + 1]).resolve() if "--numbers" in ARGV else HERE / "numbers.tex"
WRITE = NUMFILE == HERE / "numbers.tex"
RES = []


def check(name, ok, detail=""):
    RES.append(dict(name=name, ok=bool(ok), detail=str(detail)))


def macros(path):
    out = {}
    for ln in Path(path).read_text().splitlines():
        m = re.fullmatch(r"\\expandafter\\def\\csname nv@(.+?)\\endcsname\{(.*)\}", ln)
        if m: out[m.group(1)] = m.group(2)
    return out


NV = macros(NUMFILE)
check("numbers file parsed", len(NV) > 0, len(NV))

# ------------------------------------------------------------------ 1. source checkers on mirrors
SRC = {"d3": ("D3-indentation", "paper/v2-2026-11-09", "D3"),
       "d4": ("D4-ordinal-suffixes", "paper/v3-2026-11-09", "D4"),
       "d5": ("D5-capitals", "paper/v2-2026-11-08", "D5")}
for pre, (study, sub, var) in SRC.items():
    sdir = AREA / study; spaper = sdir / sub
    mine = {k[len(pre) + 1:]: v for k, v in NV.items() if k.startswith(pre + ".")}
    theirs = macros(spaper / "numbers.tex")
    check(f"{pre}: key set equals the source paper's", set(mine) == set(theirs),
          f"missing {sorted(set(theirs) - set(mine))[:5]}, extra {sorted(set(mine) - set(theirs))[:5]}")
    mirror = HERE / "build" / f"check-{pre}"
    if mirror.exists(): shutil.rmtree(mirror)
    shutil.copytree(spaper, mirror, ignore=shutil.ignore_patterns("build", "manuscript.pdf", "*.aux", "*.log"))
    with open(mirror / "numbers.tex", "w") as f:
        f.write("% mirror for checking: values of the combined paper, prefix removed\n")
        for k in sorted(mine): f.write("\\expandafter\\def\\csname nv@" + k + "\\endcsname{" + mine[k] + "}\n")
    for f in mirror.glob("fig-*.dat"):
        ours = HERE / f"fig-{pre}-{f.name[4:]}"
        check(f"{pre}: figure data {f.name} present in this paper", ours.exists())
        if ours.exists(): shutil.copyfile(ours, f)
    code = (spaper / "scripts" / "check_numbers.py").read_text()
    a = "PAPER = Path(__file__).resolve().parents[1]\n"; b = f"{var} = PAPER.parents[1]\n"
    ok_lines = code.count(a) == 1 and code.count(b) == 1
    check(f"{pre}: source checker located its two path lines", ok_lines)
    if not ok_lines: continue
    code = code.replace(a, "PAPER = Path(" + repr(str(mirror)) + ")\n").replace(b, var + " = Path(" + repr(str(sdir)) + ")\n")
    ns = {"__file__": str(spaper / "scripts" / "check_numbers.py"), "__name__": "dx_check_" + pre}
    saved_argv, saved_path = sys.argv, list(sys.path)
    sys.argv = [ns["__file__"]]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, ns["__file__"], "exec"), ns)
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv; sys.path[:] = saved_path
        for m in [m for m in sys.modules if m in ("r2_independent",)]: del sys.modules[m]
    sres = ns.get("RES", [])
    bad = [r for r in sres if not r["ok"]]
    check(f"{pre}: source paper's independent checker on this paper's values ({len(sres)} checks)", len(sres) > 0 and not bad,
          "; ".join(f"{r['name']}: {r['detail'][:150]}" for r in bad[:6]))

# ------------------------------------------------------------------ 2. X1
XC = AREA / "X1-shared-parts" / "outputs" / "conf"
raw = {}
for tag, fn in (("LB", "c01_qwen3-06b-bf16_LB.json"), ("IND", "c01_qwen3-06b-bf16_IND.json"),
                ("ORD", "c01_qwen3-06b-bf16_ORD.json"), ("CAPS", "c01_qwen3-17b_CAPS.json")):
    with open(XC / fn) as fh: raw[tag] = json.load(fh)
FIELD = {"LBw": ("LB", "width_effect", True), "LBs": ("LB", "sep", True), "cU": ("IND", "code|dedent|lq", False),
         "cT": ("IND", "code|term|lq", False), "gU": ("IND", "gen|dedent|lq", False), "gT": ("IND", "gen|term|lq", False),
         "t4": ("ORD", "d4|teen_effect", True), "t5": ("ORD", "d5|teen_effect", True), "p4": ("ORD", "d4|pull", True),
         "p5": ("ORD", "d5|pull", True), "Du": ("CAPS", "D_up", True), "Dl": ("CAPS", "D_low", True), "wl": ("CAPS", "word_lp", False)}


def fmt(x, nd=3):
    s = ("{:." + str(nd) + "f}").format(x)
    if float(s) == 0: s = s.lstrip("-")
    return s.replace("-", "\u2212")


seen = set()
for key in [k for k in NV if k.startswith("x1m.m.")]:
    parts = key.split(".")  # x1m.m.GROUP.CODE[.rlo|.rhi]
    grp, mcode = parts[2], parts[3]
    tag, field, ratio = FIELD[mcode]
    rname = grp.replace("noL17H6", "-L17H6")
    run = raw[tag]
    f = (lambda r: r[field] / run["base"][field]) if ratio else (lambda r: r[field] - run["base"][field])
    if len(parts) == 4:
        want = fmt(f(run[rname]))
    else:
        draws = [f(run[f"{rname}~r{i}"]) for i in range(20)]
        want = fmt(min(draws) if parts[4] == "rlo" else max(draws))
    check(f"x1m: {key}", NV[key] == want, f"{NV[key]} vs {want}")
    seen.add(key)
# version 4, post hoc: position of every named value against the range of its 20 random groups (unrounded values)
POS = {}
for tag, run in raw.items():
    for g in run:
        if g == "base" or "~" in g: continue
        for mcode, (t, field, ratio) in FIELD.items():
            if t != tag: continue
            b0 = run["base"][field]
            one = (lambda rr: rr[field] / b0) if ratio else (lambda rr: rr[field] - b0)
            v = one(run[g]); dr = sorted(one(run[f"{g}~r{i}"]) for i in range(20))
            pos = "below" if v < dr[0] else ("above" if v > dr[-1] else "in")
            gk = g.replace("-L17H6", "noL17H6"); POS[(gk, mcode)] = pos
            k = f"x1m.o.{gk}.{mcode}"
            check(f"x1m: {k}", NV.get(k) == pos, f"{NV.get(k)} vs {pos}"); seen.add(k)
# the criterion frozen for L17H6's shared role (PROTOCOL.md, A3), applied post hoc to each single head on indentation
ind = raw["IND"]; gt0 = ind["base"]["gen|term|lq"]
for h in ("L17H6", "L11H9", "L0H2", "L0H3"):
    ch = ind[h]["gen|term|lq"] - gt0
    lowest = min(ind[f"{h}~r{i}"]["gen|term|lq"] - gt0 for i in range(20))
    want = "met" if (ch <= -0.010 and ch < lowest) else "not met"
    k = f"x1m.ph.A3rule.{h}"
    check(f"x1m: {k}", NV.get(k) == want, f"{NV.get(k)} vs {want}"); seen.add(k)
# every named group of every task is present
for tag, run in raw.items():
    for g in run:
        if g == "base" or "~" in g: continue
        for mcode, (t, _, _) in FIELD.items():
            if t == tag:
                check(f"x1m: matrix cell {g}|{mcode} present", f"x1m.m.{g.replace('-L17H6', 'noL17H6')}.{mcode}" in NV)
lb = raw["LB"]["base"]
for k, want in (("x1m.base.LBw", fmt(lb["width_effect"])), ("x1m.base.LBs", fmt(lb["sep"])),
                ("x1m.n.LB.width", f"{lb['n_width']:,}"), ("x1m.n.LB.par", f"{lb['n_nat']:,}"),
                ("x1m.n.IND.cU", f"{raw['IND']['base']['code|dedent|n']:,}"), ("x1m.n.IND.cT", f"{raw['IND']['base']['code|term|n']:,}"),
                ("x1m.n.IND.gU", f"{raw['IND']['base']['gen|dedent|n']:,}"), ("x1m.n.IND.gT", f"{raw['IND']['base']['gen|term|n']:,}"),
                ("x1m.n.ORD.teen4", str(raw["ORD"]["base"]["d4|n_teen"])), ("x1m.n.ORD.non4", str(raw["ORD"]["base"]["d4|n_end123"])),
                ("x1m.n.CAPS.pos", f"{raw['CAPS']['base']['n_pos']:,}"), ("x1m.n.CAPS.par", f"{raw['CAPS']['base']['n_par']:,}")):
    check(f"x1m: {k}", NV.get(k) == want, f"{NV.get(k)} vs {want}"); seen.add(k)
cap = raw["CAPS"]; Du = cap["L0H3"]["D_up"] / cap["base"]["D_up"]; Dl = cap["L0H3"]["D_low"] / cap["base"]["D_low"]
for k, want in (("x1m.C7.lossDl", fmt(1 - Dl)), ("x1m.C7.threeLossDu", fmt(3 * (1 - Du)))):
    check(f"x1m: {k}", NV.get(k) == want, f"{NV.get(k)} vs {want}"); seen.add(k)
with open(XC.parent / "gate_qwen3-06b.json") as fh: g06 = json.load(fh)
want = fmt(g06["decode"]["upper_matches_lower"])  # the gate's same-word share, from the decode record
check("x1m: x1m.gate06b.c2 (and below the 0.60 bar)", NV.get("x1m.gate06b.c2") == want and g06["decode"]["upper_matches_lower"] < 0.60, f"{NV.get('x1m.gate06b.c2')} vs {want}")
seen.add("x1m.gate06b.c2")
with open(XC / "c02_attn_qwen3-06b-bf16.json") as fh: att = json.load(fh)
for task in ("LB", "IND", "ORD", "CAPS"):
    for head, slot, nm in (("L0H3", "0", "self"), ("L0H2", "1", "prev"), ("L11H9", "0", "self")):
        k = f"x1m.att.{head}.{nm}.{task}"
        want = fmt(att[task][head]["dec"][slot], 2)
        check(f"x1m: {k}", NV.get(k) == want, f"{NV.get(k)} vs {want}"); seen.add(k)
for head in ("L0H3", "L11H9"):
    vals = [att[t][head]["dec"]["0"] for t in att]
    for k, want in ((f"x1m.att.{head}.self.min", fmt(min(vals), 2)), (f"x1m.att.{head}.self.max", fmt(max(vals), 2))):
        check(f"x1m: {k}", NV.get(k) == want, f"{NV.get(k)} vs {want}"); seen.add(k)
# pass or fail: X1's own independent checker, then a recount with the protocol thresholds
p = subprocess.run([sys.executable, "-B", str(AREA / "X1-shared-parts" / "code" / "check_report.py")], capture_output=True, text=True)
check("x1m: X1's independent report check passes (recomputes all 22 pass/fail from raw files)", p.returncode == 0, (p.stdout + p.stderr)[-300:])
r = lambda tag, g, code: (FIELD[code][2] and raw[tag][g][FIELD[code][1]] / raw[tag]["base"][FIELD[code][1]]) or (not FIELD[code][2] and raw[tag][g][FIELD[code][1]] - raw[tag]["base"][FIELD[code][1]])
def rnd(tag, g, code): return [r(tag, f"{g}~r{i}", code) for i in range(20)]
def quiet(g):  # PROTOCOL.md: |change| <= 0.03 on code unmarked, code terminal, generated terminal; <= 0.10 generated unmarked
    return all(abs(r("IND", g, c)) <= 0.03 for c in ("cU", "cT", "gT")) and abs(r("IND", g, "gU")) <= 0.10
L17 = "L17H6"
T = {
 "A1": r("LB", L17, "LBw") <= 0.85 and r("LB", L17, "LBw") < min(rnd("LB", L17, "LBw")),
 "A2": r("LB", L17, "LBs") <= 0.95 and r("LB", L17, "LBs") < min(rnd("LB", L17, "LBs")),
 "A3": r("IND", L17, "gT") <= -0.010 and r("IND", L17, "gT") < min(rnd("IND", L17, "gT")),
 "A4": r("LB", "CARRY-L17H6", "LBw") >= 0.92 and r("LB", "TERM-L17H6", "LBw") >= 0.92 and r("LB", "CARRY", "LBw") <= 0.85 and r("LB", "TERM", "LBw") <= 0.85,
 "A5": abs(r("IND", L17, "cU")) <= 0.03 and abs(r("IND", L17, "gU")) <= 0.10 and all(0.90 <= r("ORD", L17, c) <= 1.10 for c in ("t4", "t5")),
 "A6": r("CAPS", L17, "Du") <= 0.97 and r("CAPS", L17, "Du") < min(rnd("CAPS", L17, "Du")) and abs(r("CAPS", L17, "wl")) <= 0.03,
 "B1": r("LB", "WRITE", "LBw") <= 0.70, "B2": r("LB", "COUNT", "LBs") <= 0.50, "B3": r("LB", "READ-L17H6", "LBw") <= 0.75,
 "B4": r("IND", "DED", "cU") <= -0.40 and r("IND", "DED", "cU") < min(rnd("IND", "DED", "cU")) and all(0.95 <= r("LB", "DED", c) <= 1.05 for c in ("LBw", "LBs")),
 "C1": all(r("ORD", "L0H2", c) <= 0.75 and r("ORD", "L0H2", c) < min(rnd("ORD", "L0H2", c)) for c in ("t4", "t5")),
 "C2": r("LB", "L0H2", "LBs") <= 0.93 and r("LB", "L0H2", "LBs") < min(rnd("LB", "L0H2", "LBs")),
 "C4": all(r("ORD", "L0H3", c) <= 0.50 and r("ORD", "L0H3", c) < min(rnd("ORD", "L0H3", c)) for c in ("p4", "p5")) and all(r("ORD", "L0H3", c) >= 0.90 for c in ("t4", "t5")),
 "C5": r("LB", "L0H3", "LBs") <= 0.50 and r("LB", "L0H3", "LBs") < min(rnd("LB", "L0H3", "LBs")),
 "C7": r("CAPS", "L0H3", "Dl") <= 0.60 and r("CAPS", "L0H3", "Dl") < min(rnd("CAPS", "L0H3", "Dl")) and r("CAPS", "L0H3", "wl") <= -0.50 and r("CAPS", "L0H3", "wl") < min(rnd("CAPS", "L0H3", "wl")) and (1 - r("CAPS", "L0H3", "Dl")) >= 3 * (1 - r("CAPS", "L0H3", "Du")),
 "C8": all(att[t]["L0H3"]["dec"]["0"] >= 0.90 for t in att) and all(att[t]["L0H2"]["dec"]["1"] >= 0.50 for t in ("LB", "ORD", "CAPS")),
 "D4": all(att[t]["L11H9"]["dec"]["0"] >= 0.85 for t in att),
 "D1": r("ORD", "L11H9", "t4") <= 0.85 and r("ORD", "L11H9", "t5") <= 0.90 and all(r("ORD", "L11H9", c) <= 0.85 for c in ("p4", "p5")) and all(r("ORD", "L11H9", c) < min(rnd("ORD", "L11H9", c)) for c in ("t4", "t5", "p4", "p5")),
 "D2": all(r("LB", "L11H9", c) >= 0.93 for c in ("LBw", "LBs")),
}
for t, g in (("C3", "L0H2"), ("C6", "L0H3")):
    T[t] = quiet(g)
T["D3"] = all(abs(r("IND", "L11H9", c)) <= 0.05 for c in ("cU", "cT", "gU", "gT"))
for t, g in (("B1", "WRITE"), ("B2", "COUNT"), ("B3", "READ-L17H6")):
    T[t] = T[t] and quiet(g)
proto = (AREA / "X1-shared-parts" / "PROTOCOL.md").read_text()
for t, ok in T.items():
    check(f"x1m: test {t} recomputed from raw files with the protocol threshold", ok == (NV.get(f"x1m.v.{t}") == "pass"), f"{ok} vs {NV.get(f'x1m.v.{t}')}")
    check(f"x1m: test {t} named in PROTOCOL.md", re.search(r"\b" + t + r"\b", proto) is not None)
    seen.add(f"x1m.v.{t}")
check("x1m: 22 tests recounted", len(T) == 22 and NV.get("x1m.ntests") == "22", len(T))
check("x1m: passes recounted", NV.get("x1m.npass") == str(sum(T.values())), f"{NV.get('x1m.npass')} vs {sum(T.values())}")
seen |= {"x1m.ntests", "x1m.npass"}
core = ("A1", "A4", "B1", "B4", "C1", "C2", "C4", "C5", "D1", "D2")  # PROTOCOL.md, "Whole account"
check("x1m: account verdict", NV.get("x1m.account") == ("supported" if all(T[c] for c in core) else "partly supported")); seen.add("x1m.account")
left = [k for k in NV if k.startswith("x1m.") and k not in seen]
check("x1m: every x1m value checked", not left, left[:8])

# ------------------------------------------------------------------ 3. totals
def num(k): return int(NV[k].replace(",", ""))
tt = sum(num(k) for k in ("d3.c.ntests", "d3.c2.ntests", "d4.c.ntests", "d4.r2.ntests", "d5.c.ntests", "d5.r2.ntests", "x1m.ntests"))
pp = sum(num(k) for k in ("d3.c.npass", "d3.c2.npass", "d4.c.npass", "d4.r2.npass", "d5.c.npass", "d5.r2.npass", "x1m.npass"))
check("dx.ntests", num("dx.ntests") == tt, tt); check("dx.npass", num("dx.npass") == pp, pp); check("dx.nfail", num("dx.nfail") == tt - pp, tt - pp)
check("no other prefixes", all(k.split(".")[0] in ("d3", "d4", "d5", "x1m", "dx", "r3") for k in NV))

# models in the ordinal study beyond Qwen3-0.6B-Base and OLMo-2: gate records that qualify, plus round-2 further models
d4o = AREA / "D4-ordinal-suffixes" / "outputs"
names = set()
for f in sorted(d4o.glob("gate_summary_*.json")):
    with open(f) as fh: g = json.load(fh)["_gate"]
    if g["qualifies"]: names.add(f.stem.replace("gate_summary_", ""))
names |= {f.stem.replace("other_", "") for f in (d4o / "r2conf").glob("other_*.json")}
names -= {"qwen3-06b", "olmo2-1b"}
spell = dict(enumerate("zero one two three four five six seven eight nine ten".split()))
check("dx.ord.nfurther recounted from the ordinal study's records", NV.get("dx.ord.nfurther") == str(len(names)), f"{NV.get('dx.ord.nfurther')} vs {sorted(names)}")
check("dx.ord.nfurther.word", NV.get("dx.ord.nfurther.word") == spell[len(names)])
check("dx.d4r2.nfail", num("dx.d4r2.nfail") == num("d4.r2.ntests") - num("d4.r2.npass"))

# ------------------------------------------------------------------ 3b. version 4: units, gate counts, model size
ev1 = json.loads((d4o / "conf" / "EVALUATION.json").read_text())["tests"]
n_pred = len(ev1); n_res = sum(1 if "passed" in t else len(t) for t in ev1.values())
check("dx.d4r1.npred: numbered round-1 predictions in the ordinal study's record", NV.get("dx.d4r1.npred") == str(n_pred), n_pred)
check("d4.c.ntests equals the predictions with the per-model one read in each model", num("d4.c.ntests") == n_res, n_res)
check("dx.d4r1.B5.nmodels.word", NV.get("dx.d4r1.B5.nmodels.word") == spell[n_res - n_pred + 1])
rows1 = re.findall(r"(?m)^([BQO]\d+) &", (HERE / "tables" / "d4-tests.tex").read_text())
check("the round-1 ordinal table lists exactly the numbered predictions", sorted(rows1) == sorted(ev1), rows1)
p1 = (AREA / "D4-ordinal-suffixes" / "PROTOCOL.md").read_text()
unnamed = [k for k in ev1 if not re.search(r"\*\*" + k + r"\b", p1)]
check("every round-1 ordinal prediction is named in its frozen protocol", not unnamed, unnamed)
ev2 = json.loads((d4o / "r2conf" / "EVALUATION-2.json").read_text())["tests"]
bad2 = [k for k, t in ev2.items() if t["passed"] is False]
check("dx.d4r2.nfail.word", NV.get("dx.d4r2.nfail.word") == spell[len(bad2)] and len(bad2) == num("dx.d4r2.nfail"), bad2)
check("dx.d4r2.nfailpred.word: distinct failed predictions", NV.get("dx.d4r2.nfailpred.word") == spell[len({k.rsplit("@", 1)[0] for k in bad2})], bad2)
check("dx.d4r2.nfaillen.word: number lengths of the failed tests", NV.get("dx.d4r2.nfaillen.word") == spell[len({k.rsplit("@", 1)[1] for k in bad2})], bad2)
check("the failed round-2 tests are the previous-token set in Qwen3-1.7B-Base", all(k.startswith("X:qwen3-17b:PREV@") for k in bad2), bad2)
gates1 = []
for f in sorted(d4o.glob("gate_summary_*.json")):
    with open(f) as fh: gates1.append(bool(json.load(fh)["_gate"]["qualifies"]))
with open(d4o / "r2" / "e04_qwen3-17b.json") as fh: q17 = bool(json.load(fh)["gate"]["qualifies"])
check("d4.gate.n.*: round-1 ordinal gate recounted", num("d4.gate.n.models") == len(gates1) and num("d4.gate.n.qualify") == sum(gates1), (len(gates1), sum(gates1)))
check("dx.ordgate.q17", NV.get("dx.ordgate.q17") == ("passed" if q17 else "failed"))
check("dx.ordgate.ngated / npass", num("dx.ordgate.ngated") == len(gates1) + 1 and num("dx.ordgate.npass") == sum(gates1) + q17, (len(gates1) + 1, sum(gates1) + q17))
check("the ordinal study's second charter requires the gate of Qwen3-1.7B-Base", "Qwen3-1.7B-Base was not gated in round 1" in (AREA / "D4-ordinal-suffixes" / "CHARTER-2.md").read_text())
capg = {}
for f in sorted((AREA / "D5-capitals" / "outputs").glob("gate_*.json")):
    with open(f) as fh: capg[f.stem] = bool(json.load(fh)["gate"]["pass"])
capg["x1:qwen3-06b"] = bool(g06["gate"]["pass"])
check("dx.capsgate.ngated.word / npass.word", NV.get("dx.capsgate.ngated.word") == spell[len(capg)] and NV.get("dx.capsgate.npass.word") == spell[sum(capg.values())], capg)
ch5 = re.search(r"Charter models \(all on disk\): (.*?)\. ", (AREA / "D5-capitals" / "CHARTER.md").read_text()).group(1)
check("dx.capsgate.ncharter.word: the capitals charter's candidate models", NV.get("dx.capsgate.ncharter.word") == spell[len(ch5.split(","))], ch5)
card = (HERE / "sources" / "gemma-4-E2B-model-card.txt").read_text()
row = [ln for ln in card.splitlines() if ln.startswith("| **Total Parameters**")][0].split("|")[2]
eff, tot = re.findall(r"([\d.]+)B", row)
check("dx.gemma.effB / totB against the saved model-card excerpt", NV.get("dx.gemma.effB") == eff and NV.get("dx.gemma.totB") == tot, row)
ATOM = "{http://www.w3.org/2005/Atom}"
for e in ET.parse(HERE / "sources" / "arxiv-api.xml").getroot().findall(ATOM + "entry"):
    if "2607.02770" in e.find(ATOM + "id").text:
        check("dx.gemma.effB is the low end of the range in the cited report's arXiv record", f"ranging from {eff}B to" in " ".join(e.find(ATOM + "summary").text.split()))
import glob as _glob, os as _os, struct as _struct
_w = _glob.glob(_os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B/snapshots/*/model.safetensors"))
if _w:  # where the weights are on disk: count them from the file header
    with open(_w[0], "rb") as fh:
        hdr = json.loads(fh.read(_struct.unpack("<Q", fh.read(8))[0]))
    npar = 0
    for k, v in hdr.items():
        if k == "__metadata__": continue
        n1 = 1
        for d in v["shape"]: n1 *= d
        npar += n1
    check("dx.gemma.totB against the weights on disk", f"{npar / 1e9:.1f}" == NV.get("dx.gemma.totB"), npar)
DXKEYS = ("dx.ntests", "dx.npass", "dx.nfail", "dx.ord.nfurther", "dx.ord.nfurther.word", "dx.d4r2.nfail", "dx.d4r2.nfail.word", "dx.d4r2.nfailpred.word",
          "dx.d4r2.nfaillen.word", "dx.d4r1.npred", "dx.d4r1.B5.nmodels.word", "dx.ordgate.q17", "dx.ordgate.ngated", "dx.ordgate.npass",
          "dx.capsgate.ngated.word", "dx.capsgate.npass.word", "dx.capsgate.ncharter.word", "dx.gemma.effB", "dx.gemma.totB")
check("every dx value checked", all(k in DXKEYS for k in NV if k.startswith("dx.")), [k for k in NV if k.startswith("dx.") and k not in DXKEYS])

# ------------------------------------------------------------------ 3c. version 5: r3.*, the indentation study's third round
# Recomputed from the raw run records (c01/c02 run files, r2c01/r2c02 per-point and per-program records, the r2c03 run
# file, X1's c01 run file and the round-3 group and pair run files, the gate's per-point records, the material files)
# by the round-3 independent checker's readers (D3-indentation/round3/code/r3_check_independent.py, checker code that
# does not use the verdict scripts or their outputs). Formats are written here: 3 decimals, U+2212, changes signed.
import importlib.util as _ilu
_R3 = AREA / "D3-indentation" / "round3"
_sp = _ilu.spec_from_file_location("r3ci", _R3 / "code" / "r3_check_independent.py"); R3C = _ilu.module_from_spec(_sp); _sp.loader.exec_module(R3C)
def _f3(x): t = f"{x:.3f}"; t = "0.000" if t == "-0.000" else t; return "−" + t[1:] if t.startswith("-") else t
def _s3(x): t = _f3(x); return t if (t.startswith("−") or t == "0.000") else "+" + t
def _s2(x): t = f"{x:.2f}"; return "−" + t[1:] if t.startswith("-") else "+" + t
def eq(k, want): check(f"{k} (recomputed from raw records)", NV.get(k) == want, f"{NV.get(k)} vs {want}")
_D3O = AREA / "D3-indentation" / "outputs"
# material
_fet = [json.loads(l) for l in open(_R3 / "data" / "code" / "fetched.jsonl")]
_ch = [json.loads(l) for l in open(_R3 / "data" / "code" / "part-held.jsonl")]
eq("r3.mat.nfiles", f"{len(_fet):,}"); eq("r3.mat.nrepos", f"{len({r['repo'] for r in _fet}):,}")
eq("r3.mat.nchunks", f"{len(_ch):,}"); eq("r3.mat.nchunkrepos", f"{len({c['repo'] for c in _ch}):,}")
eq("r3.mat.ngen", f"{sum(1 for _ in open(_R3 / 'data' / 'gen' / 'part-held.jsonl')):,}")
eq("r3.mat.ngen2", f"{sum(1 for _ in open(_R3 / 'data' / 'gen2' / 'part-held.jsonl')):,}")
check("r3: every fetched file has a permissive licence and its repository's licence text ships beside it",
      all(r["license"] in ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "Unlicense") for r in _fet)
      and all((_R3 / "data" / "code" / "licences" / (r["repo"].replace("/", "__") + ".LICENSE.txt")).exists() for r in _fet))
# R1
for tag, d in (("r3.r1", _R3 / "outputs" / "conf"), ("r3.o1", _D3O / "conf")):
    V, T = R3C.r1(d)
    for i in range(1, 10):
        h = f"H{i}"; v = V[{"H3": "H3.open", "H9": "H9.nec"}.get(h, h)]
        eq(f"{tag}.{h}", _f3(v) if h in ("H3", "H9") else _s3(v))
        if tag == "r3.r1": eq(f"{tag}.v.{h}", "pass" if T[h] else "fail")
    if tag == "r3.r1":
        eq("r3.r1.npass", str(sum(T[f"H{i}"] for i in range(1, 10)))); eq("r3.r1.ntests", "9")
        eq("r3.r1.H1.r", _s3(min(V["H1.r"]))); eq("r3.r1.H5.r", _s3(min(V["H5.r"]))); eq("r3.r1.H8.r", _s3(min(V["H8.r"])))
        eq("r3.r1.H9.suf", _f3(V["H9.suf"])); eq("r3.r1.H9.rnec", _f3(max(V["H9.rnec"])))
        check("r3: round 1's frozen verdict recomputed", T["verdict"] == "confirmed", T["verdict"])
# R2
for tag, d in (("r3.r2", _R3 / "outputs" / "r2" / "conf"), ("r3.o2", _D3O / "r2" / "conf")):
    B, T, Qv = R3C.r2(d)
    for P, h, hg in (("Q", B["q"], B["qg"]), ("O", B["o"], B["og"])):
        for i, (src, k, fm) in {"1": ("n", "att_inner_ded", _f3), "2": ("n", "ko_inner_ded", _s3), "3": ("n", "ko_inner_same", _s3),
                                "4": ("n", "ko_closed_ded", _s3), "5": ("n", "pick_not_inner", _f3), "6": ("n", "steer_dedent_head_dlq", _s2),
                                "7": ("n", "ko_true_ded", _s3), "8": ("g", "steer_term_head_dlq", _s2), "9": ("g", "share_blk_kw_return", _f3),
                                "10": ("g", "share_layer_return_max", _f3)}.items():
            eq(f"{tag}.{P}{i}", fm((h if src == "n" else hg)[k]))
    eq(f"{tag}.Q11", _s3(B["qg"]["l0h2_line_dqcur_term"]))
    for i, k in (("1", "d_lq_dedent"), ("2", "d_lq_term"), ("3", "d_lq_same"), ("4", "d_psame_dedent")): eq(f"{tag}.G{i}", _s3(B["g"][k]))
    if tag == "r3.r2":
        eq("r3.r2.G1.rlo", _s3(B["g"]["rand_lo"])); eq("r3.r2.G1.rhi", _s3(B["g"]["rand_hi"])); eq("r3.r2.q.others_maxabs", _f3(B["q"]["others_maxabs"]))
        for k, v in T.items(): eq(f"r3.r2.v.{k}", "pass" if v else "fail")
        eq("r3.r2.npass", str(sum(T.values()))); eq("r3.r2.ntests", str(len(T)))
        eq("r3.r2.nqconf", str(sum(v == "confirmed" for v in Qv.values()))); eq("r3.r2.nq", str(len(Qv)))
        eq("r3.r2.nreturn", f"{B['qg']['n.return']:,}")
# X and E
_I = json.loads((_R3 / "x1" / "outputs" / "conf" / "c01_qwen3-06b-bf16_IND.json").read_text())
_G = json.loads((_R3 / "x1" / "outputs" / "conf" / "r3x_l11h9_IND.json").read_text())
_P = json.loads((_R3 / "outputs" / "conf" / "r3_pairs.json").read_text())
CX, TX = R3C.xpart(_I); CE, TE, XE_ = R3C.epart(_I, _G, _P)
for k in CX: eq(f"r3.v.{k}", "pass" if TX[k] else "fail")
for k in ("E1", "E2", "E3", "E4", "E5", "E6"): eq(f"r3.v.{k}", "pass" if TE[k] else "fail")
eq("r3.x.npass", str(sum(TX.values()))); eq("r3.x.ntests", str(len(TX)))
eq("r3.e.npass", str(sum(TE[k] for k in ("E1", "E2", "E3", "E4", "E5", "E6")))); eq("r3.e.ntests", "6"); eq("r3.e.verdict", TE["verdict"])
for k, (a, lo, hi) in (("XA3", CX["XA3"]), ("XB4", CX["XB4"])): eq(f"r3.x.{k}", _s3(a)); eq(f"r3.x.{k}.rlo", _s3(lo)); eq(f"r3.x.{k}.rhi", _s3(hi))
eq("r3.x.XD3.max", _f3(max(abs(v) for v in CX["XD3"])))
eq("r3.e.E1", _s3(CE["E1"][0])); eq("r3.e.E1.rlo", _s3(CE["E1"][1])); eq("r3.e.E1.rhi", _s3(CE["E1"][2]))
e2 = CE["E2"]; e3 = CE["E3"]
for g, o2, o3 in (("CARRY", 0, 0), ("TERM", 3, 3)):
    eq(f"r3.e.E2.{g}", _s3(e2[o2])); eq(f"r3.e.E2.{g}.rlo", _s3(e2[o2 + 1])); eq(f"r3.e.E2.{g}.rhi", _s3(e2[o2 + 2]))
    eq(f"r3.e.E3.{g}.without", _s3(e3[o3])); eq(f"r3.e.E3.{g}.diff", _s3(e3[o3 + 2]))
eq("r3.e.E4.cU", _s3(CE["E4"][0])); eq("r3.e.E4.gU", _s3(CE["E4"][1]))
eq("r3.e.E5", _s3(CE["E5"][0])); eq("r3.e.E5.rlo", _s3(CE["E5"][1])); eq("r3.e.E5.rhi", _s3(CE["E5"][2]))
e6 = CE["E6"]; eq("r3.e.E6.nec", _f3(e6[0])); eq("r3.e.E6.rnec", _f3(e6[1])); eq("r3.e.E6.L17H6.nec", _f3(e6[4])); eq("r3.e.E6.L17H6.rnec", _f3(e6[5]))
check("r3 claim: L17H6's pair necessity is above 19 of its 20 random layer-17 heads, and the one above it is L17H4",
      XE_["L17H6.above"] == 19 and XE_["L17H6.ndraws"] == 20 and XE_["L17H6.topdraw.heads"] == ["L17H4"], XE_)
check("r3 claim: L11H9's change at generated terminal dedents is larger than L17H6's (round 3)", CE["E1"][0] < CX["XA3"][0] < 0)
# gate
_GT = {m: R3C.gate(_R3 / "outputs", m) for m in ("qwen3-06b", "olmo1b", "pythia-410m", "gemma4-e2b", "gpt2")}
eq("r3.gate.p410.head", _f3(_GT["pythia-410m"]["head"]))
eq("r3.gate.npassall", str(sum(all(v["items"].values()) for v in _GT.values()))); eq("r3.gate.nmodels", str(len(_GT)))
check("r3 claim: Pythia-410M misses only the header item and GPT-2 fails every item",
      [k for k, v in _GT["pythia-410m"]["items"].items() if not v] == ["item1"] and not any(_GT["gpt2"]["items"].values()))
for k, parts in (("r3.ntests", ("r3.r1.ntests", "r3.r2.ntests", "r3.x.ntests", "r3.e.ntests")), ("r3.npass", ("r3.r1.npass", "r3.r2.npass", "r3.x.npass", "r3.e.npass")),
                 ("r3.ind.ntests", ("r3.r1.ntests", "r3.r2.ntests")), ("r3.ind.npass", ("r3.r1.npass", "r3.r2.npass")),
                 ("r3.xe.ntests", ("r3.x.ntests", "r3.e.ntests")), ("r3.xe.npass", ("r3.x.npass", "r3.e.npass"))):
    eq(k, str(sum(int(NV[p]) for p in parts)))
_p3 = (AREA / "D3-indentation" / "PROTOCOL-3.md").read_text()
check("r3: PROTOCOL-3.md states E1 as X1's criterion and the E verdict rule",
      "≤ −0.010 **and below every one of its 20 random draws**" in _p3 and "confirmed** if E1 and E3 pass" in _p3)
_t0 = (AREA / "D3-indentation" / "PROTOCOL-3.md").stat().st_mtime
_files = [_R3 / "outputs" / "conf" / f for f in ("c01_newcode.json", "c01_gen.json", "c02_gen.json", "r3_pairs.json")] + \
         [_R3 / "outputs" / "r2" / "conf" / f for f in ("r2c01_qwen3-06b-bf16_L23H2.jsonl", "r2c01_olmo1b_L13H1.jsonl", "r2c02_qwen3-06b-bf16_L23H2.jsonl", "r2c02_olmo1b_L13H1.jsonl", "r2c03_olmo1b_DEDO.json")] + \
         [_R3 / "x1" / "outputs" / "conf" / f for f in ("c01_qwen3-06b-bf16_IND.json", "r3x_l11h9_IND.json")]
check("r3: every round-3 run file read was written after PROTOCOL-3.md was frozen", all(f.stat().st_mtime > _t0 for f in _files))

# ------------------------------------------------------------------ 4. text
tex = (HERE / "main.tex").read_text()
body = tex.split("\\begin{document}", 1)[1]
used = set(re.findall(r"\\nv\{([^{}#]+)\}", body))
for pre_macro, pre_key in re.findall(r"\\newcommand\{\\(\w+)\}\[1\]\{\\nv\{([\w.]+?)#1\}\}", tex):
    for arg in re.findall(r"\\" + pre_macro + r"\{([^{}]+)\}", body):
        used.add(pre_key + arg)
miss = sorted(k for k in used if k not in NV)
check(f"text: every \\nv key used is defined ({len(used)} keys)", not miss, miss[:10])
srcs = "".join((AREA / s / p / "main.tex").read_text() for s, p, _ in SRC.values()) + proto
plain = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", body, flags=re.S)  # drawing code: layout sizes only
plain = re.sub(r"\\(begin|end)\{[^{}]*\}(\[[^\]]*\])?(\{[^{}]*\})?", "", plain)
plain = re.sub(r"\\nv\{[^{}]*\}", "", plain)
plain = re.sub(r"\\(?:cite[tp]?|ref|cref|Cref|label|hd|includegraphics|url|href)\{[^{}]*\}", "", plain)
plain = re.sub(r"\{[\w.\-]+\.dat\}", "", plain)
plain = re.sub(r"(?m)%.*$", "", plain)
decs = sorted(set(re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", plain)))
okdec = {"4.6", "4.1", "4.2", "5.2", "0.72", "0.62", "0.56", "0.49", "1.2", "1.25", "0.35", "0.55", "0.95"}  # figure and layout sizes
bad = [d for d in decs if d not in srcs and d not in okdec]
check("text: no hand-typed decimals except thresholds found in a source paper or X1's protocol", not bad, bad)
names = re.findall(r"\b(Claude|Codex|Anthropic|OpenAI|ChatGPT|GPT-4|Gemini)\b", re.sub(r"\\bibliography\{[^}]*\}", "", body))
check("text: no product or company names in the body", not names, names)
amer = re.findall(r"\b(behavior|color|analyze|analyzed|modeling|labeled|generalize|organize|optimize|recognize|center|centered|favor|favors|neighbor|neighboring|summarize|tokenizer|tokenization|normalize|normalized|regularized)\b", re.sub(r"\\cite[tp]?\{[^}]*\}", "", plain))
check("text: British spelling", not amer, sorted(set(amer)))
caps = re.findall(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body)
check("text: no repository paths in captions", not any(re.search(r"(\w+/){2,}|\.py\b|\.json\b|outputs/|code/", c) for c in caps))
check("text: author line", "\\author{Richard Jeevash\\\\ Independent researcher\\\\ \\texttt{jeevash@richardjeevash.com}\\\\ ORCID \\href{https://orcid.org/0009-0005-7812-0939}{0009-0005-7812-0939}}" in tex)
# the companion preprint's title: the \title of the newest D1 paper version present (the study area), else the title fixed here
D1_TITLE = "Counting by dilution and a width stored at the newline: a partial account of line-breaking in Pythia-410M"
_d1v = sorted((AREA / "D1-line-breaking" / "paper").glob("v*/main.tex"), key=lambda q: int(re.match(r"v(\d+)", q.parent.name).group(1)))
_d1t = re.search(r"\\title\{(.*?)\}\n", _d1v[-1].read_text()).group(1).replace("\\\\", " ") if _d1v else D1_TITLE
check(f"text: companion title fixed here equals the newest D1 version's ({_d1v[-1].parent.name if _d1v else 'folder absent'})", _d1t == D1_TITLE, _d1t)
_cmp = re.search(r"@misc\{jeevash2026linebreaking,\s*title\s*=\s*\{(.*)\},\n", (HERE / "refs.bib").read_text()).group(1).replace("{Pythia-410M}", "Pythia-410M")
check("text: companion preprint cited by the newest D1 version's title", _cmp == _d1t, f"{_cmp} | {_d1t}")
# ---- version 4: what the text claims about the cross-study matrix, the head groups and the counts
abstract = tex.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
flat = " ".join(body.split())
for phrase in ("only one late head", "indentation alone", "It is the only way the indentation sets touch line-breaking", "nothing else beyond the general early movers",
               "under two billion parameters", "read only by the MLPs after it", "Unmarked dedents rest on one late head", "A prospective cross-study test in \\qsix{} finds",
               "an earlier draft", "smoke test", "pre-fix", "overnight stall", "to free the GPU", "during training"):
    check(f"text: withdrawn wording absent: {phrase!r}", phrase not in flat and all(phrase not in f.read_text() for f in (HERE / "tables").glob("*.tex")))
check("text (v5): abstract names both heads and says the second was confirmed by a frozen test on new material",
      "\\hd{17}{6}" in abstract and "\\hd{11}{9}" in abstract and "confirmed by a frozen test on new material" in abstract and "first noticed after the cross-study test" in abstract)
for phrase in ("read post hoc by the same criterion, a second", "on a post hoc reading", "has not been tested on new material",
               "were not run on the held-out indentation material", "We know of no", "densely dotted", "after every model's release",
               "after every model studied was released", "one found post hoc", "meets the same criterion, post hoc"):
    check(f"text (v5): replaced wording absent: {phrase!r}", phrase not in flat and phrase not in tex)
check("text (v5): training-date wording with possible overlap", "after every model studied was trained" in flat and flat.count("overlap with the training data remains possible") == 2)
check("text (v5): novelty claims scoped", flat.count("To our knowledge") >= 4)
check("text (v5): few-items caveat next to the boundary test's pass", "on too few numbers to show that the failure holds beyond them" in flat)
avail = flat.split("\\paragraph{Code and data availability.}", 1)
check("text (v5): code and data availability paragraph directly before the AI-use statement", len(avail) == 2 and avail[1].split("\\paragraph{", 1)[1].startswith("Use of language-model assistants."))
check("text (v6): repository address and the release archive DOI, no placeholder left", "\\url{https://github.com/Jeevash23/interp-small-decisions}" in flat and "\\url{https://doi.org/10.5281/zenodo.24150049} (release v1.0; later releases are listed under the same record)" in flat and ("X" * 4) not in tex)
check("text (v5): companion preprint carries paper A's archive DOI", "zenodo.24112443" in (HERE / "refs.bib").read_text())
check("text (v5): round 3 cited in Table 1, Section 3, Section 6, the limitations and Figure 1", all(x in tex for x in ("\\nv{r3.ind.npass} of \\nv{r3.ind.ntests}", "\\label{sec:r3}", "\\nv{r3.e.E1}", "round~3 tested prospectively", "confirmed on new material in round~3")))
A3 = {h: NV.get(f"x1m.ph.A3rule.{h}") for h in ("L17H6", "L11H9", "L0H2", "L0H3")}
check("claim: L17H6 and L11H9, and no other single head, meet the criterion frozen for a shared part (A3)", A3 == {"L17H6": "met", "L11H9": "met", "L0H2": "not met", "L0H3": "not met"}, A3)
check("claim: PROTOCOL.md states that criterion and the bounds for L11H9", "generated terminal-dedent change ≤ −0.010 and below every random draw" in proto
      and "L11H9: \\|change\\| ≤ 0.05 on all four indentation measures" in proto and "L11H9: LB width and separation kept ≥ 0.93" in proto)
CLAIMS = [("L11H9", "gT", "below"), ("L17H6", "gT", "below"), ("L11H9", "t4", "below"), ("L11H9", "p4", "below"), ("L11H9", "LBw", "above"), ("L11H9", "LBs", "above"),
          ("L11H9", "Du", "below"), ("L11H9", "Dl", "below"), ("TERMnoL17H6", "LBw", "below"), ("CARRYnoL17H6", "LBs", "above"), ("CARRYnoL17H6", "t4", "below"),
          ("TERMnoL17H6", "t4", "below"), ("READnoL17H6", "t4", "above"), ("L17H6", "Dl", "below"), ("CARRY", "Du", "below"), ("DED", "LBw", "in"), ("DED", "LBs", "in")]
for g, c, want in CLAIMS:
    check(f"claim: {g} on {c} is {want} its random range", POS.get((g, c)) == want, POS.get((g, c)))
check("claim: L11H9's change at generated terminal dedents is larger than L17H6's", r("IND", "L11H9", "gT") < r("IND", "L17H6", "gT") < 0)
check("claim: CARRY and TERM without L17H6 pass the frozen bound (at least 0.92 of the width effect kept)", r("LB", "CARRY-L17H6", "LBw") >= 0.92 and r("LB", "TERM-L17H6", "LBw") >= 0.92)
OWN_TASK = {"COUNT": "LB", "WRITE": "LB", "READnoL17H6": "LB", "DED": "IND"}
harm = [(g, c) for (g, c), pos in POS.items() if g in OWN_TASK and FIELD[c][0] != OWN_TASK[g] and pos == "below"]
check("claim: COUNT, WRITE, READ without L17H6 and DED are never below their random range on another task's measure", not harm, harm)
general = {"L0H2", "L0H3"}; holders = {"L17H6", "L11H9", "CARRY", "TERM", "READ", "CARRYnoL17H6", "TERMnoL17H6"}
rest = sorted((g, c) for (g, c), pos in POS.items() if pos != "in" and g not in general | holders and FIELD[c][0] != OWN_TASK.get(g))
smallv = {f"{g}|{c}": round(abs(r(FIELD[c][0], g.replace("noL17H6", "-L17H6"), c) - (1 if FIELD[c][2] else 0)), 4) for g, c in rest}
check("claim: the other marked cross-task values are small (within 0.12 of no effect)", all(v <= 0.12 for v in smallv.values()), smallv)
for g in ("COUNT", "WRITE", "READ"):
    hs = raw["ORD"][g]["heads"].split(",")
    want = g + " = \\{" + ", ".join("\\hd{%s}{%s}" % tuple(h[1:].split("H")) for h in hs) + "\\}"
    check(f"text: members of {g} as in the run files", want in flat, want)
check("text: counts are of tests, not predictions", not re.search(r"predictions (held|passed)|prediction passed", flat))
check("text: no pass count in the abstract", "npass" not in tex.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0])
check("text: AI-use statement names no product", "language-model assistants" in body.lower())

def used_keys(text):
    """Every \\nv key a LaTeX text uses, directly or through one-argument shorthand macros."""
    keys = set(re.findall(r"\\nv\{([^{}#]+)\}", text))
    for name, body in re.findall(r"\\newcommand\{\\(\w+)\}\[1\]\{(.*)\}\s*$", text, flags=re.M):
        parts = re.findall(r"\\nv\{([\w.]*)#1([\w.]*)\}", body)
        for arg in re.findall(r"\\" + name + r"\{([^{}]+)\}", text):
            keys |= {a + arg + b for a, b in parts}
    return keys


# two capitals totals the source text does not typeset: recounted here from the per-model counts it does
OWN = {"d5.c.ntests": num("d5.c.g.ntests") + num("d5.c.q.ntests"), "d5.r2.ntests": num("d5.r2.q.v.n_tests") + num("d5.r2.g.v.n_tests")}
for k, v in OWN.items():
    check(f"{k} recounted from the per-model counts", num(k) == v, f"{NV[k]} vs {v}")
mine_all = used_keys(tex + "".join(f.read_text() for f in sorted((HERE / "tables").glob("*.tex"))))
for pre, (study, sub, _) in SRC.items():
    src_used = used_keys((AREA / study / sub / "main.tex").read_text())
    extra = sorted(k for k in mine_all if k.startswith(pre + ".") and k[len(pre) + 1:] not in src_used and k not in OWN)
    check(f"{pre}: every value used here is one the source paper typesets (so its checker recomputes it)", not extra, extra[:10])


# ------------------------------------------------------------------ 5. references
bib = (HERE / "refs.bib").read_text()
entries = {m.group(1): m.group(0) for m in re.finditer(r"@\w+\{([^,]+),(?:.|\n)*?\n\}", bib)}
cited = set()
for grp in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", body):
    cited |= {c.strip() for c in grp.split(",")}
check("references: every cited key defined", not (cited - set(entries)), sorted(cited - set(entries)))
check("references: every entry cited", not (set(entries) - cited), sorted(set(entries) - cited))
ns_ = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
rec = {}
for e in ET.parse(HERE / "sources" / "arxiv-api.xml").getroot().findall("a:entry", ns_):
    i = e.find("a:id", ns_).text.rsplit("/", 1)[-1].split("v")[0]
    rec[i] = dict(title=" ".join(e.find("a:title", ns_).text.split()), authors=[a.find("a:name", ns_).text for a in e.findall("a:author", ns_)],
                  date=e.find("a:published", ns_).text[:10])
web = json.loads((HERE / "sources" / "web-records.json").read_text())
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("\\", ""))
for k, e in sorted(entries.items()):
    t = re.search(r"title\s*=\s*\{(.*)\},\n", e).group(1)
    yr = re.search(r"year\s*=\s*\{(\d{4})\}", e).group(1)
    first = re.search(r"author\s*=\s*\{(.*?)\},\n", e, flags=re.S).group(1).split(" and ")[0]
    sur = first.split(",")[0].strip("{} ").split()[-1]
    m = re.search(r"eprint\s*=\s*\{(\d{4}\.\d{4,5})\}", e)
    if m:
        a = rec.get(m.group(1))
        check(f"reference {k}: arXiv {m.group(1)} in the saved record", a is not None)
        if a is None: continue
        check(f"reference {k}: title matches", norm(t) == norm(a["title"]), f"{t} | {a['title']}")
        check(f"reference {k}: first author matches", norm(sur) in norm(" ".join(a["authors"][:2])) or "team" in norm(first), f"{sur} | {a['authors'][:2]}")
        venue = re.search(r"booktitle\s*=|journal\s*=\s*\{(?!arXiv)", e) is not None
        check(f"reference {k}: year {yr} is the first version's or a venue year", yr == a["date"][:4] or (venue and int(yr) in (int(a["date"][:4]) + 1, int(a["date"][:4]) + 2)), a["date"])
    else:
        w = web.get(k)
        check(f"reference {k}: non-arXiv entry has a saved web record", w is not None)
        if w:
            check(f"reference {k}: title and year match the web record", norm(t) == norm(w["title"]) and yr == w["year"], f"{t} | {w['title']}")

# ------------------------------------------------------------------ report
nbad = sum(not r["ok"] for r in RES)
if WRITE:
    (HERE / "checks").mkdir(exist_ok=True)
    (HERE / "checks" / "number-check.json").write_text(json.dumps(RES, indent=1, ensure_ascii=False))
    md = [f"# Number check, combined paper v6 (12 November 2026)", "",
          f"Generated by `scripts/check_numbers.py`, which shares no code with `scripts/make_numbers.py`. {len(RES)} checks, {nbad} failed.", ""]
    md += [f"- **FAILED** {r['name']}: {r['detail']}" for r in RES if not r["ok"]] + ["", "## All checks", ""]
    md += [f"- {'ok' if r['ok'] else 'FAILED'}: {r['name']}" for r in RES]
    (HERE / "checks" / "NUMBER-CHECK.md").write_text("\n".join(md) + "\n")
print(f"{len(RES)} checks, {nbad} failed (numbers file: {NUMFILE})")
for r in RES:
    if not r["ok"]: print("FAILED", r["name"], r["detail"][:300])
sys.exit(1 if nbad else 0)
