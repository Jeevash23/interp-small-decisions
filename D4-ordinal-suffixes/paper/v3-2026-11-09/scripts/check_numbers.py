"""Independent check of every number, claim, text rule and reference in the D4 write-up. Standard library only.

  python3 -B scripts/check_numbers.py

Shares no code with make_numbers.py or the study's code/. It recomputes the held-out cells from the per-item records
(AUROC by direct pair counting), the ablation and patching values from the raw run files, the gate from the per-item
gate records and the exploration sweep from its per-item records. It then compares them with numbers.tex, re-derives
every pass or fail from the thresholds and compares those with EVALUATION.json and PROTOCOL.md, and checks the text
(every \\nv key defined, no hand-typed decimals, British spelling, no product names, no repo paths in captions) and the
references (every key cited, every arXiv entry matching its saved abstract page).
Writes checks/NUMBER-CHECK.md and checks/number-check.json; exits 1 if anything fails.
"""
import json, re, sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
D4 = PAPER.parents[1]
OUT = D4 / "outputs"
RES = []


def check(name, ok, detail=""):
    RES.append(dict(name=name, ok=bool(ok), detail=str(detail)))


def readj(p):
    with open(p) as f:
        return json.load(f)


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# numbers.tex as typeset
NV = {}
for line in (PAPER / "numbers.tex").read_text().splitlines():
    m = re.match(r"\\expandafter\\def\\csname nv@(.+?)\\endcsname\{(.*)\}$", line)
    if m: NV[m.group(1)] = m.group(2)


def num(s):
    return float(s.replace("\u2212", "-").replace(",", ""))


def same(key, x, dec=3, label="recount"):
    if key not in NV:
        check(f"{label}: {key}", False, "missing from numbers.tex"); return
    want = round(x, dec)
    got = num(NV[key])
    ok = abs(got - want) < 10 ** -dec / 2 + 1e-9 or abs(got - x) <= 10 ** -dec / 2 + 1e-9
    check(f"{label}: {key}", ok, f"text {NV[key]} vs recomputed {x:.6f}")


def auroc(pos, neg):
    """P(pos > neg) + 0.5 P(tie), by counting every pair."""
    if not pos or not neg: return None
    w = 0.0
    for a in pos:
        for b in neg:
            w += 1.0 if a > b else 0.5 if a == b else 0.0
    return w / (len(pos) * len(neg))


SUF = ["st", "nd", "rd", "th"]


def suffix(n):
    if n % 100 in (11, 12, 13): return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def mval(n, lg):
    return lg[n % 10 - 1] - lg[3]


def right(n, lg):
    return SUF[max(range(4), key=lambda i: lg[i])] == suffix(n)


def mean(xs):
    xs = list(xs); return sum(xs) / len(xs)


# ------------------------------------------------------------------ held-out behaviour (B1-B5)
R = jl(OUT / "conf" / "conf01_olmo2-1b.jsonl")
rec = {}
for d in range(3, 8):
    for form in ("plain", "comma"):
        s = [r for r in R if r["set"] == "length" and r["digits"] == d and r["form"] == form]
        if not s: continue
        t = [r for r in s if r["n"] % 100 in (11, 12, 13)]; e = [r for r in s if r["n"] % 100 not in (11, 12, 13)]
        check(f"olmo {d} {form}: classes match the stored labels", all(r["cls"] == "teen" for r in t) and all(r["cls"] == "end123" for r in e))
        a = auroc([mval(r["n"], r["logits"]) for r in e], [mval(r["n"], r["logits"]) for r in t])
        k = f"c.olmo.{d}{form[0]}"
        rec[(d, form)] = a
        same(k + ".auroc", a); same(k + ".acct", mean(right(r["n"], r["logits"]) for r in t))
        same(k + ".acce", mean(right(r["n"], r["logits"]) for r in e))
        same(k + ".nnum", len(set(r["n"] for r in t)), 0)
        check(f"claim: olmo {d} {form} has 200 teen and 200 non-teen items", len(t) == 200 and len(e) == 200, f"{len(t)}/{len(e)}")
        # tokenisation claim: tens digit in the previous token exactly at 4 and 7 digits, plain form
        tin = all(r["tens_in_last"] for r in s); tout = not any(r["tens_in_last"] for r in s)
        expect_out = form == "plain" and d in (4, 7)
        check(f"claim: olmo {d} {form} tens digit {'before' if expect_out else 'in'} the last token for every item", tout if expect_out else tin)
same("c.olmo.B3.d4", rec[(4, "comma")] - rec[(4, "plain")]); same("c.olmo.B3.d7", rec[(7, "comma")] - rec[(7, "plain")])
same("c.olmo.B4.r54", rec[(5, "plain")] - rec[(4, "plain")]); same("c.olmo.B4.r67", rec[(6, "plain")] - rec[(7, "plain")])
b2 = [rec[(3, "plain")], rec[(5, "plain")], rec[(6, "plain")]] + [rec[(d, "comma")] for d in (4, 5, 6, 7)]
same("c.olmo.B2.min", min(b2))
E = readj(OUT / "conf" / "EVALUATION.json")
for key, c in E["cells"]["olmo2-1b"].items():
    d, form = key.split("|"); k = f"c.olmo.{d}{form[0]}"
    same(k + ".lo", c["auroc_ci95"][0], label="interval as saved by the frozen evaluator")
    same(k + ".hi", c["auroc_ci95"][1], label="interval as saved by the frozen evaluator")

b5 = {}
for m, k in (("gpt2", "gpt2"), ("pythia-1b", "p1b")):
    R = jl(OUT / "conf" / f"conf01_{m}.jsonl")
    cell = {}
    for s in ("boundary", "control"):
        x = [r for r in R if r["set"] == s]
        t = [r for r in x if r["n"] % 100 in (11, 12, 13)]; e = [r for r in x if r["n"] % 100 not in (11, 12, 13)]
        a = auroc([mval(r["n"], r["logits"]) for r in e], [mval(r["n"], r["logits"]) for r in t])
        acc = mean(right(r["n"], r["logits"]) for r in t)
        cell[s] = (a, acc)
        same(f"c.{k}.{s}.auroc", a); same(f"c.{k}.{s}.acct", acc)
        same(f"c.{k}.{s}.nteen", len(t), 0); same(f"c.{k}.{s}.nend", len(e), 0); same(f"c.{k}.{s}.nnum", len(set(r["n"] for r in t)), 0)
        if s == "boundary":
            check(f"claim: {m} boundary numbers are 1,000-99,999 with the tens digit before the last token",
                  all(1000 <= r["n"] <= 99999 and not r["tens_in_last"] for r in x))
        c = E["cells"][m][s]
        same(f"c.{k}.{s}.lo", c["auroc_ci95"][0], label="interval as saved by the frozen evaluator")
        same(f"c.{k}.{s}.hi", c["auroc_ci95"][1], label="interval as saved by the frozen evaluator")
    b5[m] = cell
    same(f"c.{k}.gap", cell["control"][0] - cell["boundary"][0])
check("claim: GPT-2 and Pythia-1B boundary sets each have 10 distinct teen numbers and the same teen item count",
      NV["c.gpt2.boundary.nnum"] == NV["c.p1b.boundary.nnum"] == "10" and NV["c.gpt2.boundary.nteen"] == NV["c.p1b.boundary.nteen"])

# ------------------------------------------------------------------ held-out mechanism (Q1-Q6, O1-O2)
def abl(path):
    d = readj(path)["digits"]; out = {}
    for k, x in d.items():
        b = x["base"]["effect"]
        out[int(k)] = dict(base=b, bauroc=x["base"]["auroc"], kept=x["candidate"]["effect"] / b,
                          rmin=min(r["effect"] for r in x["random"]) / b, rmax=max(r["effect"] for r in x["random"]) / b,
                          nrand=len(x["random"]), tdrop=x["base"]["acc_teen"] - x["candidate"]["acc_teen"],
                          edrop=x["base"]["acc_end123"] - x["candidate"]["acc_end123"], tacc=x["candidate"]["acc_teen"],
                          singles={h: v["effect"] / b for h, v in x["singles"].items()}, per=readj(path)["per"])
    return out


qa = abl(OUT / "conf" / "exp05_qwen3-06b_0.2-1.6.json")
for d, q in qa.items():
    for f in ("kept", "rmin", "rmax", "tdrop", "edrop", "tacc"): same(f"c.q3.{d}.{f}", q[f])
    same(f"c.q3.{d}.base", q["base"], 2); same(f"c.q3.{d}.bauroc", q["bauroc"])
    same(f"c.q3.{d}.L0H2", q["singles"]["0.2"]); same(f"c.q3.{d}.L1H6", q["singles"]["1.6"])
    check(f"claim: Qwen3 ablation at {d} digits used 96 numbers per class and 20 random pairs", q["per"] == 96 and q["nrand"] == 20)
oa = abl(OUT / "conf" / "exp05_olmo2-1b_0.5-0.0-0.9.json")
for d, y in oa.items():
    for f in ("kept", "rmin", "rmax"): same(f"c.ol.{d}.{f}", y[f])
    same(f"c.ol.{d}.base", y["base"], 2); same(f"c.ol.{d}.bauroc", y["bauroc"])
    for h, v in y["singles"].items(): same(f"c.ol.{d}.s{h.replace('.', '')}", v)
    check(f"claim: OLMo ablation at {d} digits used 96 numbers per class and 20 random triples", y["per"] == 96 and y["nrand"] == 20)
pat = {}
for d in (3, 4):
    for kind in ("intoteen", "last"):
        s = readj(OUT / "conf" / f"exp04_qwen3-06b_{d}d_{kind}.json")
        H = s["head"]; best = max(((H[l][h], l, h) for l in range(len(H)) for h in range(len(H[0]))))
        k = f"c.q3.p{d}d_{kind}"
        pat[(d, kind)] = (H[0][2], best)
        same(k + ".L0H2", H[0][2]); same(k + ".topv", best[0])
        check(f"recount: {k}.top", NV[k + ".top"] == f"L{best[1]}H{best[2]}", f"{NV[k + '.top']} vs L{best[1]}H{best[2]}")
        same(k + ".nvalid", s["n_valid"], 0)
        check(f"claim: patching scan {d} {kind} used 96 pairs", s["n"] == 96)
        if kind == "intoteen": same(f"c.q3.att{d}", s["att"][0][2][1], 2)

# ------------------------------------------------------------------ verdicts from thresholds
thr = {k[4:]: num(v) for k, v in NV.items() if k.startswith("thr.")}
prot = (D4 / "PROTOCOL.md").read_text()
for tid, keys in {"B1": ["B1"], "B2": ["B2"], "B3": ["B3"], "B4": ["B4.acc4", "B4.acc3", "B3"], "B5": ["B5.bd", "B5.ct", "B3"],
                  "Q1": ["Q1.kept", "Q1.rand"], "Q2": ["Q2.teen", "Q2.end"], "Q3": ["Q3.kept", "Q3.acc"],
                  "Q4": ["Q4.L0H2", "Q4.L1H6"], "Q5": ["Q5"], "Q6": ["Q6.abs", "Q6.top"], "O1": ["O1.kept", "O1.rand", "O1.read"], "O2": ["O2"]}.items():
    line = next((l for l in prot.splitlines() if l.startswith(f"- **{tid} ") or l.startswith(f"- **{tid}.**")), "")
    for k in keys:
        check(f"threshold {k} appears in PROTOCOL.md under {tid}", f"**{NV['thr.' + k]}**" in line or f"**≥ {NV['thr.' + k]}**" in line
              or f"**≤ {NV['thr.' + k]}**" in line or f"{NV['thr.' + k]}" in line)
P = {}
P["B1"] = rec[(4, "plain")] <= thr["B1"] and rec[(7, "plain")] <= thr["B1"]
P["B2"] = min(b2) >= thr["B2"]
P["B3"] = rec[(4, "comma")] - rec[(4, "plain")] >= thr["B3"] and rec[(7, "comma")] - rec[(7, "plain")] >= thr["B3"]
P["B4"] = (rec[(5, "plain")] - rec[(4, "plain")] >= thr["B3"] and rec[(6, "plain")] - rec[(7, "plain")] >= thr["B3"]
           and num(NV["c.olmo.4p.acct"]) <= thr["B4.acc4"] and num(NV["c.olmo.3p.acct"]) >= thr["B4.acc3"])
for m in ("gpt2", "pythia-1b"):
    c = b5[m]; P["B5." + m] = c["boundary"][1] <= thr["B5.bd"] and c["control"][1] >= thr["B5.ct"] and c["control"][0] - c["boundary"][0] >= thr["B3"]
P["Q1"] = all(qa[d]["kept"] <= thr["Q1.kept"] and qa[d]["rmin"] >= thr["Q1.rand"] for d in (3, 4))
P["Q2"] = all(qa[d]["tdrop"] >= thr["Q2.teen"] and qa[d]["edrop"] <= thr["Q2.end"] for d in (3, 4))
P["Q3"] = qa[2]["kept"] >= thr["Q3.kept"] and qa[2]["tacc"] >= thr["Q3.acc"]
P["Q4"] = all(qa[d]["singles"]["0.2"] <= thr["Q4.L0H2"] and qa[d]["singles"]["1.6"] >= thr["Q4.L1H6"] for d in (3, 4))
P["Q5"] = all(pat[(d, "intoteen")][1][1:] == (0, 2) and pat[(d, "intoteen")][0] >= thr["Q5"] for d in (3, 4))
P["Q6"] = all(abs(pat[(d, "last")][0]) <= thr["Q6.abs"] and pat[(d, "last")][1][0] >= thr["Q6.top"] for d in (3, 4))
P["O1"] = (all(oa[d]["base"] >= thr["O1.read"] for d in (4, 7))
           and all(oa[d]["kept"] <= thr["O1.kept"] and oa[d]["rmin"] >= thr["O1.rand"] for d in (4, 7)))
P["O2"] = all(oa[d]["kept"] >= thr["O2"] for d in (3, 5))
T = E["tests"]
for k, v in P.items():
    saved = T["B5"][k.split(".")[1]]["passed"] if k.startswith("B5.") else T[k]["passed"]
    check(f"verdict {k}: recomputed pass={v} agrees with the frozen evaluator", v == saved, f"saved {saved}")
check("claim: all tests passed (count in the text)", all(P.values()) and NV["c.npass"] == NV["c.ntests"] == str(len(P)), f"{sum(P.values())}/{len(P)}")
check("claim: all three accounts supported", E["verdict_boundary"] == E["verdict_qwen_copy"] == E["verdict_olmo_move"] == "supported")

# ------------------------------------------------------------------ exploration numbers
for m, k in (("qwen3-06b", "q3"), ("smollm2-17b", "smol"), ("gemma4-e2b", "gemma"), ("olmo2-1b", "olmo")):
    R = jl(OUT / f"exp03_{m}.jsonl"); au = {}
    for d in range(2, 8):
        s = [r for r in R if r["digits"] == d and r["form"] == "plain"]
        t = [r for r in s if r["cls"] == "teen"]; e = [r for r in s if r["cls"] == "end123"]
        au[d] = auroc([mval(r["n"], r["logits"]) for r in e], [mval(r["n"], r["logits"]) for r in t])
    same(f"x3.{k}.d2", au[2], label="exploration recount"); same(f"x3.{k}.min37", min(au[d] for d in range(3, 8)), label="exploration recount")
    same(f"x3.{k}.max37", max(au[d] for d in range(3, 8)), label="exploration recount")
    if k == "olmo":
        lows = sorted(range(3, 8), key=lambda d: au[d])[:2]
        check("claim: OLMo-2's two lowest exploration cells are 4 and 7 digits", sorted(lows) == [4, 7], lows)
    if k != "olmo":
        dat = {int(l.split()[0]): l.split() for l in (PAPER / "fig-auroc.dat").read_text().splitlines()[1:]}
        col = {"q3": 3, "smol": 4, "gemma": 5}[k]
        check(f"figure data {k} matches recount", all(abs(float(dat[d][col]) - au[d]) < 5e-4 for d in range(2, 8)))
dat = {int(l.split()[0]): l.split() for l in (PAPER / "fig-auroc.dat").read_text().splitlines()[1:]}
check("figure data OLMo-2 held-out plain and separator match recount",
      all(abs(float(dat[d][1]) - rec[(d, "plain")]) < 5e-4 for d in range(3, 8)) and all(abs(float(dat[d][2]) - rec[(d, "comma")]) < 5e-4 for d in range(4, 8)))
xq = abl(OUT / "exp05_qwen3-06b_0.2-1.6.json")
for d in (2, 3, 4): same(f"x5.q3.kept{d}", xq[d]["kept"], 2, "exploration recount"); same(f"x5.q3.rmin{d}", xq[d]["rmin"], 2, "exploration recount")
xo = abl(OUT / "exp05_olmo2-1b_0.5-0.0-0.9.json")
for d in (3, 4, 5, 7): same(f"x5.olmo.kept{d}", xo[d]["kept"], 2, "exploration recount")
same("x5.olmo.rmin", min(xo[d]["rmin"] for d in xo), 2, "exploration recount")
s4 = readj(OUT / "exp04_qwen3-06b_4d_intoteen.json"); s2 = readj(OUT / "exp04_qwen3-06b_2d_intoteen.json")
same("x4.q3.4d.L0H2", s4["head"][0][2], 2, "exploration recount"); same("x4.q3.4d.att", s4["att"][0][2][1], 2, "exploration recount")
same("x4.q3.2d.L0H2", s2["head"][0][2], 2, "exploration recount")
o3 = [readj(OUT / f"exp04_olmo2-1b_3d_{k}.json") for k in ("intoteen", "outteen")]
o4 = [readj(OUT / f"exp04_olmo2-1b_4d_{k}.json") for k in ("intoteen", "outteen")]
same("x4.olmo.3d.maxhead", max(v for s in o3 for row in s["head"] for v in row), 2, "exploration recount")
same("x4.olmo.3d.mlp13", min(s["mlp"][13] for s in o3), 2, "exploration recount")
check("claim: MLP13 is the largest MLP in the OLMo-2 3-digit scans", all(max(range(len(s["mlp"])), key=lambda l: s["mlp"][l]) == 13 for s in o3))
same("x4.olmo.4d.L0H9.lo", min(s["head"][0][9] for s in o4), 2, "exploration recount")
same("x4.olmo.4d.L0H5.hi", max(s["head"][0][5] for s in o4), 2, "exploration recount")
check("claim: L0H5, L0H0, L0H9 are the three largest heads in both OLMo-2 4-digit scans",
      all(sorted(((s["head"][l][h], (l, h)) for l in range(len(s["head"])) for h in range(len(s["head"][0]))), reverse=True)[i][1]
          in ((0, 5), (0, 0), (0, 9)) for s in o4 for i in range(3)))
same("x4.olmo.4d.att.L0H5", o4[0]["att"][0][5][1], 2, "exploration recount")
R6 = jl(OUT / "exp06_qwen3-06b.jsonl"); T6 = [r for r in R6 if r["n"] % 100 in (11, 12, 13)]
same("x6.att.ok", mean(r["att"]["0.2"] for r in T6 if r["ok"]), label="exploration recount")
same("x6.att.bad", mean(r["att"]["0.2"] for r in T6 if not r["ok"]), label="exploration recount")
for fam in ("rankings", "anniversaries"): same(f"x6.acc.{fam}", mean(r["ok"] for r in T6 if r["family"] == fam), label="exploration recount")
same("x6.n", len(T6), 0, "exploration recount")

# ------------------------------------------------------------------ gate, material, times
for m, k in (("qwen3-06b", "q3"), ("pythia-410m", "p410"), ("pythia-1b", "p1b"), ("olmo2-1b", "olmo"), ("smollm2-17b", "smol"),
             ("gemma4-e2b", "gemma"), ("gpt2", "gpt2")):
    R = jl(OUT / f"gate01_{m}.jsonl")
    g1 = [right(r["n"], r["logits"]) for r in R if r["family"] == "dates" and 1 <= r["n"] <= 31]
    g2 = [right(r["n"], r["logits"]) for r in R if r["family"] != "dates" and 32 <= r["n"] <= 999 and r["n"] % 10 in (1, 2, 3)
          and r["n"] % 100 not in (11, 12, 13)]
    same(f"gate.{k}.G1", mean(g1), label="gate recount"); same(f"gate.{k}.G2", mean(g2), label="gate recount")
    if k == "p410": same("gate.p410.miss", 0.80 - mean(g2), label="gate recount")
fr = jl(D4 / "data" / "frames.jsonl")
for part in ("explore", "conf", "reserve"): same(f"mat.{part}.frames", sum(f["part"] == part for f in fr), 0, "material recount")
same("mat.frames", len(fr), 0, "material recount")
check("claim: five frame families", len(set(f["family"] for f in fr)) == 5)
fz = (D4 / "FROZEN.md").read_text()
check("freeze time of the protocol", f"| PROTOCOL.md | " in fz and f"2026-11-08 {NV['freeze.protocol']}" in fz)
fz1 = fz.split("\n## Round 2")[0]   # v2: count only the first round's protocol table
nfiles = len(re.findall(r"\| code/\S+ \| [0-9a-f]{64} \| 2026-11-08", fz1))
check("claim: the protocol froze eleven scripts", nfiles == 11, nfiles)
check("run window", (OUT / "conf" / "chain_conf.started").read_text().split()[1].startswith(NV["run.start"]) and
      (OUT / "conf" / "chain_conf.done").read_text().split()[1].startswith(NV["run.end"]))

# ------------------------------------------------------------------ text
tex = (PAPER / "main.tex").read_text()
body = tex[tex.index("\\begin{document}"):tex.index("\\end{document}")]
keys = re.findall(r"\\nv\{([^}]+)\}", body)
check("every \\nv key is defined", all(k in NV for k in keys), [k for k in keys if k not in NV])
plain = re.sub(r"\\nv\{[^}]+\}", "", body)
plain = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", plain, flags=re.S)
dec = re.findall(r"(?<![\w.{])\d+\.\d+", plain)
allowed = {"0.6", "1.7", "2.5"}   # model names (Qwen3-0.6B-Base, SmolLM2-1.7B) only
check("no hand-typed decimals in the text (numbers come from numbers.tex)", all(d in allowed for d in dec), sorted(set(dec) - allowed))
check("model-name decimals only inside model names", all(re.search(rf"(Qwen3-{d}B|SmolLM2-{d}B|Qwen{d})", plain) for d in set(dec)))
low = plain.lower()
for w in ("claude", "anthropic", "codex", "openai", "chatgpt", "gpt-4", "gemini"):
    check(f"no product or company name: {w}", w not in low.replace("gpt-2", ""))
us = ["behavior", "tokenization", "tokenizer", "normalization", "analyze", "color", "modeling", "favor", "labeled",
      "summarize", "recognize", "organize", "generalize", "center ", "characterize"]
bodytext = re.sub(r"\\cite[pt]?\{[^}]*\}", "", low)
check("British spelling (no US forms outside cited titles)", not [w for w in us if w in bodytext], [w for w in us if w in bodytext])
caps = re.findall(r"\\caption\{(.*?)\}\s*\n", body)
check("no repo paths in captions", not any(re.search(r"outputs/|code/|\.py|\.json|\.jsonl|\.md", c) for c in caps))
check("AI-use statement says 'language-model assistants'", "language-model assistants" in body)
check("author line is a placeholder", "[Author line to be added]" in tex)
for lab in ("exploratory", "prospective", "post hoc"):
    check(f"label used: {lab}", lab in low)
check("claim: O1 narrow margin reported", "\\nv{c.ol.7.rmin}" in body and "narrow" in body)
check("claim: 200 teen and 200 non-teen numbers per OLMo cell", "200 teen and 200 non-teen" in body and NV["c.olmo.nteen"] == "200")

# ================================================================== round 2 (second protocol, PROTOCOL-2.md)
# Recomputed from the raw run files in outputs/r2conf/ and outputs/r2/ only; the frozen evaluator's output
# (EVALUATION-2.json) is used afterwards, to compare pass/fail and counts, never as a source of values.
R2O = OUT / "r2conf"
COV = set()   # every r2./thr2. key that some check below has compared


def dec_of(s):
    """Number of decimals in a displayed string."""
    return len(s.split(".")[1]) if "." in s else 0


def s2(key, x, label="round-2 recount"):
    """Compare a recomputed value with the displayed string at the decimals the string shows."""
    COV.add(key)
    if key not in NV:
        check(f"{label}: {key}", False, "missing from numbers.tex"); return
    same(key, x, dec_of(NV[key]), label)


def eq2(key, want, label="round-2 recount"):
    COV.add(key)
    check(f"{label}: {key}", NV.get(key) == want, f"text {NV.get(key)} vs expected {want}")


def kept(after, before):
    return after / before


NUMJ = readj(PAPER / "checks" / "numbers.json")
R2KEYS = [k for k in NUMJ if k.startswith("r2.") or k.startswith("thr2.")]
check("round 2: numbers.json and numbers.tex agree on every r2./thr2. key", all(NUMJ[k] == NV.get(k) for k in R2KEYS),
      [k for k in R2KEYS if NUMJ[k] != NV.get(k)])
thr2 = {k[5:]: num(v) for k, v in NV.items() if k.startswith("thr2.")}
READ = thr2["R1.read"]   # the 0.30-logit read rule; its presence in R1, O4 and O8 is checked against the protocol below
P2 = {}                  # test id -> True / False / None (not read), recomputed here

# ---- every summary in the confirmation run is on 96 teen and 96 non-teen items (AUROC with ties as one half on the
#      1/(2*96^2) lattice, accuracies on the 1/96 lattice); the e06 checkpoint summaries on 64 and 64
def summaries(x):
    if isinstance(x, dict):
        if {"teen", "pull", "auroc", "acc_teen", "acc_end"} <= set(x): yield x
        for v in x.values(): yield from summaries(v)
    elif isinstance(x, list):
        for v in x: yield from summaries(v)


def on_lattice(v, n):
    return abs(v * n - round(v * n)) < 1e-6


RAW = {f: readj(R2O / f"{f}.json") for f in ("qwen", "qwen2", "olmo", "other_qwen3-17b", "other_smollm2-17b", "other_gemma4-e2b")}
for f, d in RAW.items():
    ss = list(summaries(d["digits"]))
    check(f"claim: round-2 {f} summaries are on 96 teen and 96 non-teen items (AUROC and accuracy lattices; {len(ss)} summaries)",
          ss and all(on_lattice(s["auroc"], 2 * 96 * 96) and on_lattice(s["acc_teen"], 96) and on_lattice(s["acc_end"], 96) for s in ss))
check("claim: round-2 Qwen3 path run reports n = 192 items per length", all(x["n"] == 192 for x in RAW["qwen"]["digits"].values()))
prot2 = (D4 / "PROTOCOL-2.md").read_text()
check("claim: 96 teen and 96 non-teen numbers per length (PROTOCOL-2.md section 2 and the text)",
      "96 teen (ending in 11–13) and 96 non-teen numbers ending in 1–3 per length" in prot2
      and "on 96 teen and 96 non-teen numbers per length" in body and "96 teen and 96 non-teen numbers per length, plain form" in body)

# ---- R1-R6: Qwen3-0.6B-Base, path (4 and 6 digits)
Q = RAW["qwen"]["digits"]
r2q = {}
for d in ("4", "6"):
    x = Q[d]; b = x["base"]["teen"]; k = f"r2.q.{d}"
    s2(k + ".base", b); s2(k + ".bauroc", x["base"]["auroc"])
    # R1: path shares of L0H2 by receiver
    full = b - x["full_L0H2"]["teen"]
    sh = {r: (b - v["teen"]) / full for r, v in x["path_L0H2"].items()}
    via = sh["mlp0"]; other = max(v for r, v in sh.items() if r != "mlp0")
    s2(k + ".R1.via", via); s2(k + ".R1.other", other)
    P2[f"R1@{d}"] = None if full < READ else (via >= thr2["R1.via"] and other <= thr2["R1.other"])
    check(f"round 2 R1@{d}: receivers are attention layers 1-27 and MLPs 0-27 (no layer-0 attention, no unembedding)",
          set(x["path_L0H2"]) == {f"attn{i}" for i in range(1, 28)} | {f"mlp{i}" for i in range(28)})
    # R2: L0H2 through MLP0 against the other layer-0 heads through MLP0
    own = b - x["path_L0H2"]["mlp0"]["teen"]
    ctl = max(b - c["via_mlp0"]["teen"] for c in x["ctl_L0"])
    s2(k + ".R2.own", own); s2(k + ".R2.ctl", ctl)
    P2[f"R2@{d}"] = own > ctl
    check(f"claim: R2@{d} controls are the other fifteen layer-0 heads",
          sorted(c["head"] for c in x["ctl_L0"]) == [h for h in range(16) if h != 2] and len(x["ctl_L0"]) == 15)
    # R3: mediation over MLP12-27
    def med(sender, fullkey):
        fl = b - x[fullkey]["teen"]
        m = {r: (b - v["teen"]) / fl for r, v in x[sender].items()}
        return m, fl
    m0, _ = med("med_L0H2", "full_L0H2"); m11, fl11 = med("med_L11H9", "full_L11H9")
    for nm, m in (("l0h2", m0), ("l11h9", m11)):
        check(f"round 2 R3@{d}: {nm} mediators are MLP12-MLP27", set(m) == {f"mlp{i}" for i in range(12, 28)})
        s2(f"{k}.R3.{nm}_mlp24", m["mlp24"]); s2(f"{k}.R3.{nm}_next", max(v for r, v in m.items() if r != "mlp24"))
    s2(k + ".R3.l11h9_full_loss", fl11)
    P2[f"R3@{d}"] = all(m["mlp24"] >= thr2["R3"] and m["mlp24"] > max(v for r, v in m.items() if r != "mlp24") for m in (m0, m11))
    # R4: logit lens
    lens = [s["teen"] for s in x["lens"]]
    rises = {i: lens[i] - lens[i - 1] for i in range(1, len(lens))}
    nxt = max(v for i, v in rises.items() if i != 24)
    s2(k + ".R4.rise", rises[24]); s2(k + ".R4.next", nxt)
    P2[f"R4@{d}"] = rises[24] > nxt and rises[24] >= thr2["R4"]
    check(f"round 2 R4@{d}: the lens has 28 layers and its last layer equals the model's own teen effect", len(lens) == 28 and abs(lens[27] - b) < 1e-4,
          f"{lens[27]} vs {b}")
    # R5: DLA of MLP24 and the largest other MLP, as shares of the teen effect
    dla = [v[0] for v in x["dla_mlp"]]
    s5 = dla[24] / b; o5 = max(abs(v) for i, v in enumerate(dla) if i != 24) / b
    s2(k + ".R5.share", s5); s2(k + ".R5.other", o5)
    P2[f"R5@{d}"] = s5 >= thr2["R5.share"] and o5 <= thr2["R5.other"]
    # R6: suffix profiles [st, nd, rd, th]
    def thm(p): return p[3] - mean(p[:3])
    def spec(p, j): return p[j - 1] - mean(p[i] for i in range(3) if i != j - 1)
    pr = x["profile"]
    th24 = [thm(pr["mlp24"][f"teen|{j}"]) for j in (1, 2, 3)]
    sp24 = [spec(pr["mlp24"][f"teen|{j}"], j) for j in (1, 2, 3)]
    s18t = [spec(pr["mlp18"][f"teen|{j}"], j) for j in (1, 2, 3)]
    s18n = [spec(pr["mlp18"][f"end123|{j}"], j) for j in (1, 2, 3)]
    s2(k + ".R6.thlo", min(th24)); s2(k + ".R6.thhi", max(th24)); s2(k + ".R6.spec24", max(abs(v) for v in sp24))
    s2(k + ".R6.s18lo", min(s18t + s18n)); s2(k + ".R6.s18hi", max(s18t + s18n))
    for j in (1, 2, 3):
        s2(f"{k}.R6.s18.teen{j}", s18t[j - 1]); s2(f"{k}.R6.s18.non{j}", s18n[j - 1])
    P2[f"R6@{d}"] = (min(th24) >= thr2["R6.th"] and max(abs(v) for v in sp24) <= thr2["R6.spec"]
                     and min(s18t + s18n) >= thr2["R6.s18"] and abs(mean(s18t) - mean(s18n)) <= thr2["R6.s18"])
    r2q[d] = dict(via=via, other=other, rise=rises[24], next=nxt, s5=s5, o5=o5, thlo=min(th24), thhi=max(th24),
                  spec24=max(abs(v) for v in sp24), s18lo=min(s18t + s18n), s18hi=max(s18t + s18n), lens=lens)
lensdat = {int(l.split()[0]): l.split() for l in (PAPER / "fig-lens.dat").read_text().splitlines()[1:]}
check("figure data: fig-lens.dat matches the logit-lens teen effect at 4 and 6 digits",
      len(lensdat) == 28 and all(abs(float(lensdat[i][1]) - r2q["4"]["lens"][i]) < 5e-5 and abs(float(lensdat[i][2]) - r2q["6"]["lens"][i]) < 5e-5
                                 for i in range(28)))

# ---- S1-S3: the backup route (2, 3, 4 digits)
S = RAW["qwen2"]["digits"]; drop = {}; ab = {}
for d in ("2", "3", "4"):
    x = S[d]; b = x["base"]["teen"]; k = f"r2.s.{d}"
    ka, kab, kb = (kept(x["sets"][s]["teen"], b) for s in ("A", "A+B", "B"))
    rk = [kept(r["teen"], b) for r in x["random"]]
    s2(k + ".base", b); s2(k + ".a", ka); s2(k + ".ab", kab); s2(k + ".b", kb)
    s2(k + ".random_min", min(rk)); s2(k + ".random_max", max(rk)); s2(k + ".drop", ka - kab)
    drop[d] = ka - kab; ab[d] = kab
    P2[f"S1@{d}"] = kab < min(rk)
    P2[f"S2@{d}"] = kb >= thr2["S2"]
    check(f"claim: S@{d} controls are 20 random pairs from layers 2-6, never L2H4 or L5H2 (nor L0H2, L1H6)",
          len(x["random"]) == 20 and all(len(r["heads"]) == 2 and all(2 <= int(re.match(r"L(\d+)H", h).group(1)) <= 6
                                                                     and h not in ("L2H4", "L5H2") for h in r["heads"]) for r in x["random"]))
P2["S3"] = drop["2"] >= thr2["S3.drop"] and drop["2"] > drop["3"] and drop["2"] > drop["4"] and ab["2"] >= thr2["S3.ab"]
check("claim: adding B lowers the teen effect kept most at two digits", drop["2"] == max(drop.values()))
check("claim: B alone does little (keeps at least 0.90 at every length)", all(num(NV[f"r2.s.{d}.b"]) >= 0.90 for d in ("2", "3", "4")))

# ---- O3-O8: OLMo-2-1B (3-10 digits)
O = RAW["olmo"]["digits"]; IN, CR = ("3", "5", "8", "9"), ("4", "7", "10")
ol = {}
for d, x in O.items():
    b = x["base"]["teen"]; k = f"r2.o.{d}"
    rk = [kept(r["teen"], b) for r in x["random"]]
    ol[d] = dict(base=b, auroc=x["base"]["auroc"], kept=kept(x["tri"]["teen"], b), rmin=min(rk),
                 p2=x["probe"]["2"], p15=x["probe"]["15"], vr=x["vnorm_ratio"])
    s2(k + ".base", b); s2(k + ".auroc", x["base"]["auroc"]); s2(k + ".acct", x["base"]["acc_teen"])
    s2(k + ".kept", ol[d]["kept"]); s2(k + ".rmin", min(rk)); s2(k + ".p2", x["probe"]["2"]); s2(k + ".p15", x["probe"]["15"])
    eq2(k + ".pieces", "+".join(str(len(p)) for p in x["pieces"]))
    check(f"round 2 O@{d}: the pieces join back into a {d}-digit number", len("".join(x["pieces"])) == int(d) and "".join(x["pieces"]).isdigit())
    check(f"claim: O@{d} controls are 20 random triples from layers 0-2, excluding L0H5, L0H0, L0H9",
          len(x["random"]) == 20 and all(len(r["heads"]) == 3 and all(int(re.match(r"L(\d+)H", h).group(1)) <= 2
                                                                     and h not in ("L0H5", "L0H0", "L0H9") for h in r["heads"]) for r in x["random"]))
check("claim (tokens): the tens digit is in the previous token exactly at 4, 7 and 10 digits",
      all((len(O[d]["pieces"][-1]) == 1) == (d in CR) for d in O))
P2["O3"] = (all(ol[d]["auroc"] >= thr2["O3.in"] for d in IN)
            and all(ol[d]["auroc"] <= thr2["O3.cr"] and ol[d]["auroc"] <= min(ol[i]["auroc"] for i in IN) - thr2["O3.gap"] for d in CR))
for d in IN: P2[f"O4@{d}"] = ol[d]["kept"] >= thr2["O4.in"]
for d in CR: P2[f"O4@{d}"] = None if ol[d]["base"] < READ else (ol[d]["kept"] <= thr2["O4.kept"] and ol[d]["kept"] < ol[d]["rmin"])
O5GAP = 0.10   # "at least 0.10 lower" in O5's text, checked below
P2["O5"] = (all(ol[d]["p2"] >= thr2["O5.p2"] and ol[d]["p15"] <= ol[d]["p2"] - O5GAP + 1e-12 for d in CR)
            and all(ol[d]["p15"] >= thr2["O5.in"] for d in IN))
for d in CR: s2(f"r2.o.{d}.vmax", max(ol[d]["vr"]))
s2("r2.o.5.vmin", min(ol["5"]["vr"]))
P2["O6"] = all(max(ol[d]["vr"]) <= thr2["O6"] for d in CR) and min(ol["5"]["vr"]) >= thr2["O6"]
check("round 2 O6: a norm ratio at every one of the 16 layers at 4, 5, 7 and 10 digits", all(len(ol[d]["vr"]) == 16 for d in ("4", "5", "7", "10")))
o7 = True; o8 = {}
for d in CR:
    st = O[d]["steer_L8"]; rnd = [st[f"rand{i}"] for i in range(5)]
    check(f"claim: O7@{d} uses 5 random directions", sorted(k for k in st if k.startswith("rand")) == [f"rand{i}" for i in range(5)])
    s2(f"r2.o.{d}.steer", -st["v3"]); s2(f"r2.o.{d}.rand", max(abs(v) for v in rnd))
    o7 = o7 and st["v3"] <= -thr2["O7.steer"] and all(st["v3"] < v for v in rnd) and max(abs(v) for v in rnd) <= thr2["O7.rand"]
    b = O[d]["base"]["teen"]; loss = b - O[d]["tri"]["teen"]
    sh = {r: (b - v["teen"]) / loss for r, v in O[d]["path_tri"].items()}
    nx = max(v for r, v in sh.items() if r != "mlp0")
    s2(f"r2.o.{d}.via", sh["mlp0"]); s2(f"r2.o.{d}.next", nx)
    P2[f"O8@{d}"] = None if loss < READ else (sh["mlp0"] >= thr2["O8"] and sh["mlp0"] > nx)
    o8[d] = (sh["mlp0"], nx)
P2["O7"] = o7

# ---- X: other one-token-per-digit models (3 and 4 digits)
SETS = readj(D4 / "code" / "r2conf_sets.json")
xr = {}
for m, k in (("qwen3-17b", "q17"), ("smollm2-17b", "smol"), ("gemma4-e2b", "gemma")):
    dd = RAW[f"other_{m}"]
    check(f"round 2 X {m}: head sets in the run file are the frozen ones (code/r2conf_sets.json)", dd["sets"] == SETS[m], dd["sets"])
    for d in ("3", "4"):
        x = dd["digits"][d]; b, pl = x["base"]["teen"], x["base"]["pull"]; kk = f"r2.x.{k}.{d}"
        s2(kk + ".base", b); s2(kk + ".bauroc", x["base"]["auroc"])
        for sname in dd["sets"]:
            tk, pk = kept(x["sets"][sname]["teen"], b), kept(x["sets"][sname]["pull"], pl)
            rand = x["random"][sname]
            rt = min(kept(r["teen"], b) for r in rand); rp = min(kept(r["pull"], pl) for r in rand)
            named = set(h for v in dd["sets"].values() for h in v)
            check(f"claim: X {m} {sname}@{d} controls are 20 random same-size sets from layers 0-1, excluding the named heads",
                  len(rand) == 20 and all(len(r["heads"]) == len(dd["sets"][sname]) and all(int(re.match(r"L(\d+)H", h).group(1)) <= 1
                                                                                          and h not in named for h in r["heads"]) for r in rand))
            s2(f"{kk}.{sname}.teen", tk); s2(f"{kk}.{sname}.pull", pk)
            if sname == "PREV":
                s2(f"{kk}.PREV.rmin", rt)
                P2[f"X:{m}:PREV@{d}"] = tk <= thr2["X.teen"] and tk < rt and pk >= thr2["X.pull"]
            else:
                s2(f"{kk}.CUR.rmin", rp)
                P2[f"X:{m}:CUR@{d}"] = pk <= thr2["X.cur"] and pk < rp and tk >= thr2["X.curteen"]
            xr[(m, sname, d)] = dict(teen=tk, pull=pk, rt=rt, rp=rp)
check("claim: only Qwen3-1.7B has CUR heads (SmolLM2 and Gemma-4 have none)", [m for m in SETS if "CUR" in SETS[m]] == ["qwen3-17b"])
check("claim: removing Qwen3-1.7B's CUR heads reverses the sign of the pull and keeps the teen effect",
      all(xr[("qwen3-17b", "CUR", d)]["pull"] < 0 and xr[("qwen3-17b", "CUR", d)]["teen"] >= thr2["X.curteen"] for d in ("3", "4")))
check("claim: in every model PREV ablation removes part of the teen effect while random sets remove almost none (all random >= 0.98 kept)",
      all(v["teen"] < 1 and v["rt"] >= 0.98 for (m, s, d), v in xr.items() if s == "PREV"))
fails = sorted(t for t, v in P2.items() if v is False)
check("claim: one prediction failed in both of its cells (Qwen3-1.7B PREV at 3 and 4 digits), and only on the pull",
      fails == ["X:qwen3-17b:PREV@3", "X:qwen3-17b:PREV@4"]
      and all(xr[("qwen3-17b", "PREV", d)]["teen"] <= thr2["X.teen"] and xr[("qwen3-17b", "PREV", d)]["teen"] < xr[("qwen3-17b", "PREV", d)]["rt"]
              and xr[("qwen3-17b", "PREV", d)]["pull"] < thr2["X.pull"] for d in ("3", "4")), fails)
check("claim: the failed pull values are written next to the X.pull threshold in the text",
      "\\nv{r2.x.q17.3.PREV.pull} and \\nv{r2.x.q17.4.PREV.pull} kept, against the \\nv{thr2.X.pull} required" in body)

# ---- pass/fail, counts and verdicts against the frozen evaluator
E2 = readj(R2O / "EVALUATION-2.json")
ET = {t: v for t, v in E2["tests"].items() if v["rule"] != "descriptive"}
check("round 2: the recomputed test list is the evaluator's test list", set(P2) == set(ET), sorted(set(P2) ^ set(ET)))
for t in sorted(P2):
    check(f"round-2 verdict {t}: recomputed pass={P2[t]} agrees with the frozen evaluator", t in ET and P2[t] == ET[t]["passed"],
          f"saved {ET.get(t, {}).get('passed')}")
check("round 2: the evaluator ran on held-out material (dry = false)", E2["dry"] is False)


def verdict(vals):
    r = [v for v in vals if v is not None]
    if all(r): return "supported"
    return "partly supported" if sum(r) >= len(r) / 2 else "not supported"


ACC = {"path": ("R", "qwen_path"), "backup": ("S", "qwen_backup"), "olmo": ("O", "olmo_weak_route"), "other": ("X:", "other_models")}
for a, (pre, ename) in ACC.items():
    vals = [v for t, v in P2.items() if t.startswith(pre)]
    rd = [v for v in vals if v is not None]
    s2(f"r2.{a}.n", len(rd)); s2(f"r2.{a}.pass", sum(rd))
    eq2(f"r2.{a}.verdict", verdict(vals))
    check(f"round 2 verdict for account {a} agrees with the frozen evaluator ({ename})", E2["verdicts"][ename] == verdict(vals), E2["verdicts"][ename])
rd = [v for v in P2.values() if v is not None]
s2("r2.ntests", len(rd)); s2("r2.npass", sum(rd))
check("round 2: test and pass counts agree with the frozen evaluator", (len(rd), sum(rd)) == (E2["n_tests"], E2["n_passed"]),
      f"{len(rd)}/{sum(rd)} vs {E2['n_tests']}/{E2['n_passed']}")
check("round 2: every test was read (no read rule failed)", all(v is not None for v in P2.values()))

# ---- range and bound claims in the prose: each macro pair or single bound is the min/max of what it summarises
def rng(name, lo_key, hi_key, vals):
    check(f"range claim: {name} ({lo_key}–{hi_key}) is the min–max of its set", abs(num(NV[lo_key]) - round(min(vals), dec_of(NV[lo_key]))) < 1e-9
          and abs(num(NV[hi_key]) - round(max(vals), dec_of(NV[hi_key]))) < 1e-9, f"min {min(vals):.4f}, max {max(vals):.4f}")


def bound(name, key, vals, kind="max"):
    v = max(vals) if kind == "max" else min(vals)
    check(f"bound claim: {name} ({key}) is the {kind} of its set", abs(num(NV[key]) - round(v, dec_of(NV[key]))) < 1e-9, f"{kind} {v:.4f}")


rng("abstract, MLP24 writes this share of the teen effect", "r2.q.4.R5.share", "r2.q.6.R5.share", [r2q[d]["s5"] for d in r2q])
bound("no other single receiver carries more than (R1, both lengths)", "r2.q.6.R1.other", [r2q[d]["other"] for d in r2q])
bound("the next-largest lens rise is at most (R4, both lengths)", "r2.q.6.R4.next", [r2q[d]["next"] for d in r2q])
bound("no other MLP accounts directly for more than (R5, both lengths)", "r2.q.4.R5.other", [r2q[d]["o5"] for d in r2q])
rng("MLP24 raises th above the mean of st/nd/rd (teens, both lengths, each last digit)", "r2.q.6.R6.thlo", "r2.q.4.R6.thhi",
    [r2q[d]["thlo"] for d in r2q] + [r2q[d]["thhi"] for d in r2q])
bound("MLP24 favours the own suffix by at most (both lengths)", "r2.q.6.R6.spec24", [r2q[d]["spec24"] for d in r2q])
rng("MLP18 own-suffix specificity (both classes, both lengths)", "r2.q.6.R6.s18lo", "r2.q.4.R6.s18hi",
    [r2q[d]["s18lo"] for d in r2q] + [r2q[d]["s18hi"] for d in r2q])
bound("OLMo-2: no other receiver above (O8, 4/7/10 digits)", "r2.o.7.next", [o8[d][1] for d in CR])
rng("OLMo-2: probe at layer 2, cross-token lengths", "r2.o.7.p2", "r2.o.4.p2", [ol[d]["p2"] for d in CR])
rng("OLMo-2: probe at layer 15, cross-token lengths", "r2.o.10.p15", "r2.o.4.p15", [ol[d]["p15"] for d in CR])
bound("OLMo-2: in-token lengths stay at or above (probe at layer 15)", "r2.o.9.p15", [ol[d]["p15"] for d in IN], "min")
bound("abstract, OLMo-2: the tens digit arrives at no more than (norm ratio, cross-token lengths)", "r2.o.4.vmax", [max(ol[d]["vr"]) for d in CR])
check("claim: OLMo-2's three heads carry most of what arrives at 4, 7, 10 digits (kept <= 0.5) and nothing in-token (kept 0.99-1.01)",
      all(ol[d]["kept"] <= 0.5 for d in CR) and all(0.99 <= ol[d]["kept"] <= 1.01 for d in IN))
check("claim: the probe reads less well by layer 15 at every cross-token length", all(ol[d]["p15"] < ol[d]["p2"] for d in CR))
# the prose places each range claim in this order; make sure the macro pairs used in main.tex are exactly these
for lo, hi in (("r2.q.4.R5.share", "r2.q.6.R5.share"), ("r2.q.6.R6.thlo", "r2.q.4.R6.thhi"), ("r2.q.6.R6.s18lo", "r2.q.4.R6.s18hi"),
               ("r2.o.7.p2", "r2.o.4.p2"), ("r2.o.10.p15", "r2.o.4.p15")):
    check(f"range claim written as \\nv{{{lo}}}--\\nv{{{hi}}} in the text", f"\\nv{{{lo}}}--\\nv{{{hi}}}" in body)
pairs = re.findall(r"\\nv\{(r2\.[^}]+)\}--\\nv\{(r2\.[^}]+)\}", body)
known = {("r2.q.4.R5.share", "r2.q.6.R5.share"), ("r2.q.6.R6.thlo", "r2.q.4.R6.thhi"), ("r2.q.6.R6.s18lo", "r2.q.4.R6.s18hi"),
         ("r2.o.7.p2", "r2.o.4.p2"), ("r2.o.10.p15", "r2.o.4.p15")} | {(f"r2.s.{d}.random_min", f"r2.s.{d}.random_max") for d in "234"}
check("every r2 macro range in the text is one of the ranges checked here", set(pairs) <= known, sorted(set(pairs) - known))

# ---- e06: OLMo-2 stage-one checkpoints (exploratory)
E6 = {}
for p in sorted((OUT / "r2").glob("e06_olmo2-1b_*.json")):
    d = readj(p); tok = re.search(r"tokens(\d+)B", d["rev"]).group(1)
    check(f"round 2 e06 {tok}B: revision name matches the file name and is a stage-one checkpoint",
          p.name == f"e06_olmo2-1b_{d['rev']}.json" and d["rev"].startswith("stage1-"))
    x = d["digits"]; k = f"r2.e6.{tok}"; E6[int(tok)] = x
    for dg in ("3", "4", "5", "7"): s2(f"{k}.auroc{dg}", x[dg]["base"]["auroc"], "exploratory checkpoint recount")
    b4 = x["4"]["base"]["teen"]
    s2(f"{k}.base4", b4, "exploratory checkpoint recount")
    if b4 >= READ: s2(f"{k}.tri", kept(x["4"]["tri"]["teen"], b4), "exploratory checkpoint recount")
    else: eq2(f"{k}.tri", "--", "exploratory checkpoint recount (not shown: four-digit teen effect below the read rule)")
    s2(f"{k}.p2", x["4"]["probe_last"]["2"], "exploratory checkpoint recount"); s2(f"{k}.p14", x["4"]["probe_last"]["14"], "exploratory checkpoint recount")
    ss = list(summaries(x))
    check(f"claim: e06 {tok}B summaries are on 64 teen and 64 non-teen items (caption)",
          all(on_lattice(s["auroc"], 2 * 64 * 64) and on_lattice(s["acc_teen"], 64) and on_lattice(s["acc_end"], 64) for s in ss))
check("claim: six checkpoints (six e06 files)", len(E6) == 6 and "Six checkpoints" in body, sorted(E6))
check("claim: the checkpoint table lists every e06 checkpoint and no other", set(int(t) for t in re.findall(r"\\nv\{r2\.e6\.(\d+)\.auroc3\}", body)) == set(E6))
check("claim: the table caption's read rule is thr2.R1.read", "at least \\nv{thr2.R1.read} logits" in body)
aur = lambda t, dg: E6[t][dg]["base"]["auroc"]
check("claim: the in-token check forms early (AUROC >= 0.95 at 3 and 5 digits from 84B tokens)",
      all(aur(t, dg) >= 0.95 for t in E6 if t >= 84 for dg in ("3", "5")))
check("claim: the cross-token check stays near chance (0.40-0.60) until about a trillion tokens",
      all(0.40 <= aur(t, dg) <= 0.60 for t in E6 if t < 1000 for dg in ("4", "7")))
check("claim: the cross-token check is still weak at the end of the stage (4 and 7 digits below the O3 in-token line)",
      all(aur(max(E6), dg) < thr2["O3.in"] for dg in ("4", "7")))
check("claim: the tens digit is readable by the layer-2 probe from early on (>= 0.90 from 21B tokens) and the reading fades with depth",
      all(E6[t]["4"]["probe_last"]["2"] >= 0.90 and E6[t]["4"]["probe_last"]["14"] < E6[t]["4"]["probe_last"]["2"] for t in E6 if t >= 21))
check("claim: the three heads do nothing at any checkpoint where the effect is readable (kept within 0.05 of 1)",
      all(abs(kept(E6[t]["4"]["tri"]["teen"], E6[t]["4"]["base"]["teen"]) - 1) <= 0.05 for t in E6 if E6[t]["4"]["base"]["teen"] >= READ))

# ---- post hoc (9 Nov): are the e06 checkpoints ancestors of the released model? Two saved outputs written
#      by separate code in separate studies: D4's outputs/r2post/olmo_lineage.json and D3's outputs/r2/r2e10b_dist.json.
LN = readj(OUT / "r2post" / "olmo_lineage.json")
D3L = readj(D4.parent / "D3-indentation" / "outputs" / "r2" / "r2e10b_dist.json")
EK = "model.embed_tokens.weight"
e_main = LN["main_vs_stage1end"][EK]["cos"]; e_d3 = D3L["final~s1_4001B"]["emb_cosine"]
same("post.lin.emb", e_main, 3, "post hoc lineage recount")
same("post.lin.emb", e_d3, 3, "post hoc lineage, D3's separate computation")
s2c = [v[EK]["cos"] for k, v in LN.items() if k.startswith("stage2-") and k.endswith("_vs_stage1end")]
check("post hoc lineage: four stage-2 checkpoints compared", len(s2c) == 4, len(s2c))
same("post.lin.s2min", min(s2c), 4, "post hoc lineage recount")
check("post hoc lineage: D3's stage-2 (ingredient 1, 51B) embedding cosine agrees with D4's to 4 decimals",
      abs(D3L["s1_4001B~s2i1_51B"]["emb_cosine"] - LN["stage2-ingredient1-step23852-tokens51B_vs_stage1end"][EK]["cos"]) < 1e-4)
check("post hoc lineage: every released-vs-stage-1-end layer tensor has |cosine| < 0.01",
      all(abs(v["cos"]) < 0.01 for k, v in LN["main_vs_stage1end"].items()))
check("claim: the text says the released weights do not descend from the checkpoint run, in the abstract, section, caption and limits",
      body.count("released weights do not descend") >= 3 and "is not the one the released weights come from" in body)
check("claim: the text no longer dates the released model's route from these checkpoints",
      "set up in the final training stage" not in body and "only in the final training stage" not in body)

# ---- times
fz2 = (D4 / "FROZEN.md").read_text()
eq2("r2.freeze.charter", re.search(r"\| CHARTER-2\.md \| [0-9a-f]{64} \| 2026-11-08 (\d\d:\d\d) ", fz2).group(1), "round-2 times")
mp = re.search(r"PROTOCOL-2\.md`\), frozen 9 Nov 2026, (\d\d:\d\d) IST", fz2)
eq2("r2.freeze.protocol", mp.group(1) if mp else None, "round-2 times")
st_, dn_ = ((R2O / f).read_text().split() for f in ("chain_conf.started", "chain_conf.done"))
eq2("r2.run.start", st_[1][:5], "round-2 times"); eq2("r2.run.end", dn_[1][:5], "round-2 times")
check("round 2: the run started on 9 Nov no earlier than the protocol freeze and ended before the 18:00 stop rule",
      st_[0] == dn_[0] == "2026-11-09" and st_[1][:5] >= NV["r2.freeze.protocol"] and dn_[1] < "18:00", f"{st_} {dn_}")
check("claim: PROTOCOL-2.md states the same freeze time", f"FROZEN 9 Nov 2026, {NV['r2.freeze.protocol']} IST" in prot2
      and f"frozen 8 Nov 2026, {NV['r2.freeze.charter']} IST" in prot2)

# ---- threshold resets before the freeze (NOTEBOOK.md)
nb = (D4 / "NOTEBOOK.md").read_text()
nb_r2 = nb[nb.index("## Round 2"):]
entries = re.split(r"\n(?=### )", nb_r2)
def entry_with(s):
    return next((e for e in entries if s in e), "")
e_s2o6 = entry_with("**Threshold changes before the freeze**")
e_prev = entry_with("the PREV threshold is set at")
hhmm = lambda e: re.match(r"### (?:9 Nov )?(\d\d:\d\d)", e).group(1) if e else "99:99"
check("NOTEBOOK.md records the S2 reset (0.90 -> thr2.S2) before the freeze",
      f"S2: B alone keeps ≥ **{NV['thr2.S2']}** (was 0.90)" in e_s2o6 and "9 Nov" in e_s2o6[:30] and hhmm(e_s2o6) < NV["r2.freeze.protocol"])
check("NOTEBOOK.md records the O6 reset (0.75 -> thr2.O6) before the freeze",
      f"≥ **{NV['thr2.O6']}** at every layer (was 0.75)" in e_s2o6 and hhmm(e_s2o6) < NV["r2.freeze.protocol"])
check("NOTEBOOK.md records the PREV reset (0.80 -> thr2.X.teen) before the freeze",
      f"PREV threshold is set at ≤ {NV['thr2.X.teen']} kept (draft 0.80)" in e_prev and "9 Nov" in e_prev[:30] and hhmm(e_prev) < NV["r2.freeze.protocol"])
check("claim: three thresholds were reset before the freeze (S2, O6 from the dry run; PREV from the last exploration run)",
      "Two thresholds were reset from that run, and a third (for the further models) from the last exploration run" in body
      and all(s in nb_r2 for s in ("(was 0.90)", "(was 0.75)", "(draft 0.80)")))
# The text says the three were reset "from that run", i.e. the dry run of the whole pipeline on exploration items.
# The notebook ties S2 and O6 to the dry run, but the PREV reset to the full Gemma-4-E2B exploration run (E04), and
# PROTOCOL-2.md section 4 says two tests' thresholds were reset from the dry run.
prev_from_dry = "dry" in e_prev.split("the PREV threshold is set at")[1].split("\n")[0]
check("claim: all three resets came from the dry run (\"three thresholds were reset from that run\")",
      not ("three thresholds were reset from that run" in body) or prev_from_dry,
      "NOTEBOOK.md ties the PREV reset to the Gemma-4-E2B exploration run (E04, 9 Nov 09:15 entry), not to the dry run; "
      "PROTOCOL-2.md section 4 names two resets from the dry run (S2, O6)")

# ---- thresholds: each appears in the text of its test in PROTOCOL-2.md section 4 (exploration brackets removed),
#      and in the frozen evaluator's rule for that test
sec4 = prot2[prot2.index("## 4. Predictions"):prot2.index("## 5.")]
def block(tid):
    lines = sec4.splitlines(); out = []; on = False
    for l in lines:
        if l.startswith("- **"):
            on = l.startswith(f"- **{tid}.**") or l.startswith(f"- **{tid}**")
        elif l.startswith("#") or not l.strip():
            on = False
        if on: out.append(l)
    return re.sub(r"\[[^\]]*\]", "", "\n".join(out))
def has_num(text, s):
    return re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", text) is not None
T2 = {"R1.via": "R1", "R1.other": "R1", "R1.read": "R1", "R3": "R3", "R4": "R4", "R5.share": "R5", "R5.other": "R5",
      "R6.th": "R6", "R6.spec": "R6", "R6.s18": "R6", "S2": "S2", "S3.drop": "S3", "S3.ab": "S3",
      "O3.in": "O3", "O3.cr": "O3", "O3.gap": "O3", "O4.kept": "O4", "O4.in": "O4", "O5.p2": "O5", "O5.in": "O5", "O6": "O6",
      "O7.steer": "O7", "O7.rand": "O7", "O8": "O8", "X.teen": "X-PREV", "X.pull": "X-PREV", "X.cur": "X-CUR", "X.curteen": "X-CUR"}
check("every thr2 key is mapped to a protocol test", set(T2) == set(thr2), sorted(set(thr2) ^ set(T2)))
EV_ID = {"X-PREV": "X:qwen3-17b:PREV@3", "X-CUR": "X:qwen3-17b:CUR@3"}
for k, tid in T2.items():
    COV.add("thr2." + k)
    check(f"threshold thr2.{k} = {NV['thr2.' + k]} appears in PROTOCOL-2.md under {tid}", has_num(block(tid), NV["thr2." + k]), block(tid)[:80])
    eid = {"O4.in": "O4@3"}.get(k) or EV_ID.get(tid, tid if tid in ("S3", "O3", "O5", "O6", "O7") else f"{tid}@4")
    rule = E2["tests"][eid]["rule"]
    check(f"threshold thr2.{k} appears in the frozen evaluator's rule for {eid}", has_num(rule, NV["thr2." + k]), rule)
for tid in ("O4", "O8"):
    check(f"read rule {NV['thr2.R1.read']} also appears in PROTOCOL-2.md under {tid}", "read" in block(tid).lower() and has_num(block(tid), NV["thr2.R1.read"]))
check("O5's 'at least 0.10 lower' matches the gap used here", "at least 0.10 lower" in block("O5") and O5GAP == 0.10)
check("claim: twenty random sets everywhere in round 2 (pairs, triples, sets)", "20 sets of A plus two random heads" in prot2
      and "20 random triples" in prot2 and "20 random sets of the same size" in prot2 and "20 random head sets" in body)

# ---- coverage: every r2./thr2. key in numbers.json is compared by some check above
miss = sorted(set(R2KEYS) - COV)
check(f"coverage: every r2./thr2. key ({len(R2KEYS)}) is checked", not miss, miss)

# ------------------------------------------------------------------ references
bib = (PAPER / "refs.bib").read_text()
entries = dict(re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S))
cited = set(k.strip() for grp in re.findall(r"\\cite[pt]?\{([^}]+)\}", body) for k in grp.split(","))
check("every cited key is in refs.bib", cited <= set(entries), cited - set(entries))
check("every refs.bib entry is cited", set(entries) <= cited, set(entries) - cited)
abs_ = readj(PAPER / "sources" / "arxiv-abs-pages.json")
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
for k, e in entries.items():
    m = re.search(r"(?:eprint\s*=\s*\{|arXiv:)(\d{4}\.\d{5})", e)
    if not m:
        check(f"reference {k}: non-arXiv entry checked in REFERENCE-CHECK.md", k == "radford2019language"); continue
    a = abs_.get(m.group(1))
    t = re.search(r"title\s*=\s*\{(.*)\},\n", e).group(1)
    first = re.search(r"author\s*=\s*\{(.*?)\},\n", e, flags=re.S).group(1).split(" and ")[0]
    sur = first.split(",")[0].strip("{} ").split()[-1]
    yr = re.search(r"year\s*=\s*\{(\d{4})\}", e).group(1)
    ok = a is not None and norm(t) == norm(a["title"]) and (norm(sur) in norm(" ".join(a["authors"][:2])))
    check(f"reference {k}: arXiv {m.group(1)} title and first author match the abstract page", ok,
          f"{t} | {a['title'] if a else None} | {sur}")
    check(f"reference {k}: year {yr} matches the first arXiv version or the venue", a is not None and (a["date"][:4] == yr or "booktitle" in e),
          f"arXiv {a['date'] if a else None}")

# ------------------------------------------------------------------ report
nok = sum(r["ok"] for r in RES); nbad = len(RES) - nok
(PAPER / "checks").mkdir(exist_ok=True)
json.dump(RES, open(PAPER / "checks" / "number-check.json", "w"), indent=1, ensure_ascii=False)
with open(PAPER / "checks" / "NUMBER-CHECK.md", "w") as f:
    f.write(f"# Number check, D4 write-up v2\n\nGenerated by `scripts/check_numbers.py`. {len(RES)} checks, {nok} passed, {nbad} failed.\n\n")
    for r in RES:
        if not r["ok"]: f.write(f"- **FAILED** {r['name']}: {r['detail']}\n")
    f.write("\n## All checks\n\n")
    for r in RES: f.write(f"- {'ok' if r['ok'] else 'FAILED'}: {r['name']}" + (f" ({r['detail']})" if r["detail"] and not r["ok"] else "") + "\n")
print(f"{len(RES)} checks, {nok} passed, {nbad} failed")
for r in RES:
    if not r["ok"]: print("FAILED", r["name"], r["detail"])
sys.exit(1 if nbad else 0)
