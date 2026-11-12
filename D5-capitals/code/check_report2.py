"""Independent check of D5 round 2 (written 8 Nov 2026 during the confirmation run, post hoc reporting tool).
Shares no code with r2_verdict.py or make_report2.py. Reads the raw per-position records of the held-out
runs (outputs/r2/*_reserve_*), recomputes every measure and every pass/fail of PROTOCOL-2.md with its own
code, re-runs the intervals with its own bootstrap (1,000 resamples of articles, other seed; tolerance 0.03),
compares everything with outputs/r2/conf_verdicts2.json, and checks that every decimal number in
REPORT-2.md is one of the recomputed values (as printed to 3 decimals, or to 2 where the report says so).
Writes outputs/r2/REPORT2-CHECK.md. Exit status 1 on any failure."""
import json, random, re, sys
from pathlib import Path
import torch

D = Path(__file__).resolve().parent.parent; R2 = D / "outputs" / "r2"
V = json.load(open(R2 / "conf_verdicts2.json"))
fails, checks, values = [], 0, {}


def ok(cond, what):
    global checks
    checks += 1
    if not cond: fails.append(what)


def letters(s): return [c for c in s if c.isalpha()]
def kase(s):
    u = any(c.isupper() for c in s); l = any(c.islower() for c in s)
    if not u and not l: return "none"
    return "U" if u and not l else ("L" if l and not u else "M")
def same(a, b):
    a = a.strip().lower(); b = b.strip().lower()
    if not a or not b or not a.isalpha() or not b.isalpha(): return False
    return a.startswith(b) or b.startswith(a)
def klass(s):
    t = letters(s.strip())
    if not t: return "none"
    if len(t) > 1 and all(c.isupper() for c in t): return "ALL"
    if t[0].isupper() and any(c.islower() for c in t): return "Init"
    if len(t) == 1 and t[0].isupper(): return "One"
    if all(c.islower() for c in t): return "low"
    return "mix"
def mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def bootci(pairs, kind):
    """pairs: (article, value) for kind 'share', (article, (num, den)) for kind 'ratio'."""
    arts = sorted({a for a, _ in pairs}, key=str); by = {}
    for a, x in pairs: by.setdefault(a, []).append(x)
    rng = random.Random("independent-check-r2"); vals = []
    for _ in range(1000):
        xs = [x for a in (rng.choice(arts) for _ in arts) for x in by[a]]
        if kind == "share": vals.append(mean(xs))
        else: vals.append(mean([n for n, _ in xs]) / mean([d for _, d in xs]))
    vals.sort(); return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def record(model, key, pairs, kind):
    if kind == "share": pt = mean([x for _, x in pairs])
    else: pt = mean([n for _, (n, d) in pairs]) / mean([d for _, (n, d) in pairs])
    values[f"{model}.{key}"] = pt
    got = V["models"][model]["measures"].get(key)
    ok(got is not None and abs(got["point"] - pt) < 1e-9, f"{model} {key}: point {pt} vs {got and got['point']}")
    if got is not None:
        lo, hi = bootci(pairs, kind)
        ok(abs(got["lo"] - lo) < 0.03 and abs(got["hi"] - hi) < 0.03, f"{model} {key}: interval [{got['lo']:.3f}, {got['hi']:.3f}] vs [{lo:.3f}, {hi:.3f}]")
        values[f"{model}.{key}.lo"] = got["lo"]; values[f"{model}.{key}.hi"] = got["hi"]
    return pt


TH = {}
for model in ("qwen3-17b", "gemma"):
    P = {}
    s = torch.load(R2 / f"exp03_{model}_reserve_0_conf.pt", weights_only=False); recs = s["records"]; nr = len(s["random"]["writers"])
    ok(s["part"] == "reserve", f"{model} sets part")
    # writers
    up = [r for r in recs if kase(r["U"]["top"]) == "U"]
    w1 = record(model, "W_named_upper", [(r["pageid"], float(kase(r["W"][1][0]) == "U")) for r in up], "share")
    wsw = record(model, "W_named_same_word", [(r["pageid"], float(same(r["U"]["top"], r["W"][1][0]))) for r in up], "share")
    wr = [mean([float(kase(r["W"][j][0]) == "U") for r in up]) for j in range(2, 2 + nr)]
    values[f"{model}.W_random_upper.min"] = min(wr)
    record(model, "W_named_proj_removed", [(r["pageid"], (r["W"][0][2] - r["W"][1][2], r["W"][0][2] - r["L"]["proj"])) for r in recs], "ratio")
    P["W1"] = w1 <= 0.30 and wsw >= 0.55 and min(wr) >= 0.85
    # carriers
    pr = lambda key, j, c, ref: [(r["pageid"], (r[key][0][c] - r[key][j][c], r[key][0][c] - r["L"][ref])) for r in recs]
    ca = record(model, "CA_named_proj_removed", pr("CA", 1, 2, "proj"), "ratio")
    cad = record(model, "CA_named_D_removed", pr("CA", 1, 1, "D"), "ratio")
    car = [mean([n for _, (n, d) in pr("CA", j, 2, "proj")]) / mean([d for _, (n, d) in pr("CA", j, 2, "proj")]) for j in range(2, 2 + nr)]
    values[f"{model}.CA_random_proj_removed.max"] = max(car)
    values[f"{model}.CA_none_upper"] = mean([float(kase(r["CA"][0][0]) == "U") for r in recs])
    record(model, "CA_named_upper", [(r["pageid"], float(kase(r["CA"][1][0]) == "U")) for r in recs], "share")
    cu = record(model, "CU_named_upper", [(r["pageid"], float(kase(r["CU"][1][0]) == "U")) for r in up], "share")
    cur = [mean([float(kase(r["CU"][j][0]) == "U") for r in up]) for j in range(2, 2 + nr)]
    values[f"{model}.CU_random_upper.min"] = min(cur)
    P["C1"] = ca >= 0.60 and cad >= 0.30 and max(car) <= ca - 0.25
    P["C2"] = cu <= 0.60 and min(cur) >= 0.90
    # readers
    lo = [r for r in recs if kase(r["L"]["top"]) == "L"]
    r0 = record(model, "R_none_flip", [(r["pageid"], float(kase(r["R"][0][0]) == "U")) for r in lo], "share")
    r1 = record(model, "R_named_flip", [(r["pageid"], float(kase(r["R"][1][0]) == "U")) for r in lo], "share")
    rr = [mean([float(kase(r["R"][j][0]) == "U") for r in lo]) for j in range(2, 2 + nr)]
    values[f"{model}.R_random_flip.min"] = min(rr)
    P["R1"] = r0 >= 0.80 and r1 <= 0.20 and min(rr) >= r0 - 0.15
    for k, n in (("W", len(up)), ("R", len(lo)), ("CA", len(recs))): values[f"{model}.n_{k}"] = n
    # single head
    if model == "qwen3-17b":
        s1 = torch.load(R2 / f"exp03_{model}_reserve_0_l17h6.pt", weights_only=False); rc1 = s1["records"]
        f = lambda j: [(r["pageid"], (r["CA"][0][2] - r["CA"][j][2], r["CA"][0][2] - r["L"]["proj"])) for r in rc1]
        c3 = record(model, "L17H6_proj_removed", f(1), "ratio")
        c3r = [mean([n for _, (n, d) in f(j)]) / mean([d for _, (n, d) in f(j)]) for j in range(2, 2 + len(s1["random"]["carriers"]))]
        values[f"{model}.L17H6_random_proj_removed.max"] = max(c3r)
        P["C3"] = c3 >= 0.20 and max(c3r) < c3
    # attention
    a = json.load(open(R2 / f"exp06_{model}_reserve_0.json"))
    loc, brd = {"qwen3-17b": ("L17H6", "L19H0"), "gemma": ("L14H7", "L24H6")}[model]
    for h in [k for k in a["records"][0] if k.startswith("L")]:
        record(model, f"attn_{h}_recent8", [(r["pageid"], (sum(r[h][0:4]), sum(r[h][0:5]))) for r in a["records"]], "ratio")
        for b, name in enumerate(a["bins"]): values[f"{model}.attn_{h}.{name}"] = mean([r[h][b] for r in a["records"]])
    P["C4"] = values[f"{model}.attn_{loc}_recent8"] >= 0.60 and values[f"{model}.attn_{brd}_recent8"] <= 0.50
    # edits
    e = json.load(open(R2 / f"exp05_{model}_reserve_0.json"))["records"]
    fl = [r for r in e if kase(r["lo"]) == "L" and kase(r["edit"]) == "U"]
    chg = lambda r: float(not same(r["lo"], r["edit"]))
    values[f"{model}.E_flips"] = len(fl)
    record(model, "E_change", [(r["pageid"], chg(r)) for r in fl], "share")
    uc = record(model, "E_change_if_up_changes", [(r["pageid"], chg(r)) for r in fl if not same(r["lo"], r["up"])], "share")
    uk = record(model, "E_change_if_up_keeps", [(r["pageid"], chg(r)) for r in fl if same(r["lo"], r["up"])], "share")
    record(model, "E_change_not_by_up", [(r["pageid"], chg(r) if same(r["lo"], r["up"]) else 0.0) for r in fl], "share")
    t5 = record(model, "E_changed_in_lower_top5", [(r["pageid"], float(r["edit_lower_rank"] is not None and r["edit_lower_rank"] <= 5)) for r in fl if chg(r)], "share")
    x0 = record(model, "E_change_extra0", [(r["pageid"], chg(r)) for r in fl if r["pieces_up"] == r["pieces_lo"]], "share")
    x1 = record(model, "E_change_extra1p", [(r["pageid"], chg(r)) for r in fl if r["pieces_up"] - r["pieces_lo"] >= 1], "share")
    values[f"{model}.E_random_flip"] = mean([float(kase(r["rand"]) == "U") for r in fl])
    values[f"{model}.E1_diff"] = uc - uk; values[f"{model}.E2_diff"] = x1 - x0
    P["E1"] = uc - uk >= 0.30 and t5 >= 0.40
    P["E2"] = x1 - x0 >= 0.10
    # other capitals
    paras, heads = [], []
    for start in (0, 264):
        o = json.load(open(R2 / f"exp04_{model}_reserve_{start}.json")); paras += o["para"]; heads += o["head"]; gap = o["proj_upper_mean"] - o["proj_lower_mean"]
        values[f"{model}.proj_upper_mean"] = o["proj_upper_mean"]; values[f"{model}.proj_lower_mean"] = o["proj_lower_mean"]
    def kept(rs, kind, base, j, want):
        return [(r["pageid"], float(klass(r["tops"][j]) == want)) for r in rs if r["kind"] == kind and klass(r["tops"][0]) == base]
    a1 = record(model, "A_acr_rm_keeps", kept(paras, "acr", "ALL", 1, "ALL"), "share")
    a1r = record(model, "A_acr_rm_rand_keeps", kept(paras, "acr", "ALL", 2, "ALL"), "share")
    a2 = record(model, "A_init_rm_keeps", kept(paras, "init", "Init", 1, "Init"), "share")
    a2r = record(model, "A_init_rm_rand_keeps", kept(paras, "init", "Init", 2, "Init"), "share")
    caps = [r for r in heads if r["caps"]]
    a3 = record(model, "A_head_rm_keeps", kept(caps, "hin", "ALL", 1, "ALL"), "share")
    a3r = record(model, "A_head_rm_rand_keeps", kept(caps, "hin", "ALL", 2, "ALL"), "share")
    def paired(kind):
        x = {(r["item"], r["word"].lower()): r for r in heads if r["kind"] == kind and r["caps"]}
        y = {(r["item"], r["word"].lower()): r for r in heads if r["kind"] == kind and not r["caps"]}
        return [(x[k]["pageid"], (x[k]["proj"] - y[k]["proj"]) / gap) for k in x if k in y]
    hi_ = record(model, "A_head_inside_excess", paired("hin"), "share")
    ha = record(model, "A_head_after_excess", paired("hbody"), "share")
    hall = record(model, "A_head_after_allcaps_top", [(r["pageid"], float(klass(r["tops"][0]) == "ALL")) for r in heads if r["kind"] == "hbody" and r["caps"]], "share")
    for k in ("acr", "init"): values[f"{model}.n_{k}"] = sum(r["kind"] == k for r in paras)
    P["A1"] = a1 >= 0.85 and a1r >= 0.90
    P["A2"] = a2 >= 0.90 and a2r >= 0.95
    P["A3"] = a3 <= 0.85 and a3r >= 0.95
    P["A4"] = hi_ >= 0.40 and ha <= 0.25 and hall <= 0.10
    got = V["models"][model]["predictions"]
    ok(set(got) == set(P), f"{model} prediction set {sorted(got)} vs {sorted(P)}")
    for k in P: ok(got.get(k) == P[k], f"{model} {k}: {got.get(k)} vs {P[k]}")
    TH[model] = P
    # verdicts as fixed in PROTOCOL-2.md, Section 5
    def v2(a, b): return "supported" if P[a] and P[b] else ("partly supported" if P[a] or P[b] else "not supported")
    want = {"Q1_parts": v2("W1", "R1"), "Q2_carriers": v2("C1", "C2"), "Q3_word_changes": v2("E1", "E2"),
            "Q4_acronyms_initials_separate": "supported" if P["A1"] and P["A2"] else "not supported",
            "Q4_headings_same_direction_reset": "supported" if P["A3"] and P["A4"] else "not supported"}
    for k, w in want.items(): ok(V["models"][model]["verdicts"].get(k) == w, f"{model} verdict {k}")
    values[f"{model}.npass"] = sum(P.values()); values[f"{model}.ntests"] = len(P)

# protocol thresholds as written in PROTOCOL-2.md (spot check that the numbers used here are the frozen ones)
proto = (D / "PROTOCOL-2.md").read_text()
for t in ("≤ 0.30", "≥ 0.55", "≥ 0.85", "≥ 0.80", "≤ 0.20", "0.15", "≥ 0.60", "≥ 0.30", "0.25", "≤ 0.60", "≥ 0.90", "≥ 0.20",
          "≤ 0.50", "≥ 0.40", "0.10", "≥ 0.95", "≤ 0.85", "≤ 0.25", "≤ 0.10"):
    ok(t in proto, f"threshold {t} not found in PROTOCOL-2.md")

# exploration values are quoted from the frozen exploration check (digest as recorded in FROZEN.md)
import hashlib
ef = R2 / "explore_verdicts2_check.json"
dig = hashlib.sha256(ef.read_bytes()).hexdigest()
ok(dig in (D / "FROZEN.md").read_text(), "explore_verdicts2_check.json digest not in FROZEN.md")
EX = json.load(open(ef))
def walk(x, pre):
    if isinstance(x, dict):
        if "point" in x: values[pre] = x["point"]
        else:
            for k, v in x.items(): walk(v, f"{pre}.{k}")
    elif isinstance(x, list):
        if x and all(isinstance(v, (int, float)) for v in x): values[pre + ".min"] = min(x); values[pre + ".max"] = max(x)
    elif isinstance(x, (int, float)) and not isinstance(x, bool): values[pre] = x
for m_, v_ in EX["models"].items(): walk(v_["measures"], "explore." + m_)
ok(V["parts_account_both_models"] == all(TH[m][a] and TH[m][b] and TH[m][c] and TH[m][d] for m in TH for a, b, c, d in [("W1", "R1", "C1", "C2")]), "overall verdict")

# numbers in the report
rep = D / "REPORT-2.md"
if rep.exists():
    text = rep.read_text(); text = re.sub(r"<!-- nocheck -->.*", "", text)
    allowed = set()
    for v in values.values():
        if isinstance(v, (int, float)) and v == v:
            allowed |= {f"{v:.3f}", f"{v:.2f}", f"{abs(v):.3f}", f"{abs(v):.2f}"}
    thresholds = set(re.findall(r"\d\.\d+", proto))
    for num in re.findall(r"(?<![\w.])(\d\.\d{2,3})(?![\d])", text):
        ok(num in allowed or num in thresholds, f"REPORT-2.md number {num} not recomputed")
    for model in ("qwen3-17b", "gemma"):
        ok(f"{values[model + '.npass']} of {values[model + '.ntests']}" in text, f"REPORT-2.md pass count for {model}")

out = ["# Independent check of D5 round 2", "", f"Checks: {checks}; failed: {len(fails)}.", ""]
out += [f"- FAIL: {f}" for f in fails] or ["All checks passed."]
(R2 / "REPORT2-CHECK.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
sys.exit(1 if fails else 0)
