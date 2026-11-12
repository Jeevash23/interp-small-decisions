"""X1: writes REPORT.md from the saved confirmation outputs (written after the run; no model).
Every number in the tables comes from outputs/conf/verdicts.json, outputs/conf/c01_*.json and
outputs/conf/c02_attn_qwen3-06b-bf16.json; exploration values quoted for comparison come from PROTOCOL.md (as text)
and outputs/e01_summary_*.json. The prose is in code/report_template.md, with {placeholders} filled here.
Usage: build_report.py"""
import json, re
from pathlib import Path

X1 = Path(__file__).resolve().parent.parent
C = X1 / "outputs" / "conf"
VER = json.loads((C / "verdicts.json").read_text())
R = {t: json.loads((C / f"c01_{m}_{t}.json").read_text()) for m, t in
     [("qwen3-06b-bf16", "LB"), ("qwen3-06b-bf16", "IND"), ("qwen3-06b-bf16", "ORD"), ("qwen3-17b", "CAPS")]}
ATT = json.loads((C / "c02_attn_qwen3-06b-bf16.json").read_text())
T, CI = VER["tests"], VER["ci95"]
MEAS = {"LB": [("width_effect", "kept", "width kept"), ("sep", "kept", "separation kept")],
        "IND": [("code|dedent|lq", "change", "code unmarked"), ("code|term|lq", "change", "code terminal"),
                ("gen|dedent|lq", "change", "generated unmarked"), ("gen|term|lq", "change", "generated terminal")],
        "ORD": [("d4|teen_effect", "kept", "teen kept, 4 digits"), ("d5|teen_effect", "kept", "teen kept, 5 digits"),
                ("d4|pull", "kept", "pull kept, 4 digits"), ("d5|pull", "kept", "pull kept, 5 digits")],
        "CAPS": [("D_up", "kept", "D_up kept"), ("D_low", "kept", "D_low kept"), ("word_lp", "change", "word log-prob change")]}


def fm(x, d=3):
    if x is None: return "—"
    s = f"{x:.{d}f}"
    return s.replace("-", "−")


def val(t, g, key, kind):
    b = R[t]["base"][key]; v = R[t][g][key]
    return v / b if kind == "kept" else v - b


def rng(t, g, key, kind):
    xs = [val(t, x, key, kind) for x in R[t] if x.startswith(g + "~r")]
    return min(xs), max(xs)


def matrix(t):
    groups = [g for g in R[t] if g != "base" and "~r" not in g]
    head = "| Group | " + " | ".join(lab for _, _, lab in MEAS[t]) + " |"
    lines = [head, "|---" * (len(MEAS[t]) + 1) + "|"]
    for g in groups:
        cells = []
        for key, kind, _ in MEAS[t]:
            v = val(t, g, key, kind); lo, hi = rng(t, g, key, kind)
            cells.append(f"{fm(v)} ({fm(lo)} to {fm(hi)})")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def base_line(t):
    b = R[t]["base"]
    return ", ".join(f"{lab.replace(' kept', '').replace(' change', '')} {fm(b[key])}" for key, kind, lab in MEAS[t])


PASS = lambda n: {True: "pass", False: "**fail**", None: "not evaluable"}[T[n]["passed"]]
proto = (X1 / "PROTOCOL.md").read_text()
prot_rows = {}
for line in proto.splitlines():
    m = re.match(r"\| ([A-D]\d) \| (.*) \| (.*) \|$", line)
    if m: prot_rows[m.group(1)] = (m.group(2), m.group(3))


def ci(key):
    c = CI.get(key)
    return f"[{fm(c[0])}, {fm(c[1])}]" if c and c[0] is not None else "—"


def att(t, h, b): return fm(ATT[t][h]["dec"][b], 2)


F = {}
v = lambda t, g, k, kind: val(t, g, k, kind)
F.update({
    "lb_base": base_line("LB"), "ind_base": base_line("IND"), "ord_base": base_line("ORD"), "caps_base": base_line("CAPS"),
    "n_width": R["LB"]["base"]["n_width"], "n_nat": R["LB"]["base"]["n_nat"],
    "n_code_U": R["IND"]["base"]["code|dedent|n"], "n_code_T": R["IND"]["base"]["code|term|n"],
    "n_gen_U": R["IND"]["base"]["gen|dedent|n"], "n_gen_T": R["IND"]["base"]["gen|term|n"],
    "n_caps_pos": R["CAPS"]["base"]["n_pos"], "n_caps_par": R["CAPS"]["base"]["n_par"],
    "ord_n4": R["ORD"]["base"]["d4|n_teen"], "ord_n5": R["ORD"]["base"]["d5|n_teen"],
    "M_LB": matrix("LB"), "M_IND": matrix("IND"), "M_ORD": matrix("ORD"), "M_CAPS": matrix("CAPS"),
})
for t, g, key, kind, name in [
        ("LB", "L17H6", "width_effect", "kept", "a1"), ("LB", "L17H6", "sep", "kept", "a2"), ("IND", "L17H6", "gen|term|lq", "change", "a3"),
        ("LB", "CARRY", "width_effect", "kept", "a4c"), ("LB", "TERM", "width_effect", "kept", "a4t"),
        ("LB", "CARRY-L17H6", "width_effect", "kept", "a4cm"), ("LB", "TERM-L17H6", "width_effect", "kept", "a4tm"),
        ("IND", "L17H6", "code|dedent|lq", "change", "a5cu"), ("IND", "L17H6", "gen|dedent|lq", "change", "a5gu"),
        ("ORD", "L17H6", "d4|teen_effect", "kept", "a5t4"), ("ORD", "L17H6", "d5|teen_effect", "kept", "a5t5"),
        ("CAPS", "L17H6", "D_up", "kept", "a6"), ("CAPS", "L17H6", "word_lp", "change", "a6w"), ("CAPS", "L17H6", "D_low", "kept", "a6l"),
        ("LB", "WRITE", "width_effect", "kept", "b1"), ("LB", "COUNT", "sep", "kept", "b2"), ("LB", "READ-L17H6", "width_effect", "kept", "b3"),
        ("IND", "DED", "code|dedent|lq", "change", "b4"), ("LB", "DED", "width_effect", "kept", "b4w"), ("LB", "DED", "sep", "kept", "b4s"),
        ("ORD", "L0H2", "d4|teen_effect", "kept", "c1a"), ("ORD", "L0H2", "d5|teen_effect", "kept", "c1b"), ("LB", "L0H2", "sep", "kept", "c2"),
        ("ORD", "L0H3", "d4|pull", "kept", "c4a"), ("ORD", "L0H3", "d5|pull", "kept", "c4b"),
        ("ORD", "L0H3", "d4|teen_effect", "kept", "c4c"), ("ORD", "L0H3", "d5|teen_effect", "kept", "c4d"),
        ("LB", "L0H3", "sep", "kept", "c5"), ("LB", "L0H3", "width_effect", "kept", "c5w"),
        ("CAPS", "L0H3", "D_low", "kept", "c7l"), ("CAPS", "L0H3", "D_up", "kept", "c7u"), ("CAPS", "L0H3", "word_lp", "change", "c7w"),
        ("ORD", "L11H9", "d4|teen_effect", "kept", "d1a"), ("ORD", "L11H9", "d5|teen_effect", "kept", "d1b"),
        ("ORD", "L11H9", "d4|pull", "kept", "d1c"), ("ORD", "L11H9", "d5|pull", "kept", "d1d"),
        ("LB", "L11H9", "width_effect", "kept", "d2w"), ("LB", "L11H9", "sep", "kept", "d2s"),
        ("CAPS", "L11H9", "D_up", "kept", "d5u"), ("CAPS", "L11H9", "D_low", "kept", "d5l"),
        ("ORD", "CARRY", "d4|teen_effect", "kept", "ct4"), ("ORD", "CARRY-L17H6", "d4|teen_effect", "kept", "ctm4")]:
    F[name] = fm(v(t, g, key, kind)); lo, hi = rng(t, g, key, kind); F[name + "_lo"] = fm(lo); F[name + "_hi"] = fm(hi)
F["c7_ratio_l"] = fm(1 - v("CAPS", "L0H3", "D_low", "kept")); F["c7_ratio_u3"] = fm(3 * (1 - v("CAPS", "L0H3", "D_up", "kept")))
for h in ("L0H3", "L0H2", "L11H9"):
    F[f"att_{h}_IND_self"] = att("IND", h, "0")
    for t in ("LB", "IND", "ORD", "CAPS"):
        F[f"att_{h}_{t}_0"] = att(t, h, "0"); F[f"att_{h}_{t}_1"] = att(t, h, "1")
ind_abs = lambda g: max(abs(v("IND", g, k, "change")) for k in ("code|dedent|lq", "code|term|lq", "gen|term|lq"))
for g, n in (("WRITE", "b1"), ("COUNT", "b2"), ("READ-L17H6", "b3"), ("L0H2", "c3"), ("L0H3", "c6"), ("L11H9", "d3")):
    F[n + "_ind"] = fm(ind_abs(g)); F[n + "_indgu"] = fm(abs(v("IND", g, "gen|dedent|lq", "change")))
F["d3_max"] = fm(max(abs(v("IND", "L11H9", k, "change")) for k in ("code|dedent|lq", "code|term|lq", "gen|term|lq", "gen|dedent|lq")))
for k in ("L17H6|LBw", "L17H6|LBs", "L17H6|gT", "DED|cU", "L0H2|t4", "L0H2|t5", "L0H2|LBs", "L0H3|p4", "L0H3|p5", "L0H3|LBs",
          "L0H3|Dl", "L0H3|wl", "L11H9|t4", "L11H9|t5", "L11H9|p4", "L11H9|p5", "L17H6|Du"):
    F["ci_" + k.replace("|", "_")] = ci(k)
rows = []
for n in T:
    pred, expl = prot_rows.get(n, (T[n]["rule"], ""))
    rows.append(f"| {n} | {pred} | {expl} | {PASS(n)} |")
F["test_table"] = "\n".join(rows)
Vd = VER["verdicts"]
F.update({k: Vd[k] for k in ("lead1_L17H6", "lead2_separate", "lead3_L0H2", "lead3_L0H3", "lead4_L11H9", "account")})
F["n_passed"], F["n_tests"] = Vd["n_passed"], Vd["n_tests"]
F["failed"] = ", ".join(n for n in T if T[n]["passed"] is False) or "none"

# ---- prose pieces that depend on the outcome (all numbers filled from the files above)
E = json.loads((X1 / "outputs" / "e01_summary_qwen3-06b-bf16_all.json").read_text())["groups"]
E2 = json.loads((X1 / "outputs" / "e01_summary_qwen3-06b-bf16_dec.json").read_text())["groups"]
ex2 = lambda g, k, d=2: fm(E2[g][k]["value"], d)
ex = lambda g, k, d=2: fm(E[g][k]["value"], d)
exr = lambda g, k, d=3: f"{fm(E[g][k]['rand_min'], d)} to {fm(E[g][k]['rand_max'], d)}"
log = (X1 / "outputs" / "logs" / "conf_06b_IND.log").read_text()
F["end_time"] = re.findall(r"queue done (\d\d:\d\d)", log)[-1]
F["short_answer"] = ("Mostly separate parts. Line-breaking and indentation meet at a single head, L17H6, and the early heads that "
                     "the studies named are general movers of the previous or the current token, not parts of one task. "
                     "Every frozen prediction held on the held-out material." if Vd["account"] == "supported" else
                     "See the verdicts in Section 3: the account was only partly supported.")
F["a3_sentence"] = ("That change is small, but it lies below all 20 random heads of layer 17 (lowest {a3_lo}), as predicted."
                    .format(**F) if T["A3"]["passed"] else "That change did not pass its frozen test (A3).")
F["reading"] = "\n".join([
    "- **What \"separate\" means here.** D1-X's groups without L17H6 (COUNT, WRITE, READ−L17H6) each remove much of a "
    "line-breaking measure, and on no other task tested do they do more harm than the worst of their random groups. "
    "READ−L17H6 even raises the ordinal measures slightly above its random range. DED, likewise, harms only indentation. For COUNT, the held-out "
    f"separation kept is {F['b2']}, while the largest indentation change is {F['b2_ind']}. In exploration, an all-position "
    "ablation added a general-damage control: the mean next-token loss on ordinary prose. Removing COUNT everywhere "
    f"removed most of the width effect (kept {ex('COUNT', 'LB:width_effect')}) yet raised that loss by only "
    f"{ex('COUNT', 'CAPS:gen_loss', 3)} nats (random groups {exr('COUNT', 'CAPS:gen_loss')}). For L17H6 the figures were "
    f"{ex('L17H6', 'LB:width_effect')} and {ex('L17H6', 'CAPS:gen_loss', 3)} (random {exr('L17H6', 'CAPS:gen_loss')}) "
    "(exploratory). These are specialised parts.",
    "- **L17H6 as a format reader.** On held-out text L17H6 does three things. It carries the stored line length to the "
    f"line-break decision (width kept {F['a1']}). It adds a little to the dedent after a terminal statement ({F['a3']}). In "
    f"Qwen3-1.7B, by index, it trims the case gap ({F['a6']}) without changing which word is chosen ({F['a6w']}). Each role "
    "seems to move a fact about the recent text's format, not its content, to where the format is decided. This fits the "
    "charter's lead 5 (content first, format late), but X1 did not test that lead. So L17H6's place in both D1-X's "
    "readers and D3's CARRY group is not incidental: the head has a measurable role in both tasks. Its share of the "
    "indentation job is small, though. In exploration, CARRY's change on generated terminal dedents was "
    f"{ex2('CARRY', 'IND:gen|term|lq')} with L17H6 and {ex2('CARRY-L17H6', 'IND:gen|term|lq')} without it "
    "(exploratory; CARRY was not part of the held-out indentation run).",
    "- **Early heads.** L0H3 attends to the current token in every task. L0H2 attends mainly to the previous token "
    f"({F['att_L0H2_LB_1']} to {F['att_L0H2_ORD_1']} in prose, numbers and capitals text). At the newline token in code it "
    f"puts {F['att_L0H2_IND_0']} on itself and {F['att_L0H2_IND_1']} on the previous token. What a task takes from them "
    "is decided downstream. Ordinals need the tens digit moved one place "
    "(L0H2) and the last digit read where it stands (L0H3); line-breaking needs both. At the newline token that ends a "
    "line of code, neither matters to the indent decision. Exploration with all-position ablation (not tested here) found "
    f"that L0H2 does matter for indentation upstream: removing it everywhere moved generated terminal dedents by "
    f"{ex('L0H2', 'IND:gen|term|lq')} (random {exr('L0H2', 'IND:gen|term|lq', 2)}). Removing it everywhere also raised "
    f"general loss by {ex('L0H2', 'CAPS:gen_loss')} nats, and L0H3 by {ex('L0H3', 'CAPS:gen_loss')} nats, so neither is a "
    "task part.",
    "- **L11H9** attends to the current token. Ablating it shrinks the teen effect and the last-digit pull by similar "
    f"amounts ({F['d1a']} and {F['d1c']} kept at 4 digits). That fits a head that serves the suffix choice in general, "
    "not the teen check, as D4's notebook described it. Its place in D3's CARRY and TERM groups adds "
    f"little to indentation ({F['d3_max']} at most). In exploration, removing it everywhere raised general loss by "
    f"{ex('L11H9', 'CAPS:gen_loss', 3)} nats against at most {fm(E['L11H9']['CAPS:gen_loss']['rand_max'], 3)} for random "
    "layer-11 heads. That suggests a broader next-token role, which was not tested here.",
    "- **Lead 2 (the newline as a shared scratchpad) is answered no**, at least for these parts. D1-X's newline writers "
    "carry the stored length that line-breaking reads, but D3's indent decision at the same newline token does not use "
    f"them (largest change {F['b1_ind']}). Indentation reads the previous line through its own heads.",
])
F["limits"] = "\n".join([
    "- One model family. Line-breaking, indentation and ordinals were tested in Qwen3-0.6B-Base; capitals in Qwen3-1.7B-Base, "
    "with heads matched by index (D1-X showed the two models share line-breaking heads by index, but X1 assumed the match "
    "for the capitals heads).",
    "- Mean ablation at the decision positions only, with exploration means. Effects that pass through earlier positions "
    "(such as L0H2 on the terminal line, seen only in exploration) are outside the confirmatory tests.",
    "- \"Separate\" means within the range of 20 layer-matched random groups on these measures. Smaller shared effects, "
    "or sharing through MLPs and the residual stream, are not excluded.",
    "- Measures are the source studies' own, on new material of the same kinds: Wikipedia prose wrapped at width 60, "
    "Python from new repositories and generated programs, English ordinal frames up to five digits, and capitals in "
    "Wikipedia paragraphs.",
    "- The held-out material is held out from these analyses, not necessarily from the models' training data.",
])
tmpl = (X1 / "code" / "report_template.md").read_text()
(X1 / "REPORT.md").write_text(tmpl.format(**F))
print("REPORT.md written;", Vd)
