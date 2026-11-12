"""Copy the full prediction tables of the three source papers into tables/*.tex for this paper's appendices.

  python3 -B scripts/extract_tables.py

Only layout is copied. Every number in a copied table is a \\nv{...} macro (or a macro that expands to one); its key
is given the study prefix (d3., d4., d5.), so the value comes from this paper's numbers.tex like every other
number. Labels get the prefix too, and the model-name macros are renamed to this paper's.
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
AREA = PAPER.parent
SRC = {"d3": AREA / "D3-indentation/paper/v2-2026-11-09/main.tex",
       "d4": AREA / "D4-ordinal-suffixes/paper/v3-2026-11-09/main.tex",
       "d5": AREA / "D5-capitals/paper/v2-2026-11-08/main.tex"}
WANT = {"d3": ["tab:gate", "tab:conf", "tab:conf2q", "tab:conf2o", "tab:ckpt", "tab:prefix"],
        "d4": ["tab:tests", "tab:olmo", "tab:qwen", "tab:olmoabl", "tab:r2tests", "tab:backup", "tab:olmo2", "tab:other", "tab:ckpt"],
        "d5": ["tab:preds", "tab:head", "tab:preds2"]}
MODEL = {"d3": {r"\qw{}": r"\qsix{}", r"\ol{}": r"\olmo{}"}, "d4": {r"\qw{}": r"\qsix{}", r"\ol{}": r"\olmo{}"},
         "d5": {r"\qw{}": r"\qbig{}"}}
# v4: plain wording for three copied captions and the row labels of the two-measure gate table. Layout and numbers
# are untouched; the source papers are not edited.
WORDING = {
    ("d3", "tab:gate"): [
        ("Values after the measurement fix described in the text; \\cref{tab:d3-prefix} gives both versions.",
         "Values use the corrected measure; \\cref{tab:d3-prefix} gives both measures. ``Top-1 minus copy'' is the top-1 accuracy of the indent minus the accuracy of copying the previous line's indent; ``never-opened mass'' is the share of $q$ at dedents that falls on shallower indents that no open block has, for the indent width (two spaces, three spaces or mixed) where it is largest.")],
    ("d3", "tab:prefix"): [
        ("Gate on post-cut-off code with the first (pre-fix) and corrected (post-fix) measure. Gemma-4-E2B's first run failed at load and has no pre-fix row.",
         "Gate on post-cut-off code under the first measure, which left out decision points after blank lines for two tokenisers, and under the corrected measure. Gemma-4-E2B has no value under the first measure."),
        ("& pre-fix &", "& first &"), ("& post-fix &", "& corrected &")],
    ("d3", "tab:conf2q"): [
        ("$q$ on the current level changes by at most 0.05", "the probability $q$ of staying at the current indent changes by at most 0.05")],
    ("d3", "tab:ckpt"): [
        ("Exploratory. OLMo-2-0425-1B during training, on the same post-cut-off code.",
         "Exploratory. Checkpoints of a published training run of the OLMo-2 recipe, from which the released \\olmo{} weights do not descend, and the released model (last row), on the same post-cut-off code.")],
}
(PAPER / "tables").mkdir(exist_ok=True)
for pre, path in SRC.items():
    tex = path.read_text()
    blocks = re.findall(r"\\begin\{table\}.*?\\end\{table\}", tex, flags=re.S)
    for lab in WANT[pre]:
        hit = [b for b in blocks if "\\label{" + lab + "}" in b]
        assert len(hit) == 1, (pre, lab, len(hit))
        b = hit[0]
        b = re.sub(r"\\begin\{table\}\[[^\]]*\]", r"\\begin{table}[H]", b)
        b = b.replace("\\nv{", "\\nv{" + pre + ".")
        b = re.sub(r"\\(label|ref|cref|Cref)\{tab:", lambda m: "\\" + m.group(1) + "{tab:" + pre + "-", b)
        b = b.replace("\\cref{app:prefix}", "\\cref{tab:d3-prefix}")
        for a, c in MODEL[pre].items():
            b = b.replace(a, c)
        if pre == "d5":  # D5's shorthand macros become this paper's d5-prefixed ones
            for m in ("GI", "QI", "EG", "EQ", "RQI", "RGI", "RQ", "RG", "XQ", "XG", "G", "Q"):
                b = re.sub(r"\\" + m + r"\{", r"\\D" + m + "{", b)
        if pre == "d5" and lab == "tab:preds":  # symbols the combined paper's text does not define
            b = b.replace("$\\ell$ is layer 24", "$d$ is the capitals direction and $r$ a random direction of the same norm; $\\ell$ is layer 24", 1)
            assert "$r$ a random direction" in b
        if pre == "d5" and lab == "tab:preds2":
            b = b.replace("\\caption{Round 2: the frozen predictions and their held-out outcome.", "\\caption{Round 2: the frozen predictions and their held-out outcome. $u$ is the unit vector of the capitals direction $d$; U, L and A are the upper-case, lower-case and context-only passages, and $D=\\log p(U)-\\log p(L_c)$ as in \\cref{tab:d5-head}.", 1)
            assert "unit vector of the capitals direction" in b
        for old, new in WORDING.get((pre, lab), []):  # v4: wording of copied captions and row labels (no number changes)
            assert b.count(old) >= 1, (pre, lab, old)
            b = b.replace(old, new)
        (PAPER / "tables" / f"{pre}-{lab[4:]}.tex").write_text("% Layout copied from the source paper by scripts/extract_tables.py; numbers are \\nv macros.\n" + b + "\n")
        print(pre, lab, len(b))

# ---------------------------------------------------------------- X1 matrix tables (layout from the key names only)
keys = set(re.findall(r"nv@(x1m\.m\.[^}]+)\\endcsname", (PAPER / "numbers.tex").read_text()))
NAME = {"COUNT": "COUNT", "WRITE": "WRITE", "READ": "READ", "READnoL17H6": "READ$-$\\hd{17}{6}", "L17H6": "\\hd{17}{6}",
        "CARRY": "\\carry{}", "CARRYnoL17H6": "\\carry{}$-$\\hd{17}{6}", "TERM": "\\term{}", "TERMnoL17H6": "\\term{}$-$\\hd{17}{6}",
        "DED": "\\ded{}", "L0H2": "\\hd{0}{2}", "L0H3": "\\hd{0}{3}", "L11H9": "\\hd{11}{9}"}
ORDER = list(NAME)


def cell(g, c, rng=False):
    k = f"x1m.m.{g}.{c}"
    if k not in keys: return "--"
    s = f"\\nv{{{k}}}\\xo{{{g}}}{{{c}}}"  # v4: \xo marks a value outside the range of its random groups
    return s + (f" {{\\scriptsize(\\nv{{{k}.rlo}} to \\nv{{{k}.rhi}})}}" if rng else "")


def groups(codes): return [g for g in ORDER if any(f"x1m.m.{g}.{c}" in keys for c in codes)]


# main-text summary
cols = [("LBw", "width"), ("LBs", "sep."), ("cU", "code unm."), ("gT", "gen.\\ term."), ("t4", "teen"), ("p4", "pull"), ("Du", "$D_{\\text{up}}$"), ("Dl", "$D_{\\text{low}}$")]
rows = [g for g in ["COUNT", "WRITE", "READnoL17H6", "L17H6", "CARRY", "TERM", "DED", "L0H2", "L0H3", "L11H9"]]
out = ["% Generated by scripts/extract_tables.py from the key names in numbers.tex; numbers are \\nv macros.",
       "\\begin{tabular}{@{}l cc cc cc cc@{}}", "\\toprule",
       " & \\multicolumn{2}{c}{Line-breaking (kept)} & \\multicolumn{2}{c}{Indentation (change)} & \\multicolumn{2}{c}{Ordinals, 4 digits (kept)} & \\multicolumn{2}{c}{Capitals (kept)} \\\\",
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
       "Group & " + " & ".join(n for _, n in cols) + " \\\\", "\\midrule"]
for g in rows:
    out.append(NAME[g] + " & " + " & ".join(cell(g, c) for c, _ in cols) + " \\\\")
out += ["\\bottomrule", "\\end{tabular}"]
(PAPER / "tables" / "x1-summary.tex").write_text("\n".join(out) + "\n")
# appendix: full matrices with random ranges
for tag, title, codes, heads in (
        ("LB", "Line-breaking, \\qsix{} (effect kept)", ["LBw", "LBs"], ["width effect", "separation"]),
        ("IND", "Indentation, \\qsix{} (change in mean log-probability of the right indent)", ["cU", "cT", "gU", "gT"], ["code, unmarked", "code, terminal", "generated, unmarked", "generated, terminal"]),
        ("ORD", "Ordinal suffixes, \\qsix{} (effect kept)", ["t4", "t5", "p4", "p5"], ["teen, 4 digits", "teen, 5 digits", "pull, 4 digits", "pull, 5 digits"]),
        ("CAPS", "Capitals, \\qbig{}, heads by index ($D$ kept; word log-probability change)", ["Du", "Dl", "wl"], ["$D_{\\text{up}}$ kept", "$D_{\\text{low}}$ kept", "word log-prob.\\ change"])):
    out = ["% Generated by scripts/extract_tables.py from the key names in numbers.tex; numbers are \\nv macros.",
           f"\\paragraph{{{title}.}}", "\\begin{center}\\footnotesize\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}l" + " l" * len(codes) + "@{}}", "\\toprule",
           "Group & " + " & ".join(heads) + " \\\\", "\\midrule"]
    for g in groups(codes):
        out.append(NAME[g] + " & " + " & ".join(cell(g, c, True) for c in codes) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    (PAPER / "tables" / f"x1-{tag}.tex").write_text("\n".join(out) + "\n")
print("x1 tables written")
