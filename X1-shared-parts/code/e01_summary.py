"""X1 e01 summary (exploratory, no model): reads outputs/e01_<model>_<scope>.json and prints/saves, for every
named group and measure, the group's value, the random draws' range and whether the group lies beyond every draw.
Measures: ratios (ablated / base) for effects that ablation should shrink; differences (ablated - base) for log-probs
and losses. Usage: e01_summary.py <model> <scope>  -> outputs/e01_summary_<model>_<scope>.json and .md"""
import sys, json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "outputs"
model, scope = sys.argv[1], sys.argv[2]
R = json.loads((OUT / f"e01_{model}_{scope}.json").read_text())
B = R["base"]
MEAS = [  # (task, key, kind)
    ("LB", "width_effect", "ratio"), ("LB", "sep", "ratio"),
    ("IND", "code|dedent|lq", "diff"), ("IND", "code|term|lq", "diff"), ("IND", "code|same|lq", "diff"),
    ("IND", "gen|dedent|lq", "diff"), ("IND", "gen|term|lq", "diff"), ("IND", "gen|same|lq", "diff"),
    ("ORD", "d4|teen_effect", "ratio"), ("ORD", "d5|teen_effect", "ratio"), ("ORD", "d4|pull", "ratio"), ("ORD", "d5|pull", "ratio"),
    ("CAPS", "D_up", "ratio"), ("CAPS", "D_low", "ratio"), ("CAPS", "word_lp", "diff"), ("CAPS", "gen_loss", "diff"),
]


def val(run, t, k, kind):
    if t not in run or t not in B or run[t].get(k) is None or B[t].get(k) is None: return None
    v, b = run[t][k], B[t][k]
    if v != v or b != b: return None
    return v / b if kind == "ratio" else v - b


groups = [k for k in R if k != "base" and "~r" not in k]
out = {}; lines = [f"# e01 summary: {model}, scope {scope}", "", "Ratio measures: ablated/base (1 = untouched). Diff measures: ablated − base.",
                   "`*` = beyond every random draw in the harmful direction; `rand` = min..max of the draws.", ""]
cols = [(t, k, kind) for t, k, kind in MEAS if t in B and B[t].get(k) is not None and B[t].get(k) == B[t].get(k)]
lines.append("| group | " + " | ".join(f"{t}:{k}" for t, k, _ in cols) + " |")
lines.append("|---" * (len(cols) + 1) + "|")
for g in groups:
    rr = [R[k] for k in R if k.startswith(g + "~r")]
    row = {}; cells = []
    for t, k, kind in cols:
        v = val(R[g], t, k, kind); rv = [x for x in (val(r, t, k, kind) for r in rr) if x is not None]
        harm_low = not (kind == "diff" and k == "gen_loss")  # for gen_loss, harm = increase
        beyond = None
        if v is not None and rv:
            beyond = bool(v < min(rv)) if harm_low else bool(v > max(rv))
        row[f"{t}:{k}"] = dict(value=v, rand_min=min(rv) if rv else None, rand_max=max(rv) if rv else None,
                               rand_mean=float(np.mean(rv)) if rv else None, n_rand=len(rv), beyond=beyond)
        cells.append("—" if v is None else f"{v:.2f}{'*' if beyond else ''} ({min(rv):.2f}..{max(rv):.2f})" if rv else f"{v:.2f}")
    out[g] = row; lines.append(f"| {g} | " + " | ".join(cells) + " |")
base = {f"{t}:{k}": B[t][k] for t, k, _ in cols}
(OUT / f"e01_summary_{model}_{scope}.json").write_text(json.dumps(dict(base=base, groups=out), indent=1))
lines += ["", "Base values: " + ", ".join(f"{k} {v:.3f}" for k, v in base.items())]
(OUT / f"e01_summary_{model}_{scope}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
