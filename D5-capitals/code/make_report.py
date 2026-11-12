"""D5 report builder (post hoc reporting tool, written 8 Nov 2026 after the confirmation runs; changes no data).
Fills every number in code/report_template.md from saved outputs and writes REPORT.md.
Placeholders: {{src:path|fmt}}
  src  g / q    = outputs/conf_verdicts.json, Gemma / Qwen3 (measures; 'P.' prefix = predictions)
       eg / eq  = outputs/explore_verdicts_check.json, same layout (exploration values)
       lg / lq  = outputs/conf_exp01_<m>.json, key lens_upper / lens_lower ('up.' / 'lo.' prefix), per-layer lists
       x8g / x8q = outputs/conf_exp08_<m>.json summary
  path dot-separated keys or list indices
  fmt  (none) = point estimate or value to 3 decimals; ci = point [lo, hi]; n; np = paragraphs; 2 = 2 decimals;
       int = integer; med = median (one decimal when it falls between two ranks); pf = pass/fail; raw = as stored
Fails if any placeholder is left or any source value is missing."""
import json, re, sys
from pathlib import Path

D = Path(__file__).resolve().parent.parent; O = D / "outputs"
C = json.load(open(O / "conf_verdicts.json")); E = json.load(open(O / "explore_verdicts_check.json"))
assert C["prefix"] == "conf_"
SRC = {"g": C["models"]["gemma"], "q": C["models"]["qwen3-17b"],
       "eg": E["models"]["gemma"], "eq": E["models"]["qwen3-17b"]}
CALC = {}
LAYER = {"gemma": 24, "qwen3-17b": 22}
for k, m in (("lg", "gemma"), ("lq", "qwen3-17b")):
    x = json.load(open(O / f"conf_exp01_{m}.json")); SRC[k] = {"up": x["lens_upper"], "lo": x["lens_lower"]}
    # check of L2's two shares: how many positions pass in each run, and at how many the two runs agree
    l = LAYER[m]; a = [rc["upper"]["lens"][l][1] > rc["upper"]["lens"][l][2] for rc in x["records"]]
    b = [rc["lower"]["lens"][l][1] > rc["lower"]["lens"][l][2] for rc in x["records"]]
    CALC[f"{k}_n"] = len(a); CALC[f"{k}_pass_upper"] = sum(a); CALC[f"{k}_pass_lower"] = sum(b)
    CALC[f"{k}_agree"] = sum(p == q for p, q in zip(a, b))
    del x
# material: conf paragraphs and articles
rows = [json.loads(s) for s in open(D / "data" / "part-conf.jsonl")]
CALC["conf_paragraphs"] = len(rows); CALC["conf_articles"] = len({r["pageid"] for r in rows})
# largest gap between conf and exploration point estimates, over all shared measures
gaps = []
for mk, ek, name in (("g", "eg", "Gemma"), ("q", "eq", "Qwen3")):
    for key, v in SRC[mk]["measures"].items():
        e = SRC[ek]["measures"].get(key)
        if isinstance(v, dict) and "point" in v and isinstance(e, dict) and "point" in e:
            gaps.append((abs(v["point"] - e["point"]), name, key))
gaps.sort(reverse=True)
CALC["maxgap"] = gaps[0][0]; CALC["maxgap_model"] = gaps[0][1]; CALC["maxgap_key"] = gaps[0][2]
CALC["gap_over_005"] = sum(g[0] > 0.05 for g in gaps); CALC["n_gaps"] = len(gaps)
SRC["calc"] = CALC
print("largest conf-exploration gaps:", [(round(g, 3), n, k) for g, n, k in gaps[:5]])
for k, m in (("x8g", "gemma"), ("x8q", "qwen3-17b")):
    SRC[k] = json.load(open(O / f"conf_exp08_{m}.json"))["summary"]


def get(src, path):
    cur = SRC[src]
    if src in ("g", "q", "eg", "eq"):
        cur = cur["predictions"] if path.startswith("P.") else cur["measures"]
        if path.startswith("P."): path = path[2:]
    for p in path.split("."):
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    return cur


def fmt(v, f):
    if f == "ci": return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
    if f == "n": return f"{v['n']:,}"
    if f == "np": return str(v["n_para"])
    if f == "pf": return "passes" if v is True else ("fails" if v is False else str(v))
    if f == "raw": return str(v)
    if f == "med": return f"{v:,.0f}" if v == int(v) else f"{v:,.1f}"
    if isinstance(v, dict): v = v["point"]
    if f == "int": return f"{int(v):,}"
    if f == "2": return f"{v:.2f}"
    if isinstance(v, bool): return str(v)
    if isinstance(v, int): return f"{v:,}"
    return f"{v:.3f}"


tpl = (D / "code" / "report_template.md").read_text()
n = 0
def sub(mo):
    global n; n += 1
    src, rest = mo.group(1), mo.group(2)
    path, _, f = rest.partition("|")
    return fmt(get(src, path), f)
out = re.sub(r"\{\{(\w+):([^}]+)\}\}", sub, tpl)
assert "{{" not in out and "}}" not in out, "unfilled placeholder"
(D / "REPORT.md").write_text(out)
print(f"REPORT.md written, {n} numbers filled from saved outputs")
