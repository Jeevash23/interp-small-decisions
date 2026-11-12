"""D5 round 2 report builder (post hoc reporting tool, written 8 Nov 2026 after the confirmation run; changes no data).
Fills every number in code/report2_template.md from outputs/r2/conf_verdicts2.json (held out) and
outputs/r2/explore_verdicts2_check.json (exploration) and writes REPORT-2.md.
Placeholders: {{src:path|fmt}}
  src   q / g   = held-out measures of Qwen3-1.7B / Gemma ('P.' prefix: predictions; 'V.' prefix: verdicts)
        eq / eg = exploration measures, same layout
  path  dot-separated keys or list indices; 'min' / 'max' of a list
  fmt   (none) = point estimate to 3 decimals; ci = point [lo, hi]; n = count; 2 = 2 decimals; pf = passes/fails; raw
Fails if any placeholder is left or any value is missing."""
import json, re
from pathlib import Path

D = Path(__file__).resolve().parent.parent; R2 = D / "outputs" / "r2"
C = json.load(open(R2 / "conf_verdicts2.json")); E = json.load(open(R2 / "explore_verdicts2_check.json"))
assert C["mode"] == "conf" and E["mode"] == "explore"
SRC = {"q": C["models"]["qwen3-17b"], "g": C["models"]["gemma"], "eq": E["models"]["qwen3-17b"], "eg": E["models"]["gemma"]}


def get(src, path):
    cur = SRC[src]
    if path.startswith("P."): cur, path = cur["predictions"], path[2:]
    elif path.startswith("V."): cur, path = cur["verdicts"], path[2:]
    else: cur = cur["measures"]
    for p in path.split("."):
        if p == "min": cur = min(cur)
        elif p == "max": cur = max(cur)
        elif isinstance(cur, list): cur = cur[int(p)]
        else: cur = cur[p]
    return cur


def fmt(v, f):
    if f == "ci": return f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
    if f == "n": return f"{v['n']:,}"
    if f == "pf": return "passes" if v is True else ("fails" if v is False else str(v))
    if f == "raw": return str(v)
    if isinstance(v, dict): v = v["point"]
    if f == "2": return f"{v:.2f}"
    if isinstance(v, bool): return str(v)
    if isinstance(v, int): return f"{v:,}"
    return f"{v:.3f}".replace("-", "−")


tpl = (D / "code" / "report2_template.md").read_text(); n = 0
def sub(mo):
    global n; n += 1
    path, _, f = mo.group(2).partition("|")
    return fmt(get(mo.group(1), path), f)
out = re.sub(r"\{\{(\w+):([^}]+)\}\}", sub, tpl)
assert "{{" not in out and "}}" not in out, "unfilled placeholder"
(D / "REPORT-2.md").write_text(out)
print(f"REPORT-2.md written, {n} numbers filled from saved outputs")
