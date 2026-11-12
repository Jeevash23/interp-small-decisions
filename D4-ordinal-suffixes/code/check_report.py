"""Check every number in REPORT.md against the saved outputs.
Usage: check_report.py [REPORT.md]
A number passes if it equals, at the precision it is written with, a number in outputs/conf/EVALUATION.json, in an
exploration output the report cites (outputs/explore/conf01_*_summary.json, outputs/exp05_*.json, outputs/exp04_*.json),
or a threshold, count or date written in PROTOCOL.md. Numbers inside back-ticks (file names, commands) and in
lines starting with '<!-- nocheck -->' are skipped. Prints every number that fails and exits 1 if any does.
"""
import sys, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
rep = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "REPORT.md"

pool = set()


def walk(x):
    if isinstance(x, bool) or x is None:
        return
    if isinstance(x, (int, float)):
        pool.add(float(x)); pool.add(-float(x)); return
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v)
            if isinstance(k, str) and re.fullmatch(r"-?\d+(\.\d+)?", k): pool.add(float(k))
    elif isinstance(x, (list, tuple)):
        for v in x: walk(v)


files = [OUT / "conf" / "EVALUATION.json"] + sorted((OUT / "explore").glob("conf01_*_summary.json")) \
    + sorted(OUT.glob("exp05_*.json")) + [OUT / "explore" / "EVALUATION.json"]
for f in files:
    if f.exists(): walk(json.load(open(f)))
for f in sorted(OUT.glob("exp04_*.json")):
    d = json.load(open(f))
    walk({k: v for k, v in d.items() if k in ("n", "n_valid", "src_ok", "tgt_ok")})   # not the full head/MLP grids: too many values to check against
# protocol thresholds, counts and dates
for t in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", (ROOT / "PROTOCOL.md").read_text()):
    pool.add(float(t))
vals = sorted(pool)


def matches(tok):
    dec = len(tok.split(".")[1]) if "." in tok else 0
    v = float(tok); h = 0.5 * 10 ** -dec + 1e-9
    import bisect
    i = bisect.bisect_left(vals, v - h)
    return i < len(vals) and vals[i] <= v + h


bad = []; n = 0
for ln, line in enumerate(rep.read_text().splitlines(), 1):
    if line.startswith("<!-- nocheck -->"): continue
    line = re.sub(r"`[^`]*`", "", line)
    line = re.sub(r"\b(P|Q|B|O|D|L|H|MLP|v)\d+\b", "", line)      # labels such as Q1, L0H2, MLP24, D4
    line = re.sub(r"L\d+H\d+", "", line)
    for m in re.finditer(r"(?<![\w.])[−-]?\d[\d,]*(?:\.\d+)?", line):
        tok = m.group().replace("−", "-").replace(",", "")
        n += 1
        if not matches(tok): bad.append((ln, m.group(), line.strip()[:100]))
print(f"{n} numbers checked, {len(bad)} unmatched")
for b in bad: print(f"  line {b[0]}: {b[1]}   | {b[2]}")
sys.exit(1 if bad else 0)
