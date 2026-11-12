"""Post hoc helper (written 8 Nov after the confirmation run; not part of the frozen protocol).
Checks the REPORT.md numbers that come from exploration outputs outside check_report.py's pool, on the lines
marked '<!-- nocheck -->', then runs the frozen check_report.py. Exits 1 if anything fails."""
import json, re, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT / "outputs"
g = json.load(open(OUT / "gate_summary_pythia-410m.json"))["_gate"]
R = [json.loads(l) for l in open(OUT / "exp06_qwen3-06b.jsonl")]; T = [r for r in R if r["teen"]]
fam = lambda f: float(np.mean([r["ok"] for r in T if r["family"] == f]))
expected = {"0.006": 0.80 - g["G2"], "0.755": fam("rankings"), "0.973": fam("anniversaries")}
txt = (ROOT / "REPORT.md").read_text(); bad = 0
for k, v in expected.items():
    ok = abs(float(k) - v) <= 0.0005 + 1e-9 and any(l.startswith("<!-- nocheck -->") and k in l for l in txt.splitlines())
    print(f"{k}: saved {v:.4f} {'ok' if ok else 'MISMATCH or not on a nocheck line'}"); bad += not ok
for l in txt.splitlines():
    if l.startswith("<!-- nocheck -->"):
        body = re.sub(r"`[^`]*`", "", l)
        for t in re.findall(r"(?<![\w.:])\d+\.\d+", body):
            if t not in expected: print(f"unverified number on a nocheck line: {t}"); bad += 1
r = subprocess.run([sys.executable, str(ROOT / "code" / "check_report.py")]); bad += r.returncode
sys.exit(1 if bad else 0)
