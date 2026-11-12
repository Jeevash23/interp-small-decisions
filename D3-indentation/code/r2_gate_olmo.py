"""D3 round 2 (question 4): OLMo-2-0425-1B gate summary with round 1's frozen gate rules (code/gate_summary.py,
imported unchanged), written to outputs/r2/gate_olmo1b.json instead of overwriting outputs/gate_summary.json.
Reads outputs/e01_olmo1b_{newcode,gen}.jsonl (round 1's frozen e01_behaviour.py, 8 Nov). No model is loaded."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_summary as G
r = {}
nc = G.rows("olmo1b", "newcode"); r["newcode"] = G.stats(nc); g = G.rows("olmo1b", "gen")
for w in ["2", "3", "4", "mixed"]: r[f"gen_w{w}"] = G.stats([x for x in g if str(x["width"]) == w])
s = r["newcode"]
r["gate"] = dict(item1=s["header_top1"] >= 0.90, item2=(s["terminal_shopen"] >= 0.7 and s["terminal_top1"] >= 0.5),
                 item3=(s["top1"] - s["copy"]) >= 0.10, open_level=max(r[f"gen_w{w}"]["dedent_shnotopen"] for w in ["2", "3", "mixed"]) <= 0.10)
out = G.OUT / "r2" / "gate_olmo1b.json"; out.write_text(json.dumps({"olmo1b": r}, indent=1))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), r["gate"])
print({w: round(r[f"gen_w{w}"]["dedent_shnotopen"], 3) for w in ["2", "3", "4", "mixed"]})
