"""Print the gate table for NOTEBOOK.md from outputs/gate_summary_<model>.json (no model weights)."""
import json
from common import OUT
M=[("qwen3-06b","Qwen3-0.6B-Base"),("pythia-410m","Pythia-410M"),("pythia-1b","Pythia-1B"),("olmo2-1b","OLMo-2-1B"),
   ("smollm2-17b","SmolLM2-1.7B"),("gemma4-e2b","Gemma-4-E2B"),("gpt2","GPT-2 Small")]
print("| Model | G1 | G2 | Qualifies | Teens 2-digit | Teens 3-digit | Teens 4-digit | 4-digit ending 1/2/3 |")
print("|---|---|---|---|---|---|---|---|")
for k,lab in M:
    o=json.load(open(OUT/f"gate_summary_{k}.json")); g=o["_gate"]
    t=lambda d:o[f"ending=teen|digits={d}"]["acc"]
    e4=sum(o[f"ending={e}|digits=4"]["acc"]*o[f"ending={e}|digits=4"]["n"] for e in "123")/sum(o[f"ending={e}|digits=4"]["n"] for e in "123")
    print(f"| {lab} | {g['G1']:.3f} | {g['G2']:.3f} | {'yes' if g['qualifies'] else 'no'} | {t(2):.3f} | {t(3):.3f} | {t(4):.3f} | {e4:.3f} |")
