"""D3 round 2, r2e10 (exploratory, question 5): are the released OLMo-2-0425-1B weights the mean of the three stage-2
ingredients' final checkpoints? Compares tensor by tensor (lazy loading), and reports how far the released weights sit
from the mean and from each ingredient. Writes outputs/r2/r2e10_merge.json.
Usage: python r2e10_merge.py"""
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open
hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--OLMo-2-0425-1B"
snap = lambda ref: hub / "snapshots" / (hub / "refs" / ref).read_text().strip()
def tensors(d):
    out = {}
    for f in sorted(d.glob("*.safetensors")):
        h = safe_open(str(f), "pt")
        for k in h.keys(): out[k] = (h, k)
    return out
R = tensors(snap("main"))
I = [tensors(snap(f"stage2-ingredient{i}-step23852-tokens51B")) for i in (1, 2, 3)]
num = dict(mean=0.0, i1=0.0, i2=0.0, i3=0.0); den = 0.0; worst = {}
for k in R:
    r = R[k][0].get_tensor(k).float(); xs = [t[k][0].get_tensor(k).float() for t in I]; m = sum(xs) / 3
    den += float((r ** 2).sum()); num["mean"] += float(((r - m) ** 2).sum())
    for j, x in enumerate(xs): num[f"i{j+1}"] += float(((r - x) ** 2).sum())
    worst[k] = float((r - m).abs().max())
res = dict(rel_dist={k: (v / den) ** 0.5 for k, v in num.items()}, max_abs_released_minus_mean=max(worst.values()),
           worst_tensor=max(worst, key=worst.get), n_tensors=len(R))
D3 = Path(__file__).resolve().parent.parent
(D3 / "outputs" / "r2" / "r2e10_merge.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
