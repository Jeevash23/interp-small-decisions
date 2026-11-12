# POST HOC (9 Nov 2026): lineage check of OLMo-2-0425-1B released weights vs published checkpoints, prompted by D3.
# D4 post hoc lineage check (independent of D3's code). CPU, float64 reductions.
import json, os, sys, glob
from safetensors import safe_open
import torch
H = os.path.expanduser("~/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B")
def snap(ref):
    return os.path.join(H, "snapshots", open(os.path.join(H, "refs", ref)).read().strip())
def tensors(ref, names):
    out = {}
    for f in sorted(glob.glob(os.path.join(snap(ref), "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if names is None or k in names:
                    out[k] = s.get_tensor(k).to(torch.float32)
    return out
refs = ["main", "stage1-step1907359-tokens4001B", "stage1-step480000-tokens1007B",
        "stage2-ingredient1-step1000-tokens3B", "stage2-ingredient1-step23852-tokens51B",
        "stage2-ingredient2-step23852-tokens51B", "stage2-ingredient3-step23852-tokens51B"]
keys = ["model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.o_proj.weight",
        "model.layers.8.mlp.down_proj.weight", "model.layers.15.self_attn.v_proj.weight"]
def _d(a,b): return sum(float((x.double()*y.double()).sum()) for x,y in zip(a.split(8192),b.split(8192)))
def cos(a, b): return _d(a,b)/(_d(a,a)*_d(b,b))**0.5
def rel(a, b): return (_d(a,a)-2*_d(a,b)+_d(b,b))**0.5/_d(b,b)**0.5
res = {}
base = "stage1-step1907359-tokens4001B"
B = tensors(base, set(keys)); M = tensors("main", set(keys))
for r in refs:
    if r == base: continue
    X = M if r == "main" else tensors(r, set(keys))
    res[r + "_vs_stage1end"] = {k: {"cos": round(cos(X[k], B[k]), 4), "rel": round(rel(X[k], B[k]), 4)} for k in keys}
    if r in refs[4:]:
        res["main_vs_" + r] = {k: round(cos(M[k], X[k]), 4) for k in keys}
    if X is not M: del X
    print(r, "done", flush=True)
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
