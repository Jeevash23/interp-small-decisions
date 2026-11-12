# POST HOC (9 Nov 2026): lineage check of OLMo-2-0425-1B released weights vs published checkpoints, prompted by D3.
# Is the released model the stage-1 model in a permuted or rotated basis? Basis-free comparisons.
import os, glob, json, sys, torch
from safetensors import safe_open
H = os.path.expanduser("~/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B")
def get(ref, k):
    d = os.path.join(H, "snapshots", open(os.path.join(H, "refs", ref)).read().strip())
    for f in glob.glob(d + "/*.safetensors"):
        with safe_open(f, "pt") as s:
            if k in s.keys(): return s.get_tensor(k).float()
out = {}
refs = ["main", "stage1-step1907359-tokens4001B", "stage2-ingredient1-step23852-tokens51B"]
E = {r: get(r, "model.embed_tokens.weight") for r in refs}
n = {r: E[r].norm(dim=1) for r in refs}
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (a.norm()*b.norm()))
out["token_norm_corr_main_vs_s1end"] = round(corr(n["main"], n[refs[1]]), 4)
out["token_norm_corr_s2_vs_s1end"] = round(corr(n[refs[2]], n[refs[1]]), 4)
# Gram matrix of 2000 frequent-ish token ids: invariant to any rotation of the hidden basis
idx = torch.arange(1000, 3000)
G = {r: torch.nn.functional.normalize(E[r][idx], dim=1) @ torch.nn.functional.normalize(E[r][idx], dim=1).T for r in refs}
tri = torch.triu_indices(2000, 2000, 1)
g = {r: G[r][tri[0], tri[1]] for r in refs}
out["gram_corr_main_vs_s1end"] = round(corr(g["main"], g[refs[1]]), 4)
out["gram_corr_s2_vs_s1end"] = round(corr(g[refs[2]], g[refs[1]]), 4)
out["gram_corr_main_vs_s2"] = round(corr(g["main"], g[refs[2]]), 4)
for k in ["model.layers.0.self_attn.q_proj.weight", "model.layers.8.mlp.down_proj.weight"]:
    sv = {r: torch.linalg.svdvals(get(r, k)) for r in refs}
    out["svals_rel_diff_main_vs_s1end:" + k] = round(float((sv["main"] - sv[refs[1]]).norm() / sv[refs[1]].norm()), 4)
    out["svals_rel_diff_s2_vs_s1end:" + k] = round(float((sv[refs[2]] - sv[refs[1]]).norm() / sv[refs[1]].norm()), 4)
out["main_snapshot"] = open(os.path.join(H, "refs", "main")).read().strip()
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[1], "w"), indent=1)
