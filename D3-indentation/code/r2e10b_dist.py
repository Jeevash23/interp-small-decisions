"""D3 round 2, r2e10b (exploratory, question 5): relative weight distance ||a-b|| / ||a|| between pairs of OLMo-2-0425-1B
checkpoints (lazy, tensor by tensor), to see which checkpoints descend from which. Writes outputs/r2/r2e10b_dist.json. Sums in float64.
Also the cosine of the token embeddings alone."""
import json
from pathlib import Path
from safetensors import safe_open
hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--allenai--OLMo-2-0425-1B"
snap = lambda ref: hub / "snapshots" / (hub / "refs" / ref).read_text().strip()
def T(ref):
    out = {}
    for f in sorted(snap(ref).glob("*.safetensors")):
        h = safe_open(str(f), "pt")
        for k in h.keys(): out[k] = h
    return out
refs = dict(final="main", s1_1007B="stage1-step480000-tokens1007B", s1_4001B="stage1-step1907359-tokens4001B",
            s2i1_3B="stage2-ingredient1-step1000-tokens3B", s2i1_51B="stage2-ingredient1-step23852-tokens51B", s2i2_51B="stage2-ingredient2-step23852-tokens51B")
H = {k: T(v) for k, v in refs.items()}
EMB = "model.embed_tokens.weight"
pairs = [("final", "s1_4001B"), ("final", "s1_1007B"), ("s1_1007B", "s1_4001B"), ("s1_4001B", "s2i1_3B"), ("s1_4001B", "s2i1_51B"),
         ("s2i1_51B", "s2i2_51B")]
res = {}
for a, b in pairs:
    num = den = 0.0; dot = 0.0; nb = 0.0
    for k in H[a]:
        X = H[a][k].get_tensor(k).flatten(); Y = H[b][k].get_tensor(k).flatten()
        for s in range(0, X.numel(), 1 << 22):  # float64 in chunks (float32 sums over 2e8 elements drift)
            x = X[s:s + (1 << 22)].double(); y = Y[s:s + (1 << 22)].double()
            num += float(((x - y) ** 2).sum()); den += float((x ** 2).sum()); dot += float((x * y).sum()); nb += float((y ** 2).sum())
    X = H[a][EMB].get_tensor(EMB).flatten(); Y = H[b][EMB].get_tensor(EMB).flatten(); e = [0.0, 0.0, 0.0]
    for s in range(0, X.numel(), 1 << 22):
        x = X[s:s + (1 << 22)].double(); y = Y[s:s + (1 << 22)].double(); e[0] += float((x * y).sum()); e[1] += float((x * x).sum()); e[2] += float((y * y).sum())
    res[f"{a}~{b}"] = dict(rel_dist=(num / den) ** 0.5, cosine=dot / (den * nb) ** 0.5, emb_cosine=e[0] / (e[1] * e[2]) ** 0.5)
    print(a, b, res[f"{a}~{b}"], flush=True)
(Path(__file__).resolve().parent.parent / "outputs" / "r2" / "r2e10b_dist.json").write_text(json.dumps(res, indent=1))
