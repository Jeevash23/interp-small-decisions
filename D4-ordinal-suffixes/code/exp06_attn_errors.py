"""Exp06 (exploratory): does the attention of candidate heads to the tens digit predict teen errors?
Usage: exp06_attn_errors.py <model> --heads 0.2,1.6 [--max 2000]
Items: the gate's exploration items with N >= 100 ending in 11-13 (same sample rule as gate01), plus as many
non-teen items ending in 1-3. For each item: each head's attention from the last token to the previous token
(the tens digit in one-token-per-digit models), the norm of the head's output at the last position, and
m = logit(last-digit suffix) - logit(th). Writes outputs/exp06_<model>.jsonl and prints correlations.
"""
import sys, random, json
from common import *
from teen_effect import auroc

name = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
HEADS = [tuple(map(int, x.split("."))) for x in arg("--heads", "0.2,1.6").split(",")]
MAX = int(arg("--max", "2000"))
rng = random.Random("d4-exp06")
items = load_part("explore")
teen = [it for it in items if it["n"] >= 100 and it["n"] % 100 in (11, 12, 13)]
end = [it for it in items if it["n"] >= 100 and it["n"] % 10 in (1, 2, 3) and it["n"] % 100 not in (11, 12, 13)]
sample = rng.sample(teen, min(MAX, len(teen))) + rng.sample(end, min(MAX, len(teen)))

tok, model = load(name)
cols = suffix_ids(tok)
by = {}
for i, it in enumerate(sample): by.setdefault(len(encode(tok, it["prefix"])), []).append(i)
rows = [None] * len(sample)
with torch.no_grad():
    for _, idx in by.items():
        for b in range(0, len(idx), 64):
            ii = idx[b:b + 64]
            ids = torch.tensor([encode(tok, sample[i]["prefix"]) for i in ii], device=DEVICE)
            o = model.base_model(input_ids=ids, output_attentions=True)
            lg = unembed(model, o.last_hidden_state[:, -1])[:, cols].cpu().numpy()
            for j, i in enumerate(ii):
                n = sample[i]["n"]
                att = {f"{l}.{h}": float(o.attentions[l][j, h, -1, -2]) for l, h in HEADS}
                rows[i] = dict(n=n, family=sample[i]["family"], fid=sample[i]["fid"], teen=n % 100 in (11, 12, 13),
                               m=float(lg[j][n % 10 - 1] - lg[j][3]), ok=SUFFIXES[int(np.argmax(lg[j]))] == suffix(n), att=att)
with open(OUT / f"exp06_{name}.jsonl", "w") as fo:
    for r in rows: fo.write(json.dumps(r) + "\n")
T = [r for r in rows if r["teen"]]
print(f"{name}: {len(T)} teen items, acc {np.mean([r['ok'] for r in T]):.3f}")
for k in [f"{l}.{h}" for l, h in HEADS]:
    a = np.array([r["att"][k] for r in T]); ok = np.array([r["ok"] for r in T]); m = np.array([r["m"] for r in T])
    print(f"  head {k}: attention to tens digit, correct {a[ok].mean():.3f} vs wrong {a[~ok].mean():.3f}; "
          f"AUROC(att, correct) {auroc(a[ok], a[~ok]):.3f}; corr(att, -m) {np.corrcoef(a, -m)[0, 1]:+.3f}")
    for fam in sorted(set(r["family"] for r in T)):
        s = [r for r in T if r["family"] == fam]
        print(f"     {fam:14s} n={len(s):4d} acc={np.mean([r['ok'] for r in s]):.3f} att={np.mean([r['att'][k] for r in s]):.3f}")
