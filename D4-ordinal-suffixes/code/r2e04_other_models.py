"""R2 exploration E04 (CHARTER-2.md question 4): does the round-1 account carry to other one-token-per-digit
models? Heads are chosen by attention alone, then ablated.
Usage: r2e04_other_models.py <model> [--per 96] [--gate]
  --gate: round-1 gate rule on exploration items (G1 dates 1-31 >= 0.90; G2 32-999 ending 1-3, not 11-13,
          non-date families >= 0.80); the analysis runs only if both hold.
Items: 96 teen and 96 non-teen (ending 1-3) at 3 and 4 digits, exploration frames and numbers.
Candidate sets from attention at the last token (mean over 3- and 4-digit items), layers 0-1 only:
  PREV = heads attending >= 0.5 to the previous token (the tens digit); CUR = heads attending >= 0.5 to the
  current token; also the single top PREV head and top CUR head.
Mean ablation at the last token; kept teen effect and kept last-digit pull; 20 random sets of the same size
from layers 0-1 outside PREV and CUR.
Writes outputs/r2/e04_<model>.json."""
import sys, time, json, functools
print = functools.partial(print, flush=True)
import torch; torch.set_grad_enabled(False)
from r2lib import *

NAME = sys.argv[1]
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
PER, NR = int(arg("--per", "96")), int(arg("--random", "20"))
M = Hooked(NAME); L, Hn = M.L, M.H
res = dict(model=NAME); t0 = time.time()
if "--gate" in sys.argv:
    fr = r2_frames("explore")
    g1 = [dict(prefix=f["text"].replace("{N}", str(n)), n=n, cls="x") for f in fr if f["family"] == "dates" for n in range(1, 32)]
    rng = random.Random("d4r2-e04-gate")
    nd = [f for f in fr if f["family"] != "dates"]
    g2 = []
    while len(g2) < 600:
        n = rng.randint(32, 999)
        if n % 10 in (1, 2, 3) and n % 100 not in (11, 12, 13) and r2_number_ok(n, "explore"):
            f = rng.choice(nd); g2.append(dict(prefix=f["text"].replace("{N}", str(n)), n=n, cls="x"))
    acc = lambda its: float(np.mean(np.argmax(M.run(its), 1) == np.array([SUFFIXES.index(suffix(i["n"])) for i in its])))
    res["gate"] = dict(G1=acc(g1), G2=acc(g2), n1=len(g1), n2=len(g2))
    res["gate"]["qualifies"] = res["gate"]["G1"] >= 0.90 and res["gate"]["G2"] >= 0.80
    print("gate", res["gate"])
    if not res["gate"]["qualifies"]:
        json.dump(res, open(OUT2 / f"e04_{NAME}.json", "w"), default=float); sys.exit(0)
IT = {d: make_items(d, PER, f"d4r2-e04-{d}") for d in (3, 4)}
# attention at the last token, layers 0-1
att = np.zeros((2, Hn, 2)); cnt = 0
for d, items in IT.items():
    for i in range(0, len(items), 48):
        M.clear(); ids, am, last = M.encode_batch([it["prefix"] for it in items[i:i + 48]]); M.last = last
        o = M.model.base_model(input_ids=ids, attention_mask=am, output_attentions=True)
        r = torch.arange(len(last), device=ids.device)
        for l in range(2):
            w = o.attentions[l].float()
            att[l, :, 0] += w[r, :, last, last].sum(0).cpu().numpy(); att[l, :, 1] += w[r, :, last, last - 1].sum(0).cpu().numpy()
        cnt += len(last)
att /= cnt
res["att_l01_self_prev"] = att.tolist()
PREV = [(l, h) for l in range(2) for h in range(Hn) if att[l, h, 1] >= 0.5]
CUR = [(l, h) for l in range(2) for h in range(Hn) if att[l, h, 0] >= 0.5]
topP = max(((att[l, h, 1], (l, h)) for l in range(2) for h in range(Hn)))[1]
topC = max(((att[l, h, 0], (l, h)) for l in range(2) for h in range(Hn)))[1]
sets = {"PREV": PREV, "CUR": CUR, "topPREV": [topP], "topCUR": [topC]}
res["sets"] = {k: [f"L{l}H{h}" for l, h in v] for k, v in sets.items()}
print("sets", res["sets"])
pool = [(l, h) for l in range(2) for h in range(Hn) if (l, h) not in PREV and (l, h) not in CUR]
res["digits"] = {}
for d, items in IT.items():
    M.clear(); clean = M.run(items, cap=("oin",)); means = {l: v.mean(0) for l, v in M.c["oin"].items()}
    base = summary(clean, items); R = dict(base=base, sets={}, random={})
    for k, S in sets.items():
        if not S: continue
        M.clear(); M.ablate_heads(S, means); R["sets"][k] = summary(M.run(items), items)
        rng = random.Random(f"d4r2-e04-rand-{NAME}-{d}-{k}")
        rr = []
        for _ in range(NR):
            if len(S) > len(pool): break
            M.clear(); M.ablate_heads(rng.sample(pool, len(S)), means); rr.append(summary(M.run(items), items))
        R["random"][k] = rr
    # single heads in layers 0-1
    R["single"] = {}
    for l in range(2):
        for h in range(Hn):
            M.clear(); M.ablate_heads([(l, h)], means); R["single"][f"L{l}H{h}"] = summary(M.run(items), items)
    M.clear()
    res["digits"][str(d)] = R
    print(f"== {d} digits ({time.time() - t0:.0f}s) teen {base['teen']:+.2f} pull {base['pull']:+.2f} auroc {base['auroc']:.3f}")
    for k, v in R["sets"].items():
        rr = R["random"][k]
        rt = [x["teen"] / base["teen"] for x in rr]; rp = [x["pull"] / base["pull"] for x in rr]
        print(f"  {k} {res['sets'][k]}: teen kept {v['teen'] / base['teen']:.2f} (random {min(rt, default=float('nan')):.2f}-{max(rt, default=float('nan')):.2f}); pull kept {v['pull'] / base['pull']:.2f} (random {min(rp, default=float('nan')):.2f}-{max(rp, default=float('nan')):.2f})")
    top = sorted((v["teen"] / base["teen"], k) for k, v in R["single"].items())[:6]
    topp = sorted((v["pull"] / base["pull"], k) for k, v in R["single"].items())[:6]
    print("  single, teen kept:", " ".join(f"{k}:{v:.2f}(prev {att[int(k[1]), int(k.split('H')[1]), 1]:.2f})" for v, k in top))
    print("  single, pull kept:", " ".join(f"{k}:{v:.2f}(self {att[int(k[1]), int(k.split('H')[1]), 0]:.2f})" for v, k in topp))
json.dump(res, open(OUT2 / f"e04_{NAME}.json", "w"), default=float)
print(f"done {time.time() - t0:.0f}s")
