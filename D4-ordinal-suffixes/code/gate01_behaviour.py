"""D4 gate (CHARTER.md, 'Gate'): suffix choice among st/nd/rd/th at the number's last token, exploration
items only. Exploratory. Usage: gate01_behaviour.py <model> [--smoke]

Item sample (fixed seed, exploration part):
  * every item with N <= 99 (all date and century items, and 1-99 in the other families);
  * every item with N >= 100 ending in 11-13;
  * 3000 random items with 100 <= N <= 999 and 3000 with N >= 1000 (other endings).
Writes outputs/gate01_<model>.jsonl (one line per item: fid, family, n, logits of the four, lse).
"""
import sys, json, random, time
from common import *

name = sys.argv[1]
smoke = "--smoke" in sys.argv
items = load_part("explore")
rng = random.Random("d4-gate01")
small = [it for it in items if it["n"] <= 99]
teen = [it for it in items if it["n"] >= 100 and it["n"] % 100 in (11, 12, 13)]
mid = [it for it in items if 100 <= it["n"] <= 999 and it["n"] % 100 not in (11, 12, 13)]
big = [it for it in items if it["n"] >= 1000 and it["n"] % 100 not in (11, 12, 13)]
sample = small + teen + rng.sample(mid, 3000) + rng.sample(big, 3000)
if smoke:
    sample = rng.sample(sample, 200)
print(f"{name}: {len(sample)} items (small {len(small)}, teen {len(teen)})", flush=True)

t0 = time.time()
tok, model = load(name)
cols = suffix_ids(tok)
bs = 32 if name in ("gemma4-e2b",) else 64
lg, lse = last_logits(model, tok, [it["prefix"] for it in sample], cols, bs=bs)
tag = "_smoke" if smoke else ""
with open(OUT / f"gate01_{name}{tag}.jsonl", "w") as fo:
    for it, l, s in zip(sample, lg, lse):
        fo.write(json.dumps(dict(fid=it["fid"], family=it["family"], n=it["n"],
                                 logits=[round(float(x), 4) for x in l], lse=round(float(s), 4))) + "\n")
acc = np.mean([SUFFIXES[int(np.argmax(l))] == suffix(it["n"]) for it, l in zip(sample, lg)])
print(f"done in {time.time() - t0:.0f}s; overall among-four accuracy {acc:.3f}", flush=True)
