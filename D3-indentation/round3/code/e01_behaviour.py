"""D3 e01 (exploratory; the gate of CHARTER.md): indent distributions at every decision point of a source's
exploration part. Saves one record per point to outputs/e01_<model>_<source>.jsonl.
Usage: python e01_behaviour.py <model> <source> [max_chunks] [--time-limit S]"""
import sys, json, time
from common import *

name, source = sys.argv[1], sys.argv[2]
nmax = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("-") else 10**9
tlim = float(sys.argv[sys.argv.index("--time-limit") + 1]) if "--time-limit" in sys.argv else 1e9
chunks = load_chunks(source, "explore")[:nmax]
outp = OUT / f"e01_{name}_{source}.jsonl"
done = set()
if outp.exists():
    for l in open(outp): done.add(json.loads(l)["chunk"])
m = M(name, attn="sdpa"); t0 = time.time(); n = 0; skipped = 0
with open(outp, "a") as fh:
    for ci, c in enumerate(chunks):
        if ci in done: continue
        if time.time() - t0 > tlim: print("time limit; resume later"); break
        ids, pts = prepare(m, c["text"])
        if pts is None: skipped += 1; continue
        indent_dist(m, ids, pts)
        for p in pts:
            r = {k: p[k] for k in ("line", "true", "prev", "stack", "cls", "ndedent", "header", "terminal", "prevword", "blank", "word", "K", "mass")}
            r.update(summarise_point(p)); r["chunk"] = ci; r["width"] = c.get("width"); r["src"] = c.get("repo", c.get("file", c.get("seed")))
            fh.write(json.dumps(r) + "\n"); n += 1
        if not pts: fh.write(json.dumps(dict(chunk=ci, empty=True)) + "\n")
print(json.dumps(dict(model=name, source=source, points=n, skipped_long=skipped, secs=round(time.time() - t0))))
