"""D3 material from natural code (CHARTER.md, Material). Rule, fixed before any model read the code:
 newcode: every fetched file (data/newcode/fetched.jsonl, raw/) is normalised (pyindent.normalise; files
   failing it are dropped) and cut into chunks (pyindent.chunks, at most 1,400 characters, whole top-level
   units, starting at indent 0). Chunks with fewer than 4 decision points are dropped. Exact duplicate
   chunks are kept once (first by key order). Split by repository:
   part = int(sha256("D3-newcode-" + repo), 16) % 2 -> 0 explore, 1 conf.
 stdlib (exploration only): /usr/lib/python3.14 top-level modules and packages, excluding test/, idlelib/,
   site-packages/, lib2to3/, turtledemo/, __pycache__; same normalisation and chunking; at most 3 chunks per
   file, in file order.
Usage: python build_material.py newcode|stdlib"""
import hashlib, json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pathlib import Path
import pyindent as P

D = Path(__file__).resolve().parent.parent / "data"


def chunk_records(src, base, cap=None):
    out = []
    for i, c in enumerate(P.chunks(src)):
        if len(P.decision_points(c)) < 4: continue
        out.append(dict(base, idx=i, text=c))
        if cap and len(out) >= cap: break
    return out


def newcode():
    rows = [json.loads(l) for l in open(D / "newcode" / "fetched.jsonl")]
    parts = {"explore": [], "conf": []}; seen = set(); stats = dict(files=len(rows), normalised=0, chunks=0, dup=0)
    for r in sorted(rows, key=lambda r: r["key"]):
        src = P.normalise((D / "newcode" / "raw" / f"{r['key']}.py").read_bytes())
        if src is None: continue
        stats["normalised"] += 1
        p = int(hashlib.sha256(f"D3-newcode-{r['repo']}".encode()).hexdigest(), 16) % 2
        for c in chunk_records(src, dict(repo=r["repo"], path=r["path"], key=r["key"])):
            h = hashlib.sha256(c["text"].encode()).hexdigest()
            if h in seen: stats["dup"] += 1; continue
            seen.add(h); parts[["explore", "conf"][p]].append(c); stats["chunks"] += 1
    for k, v in parts.items():
        with open(D / "newcode" / f"part-{k}.jsonl", "w") as fh:
            for x in v: fh.write(json.dumps(x) + "\n")
    stats.update({f"{k}_chunks": len(v) for k, v in parts.items()},
                 **{f"{k}_repos": len({x['repo'] for x in v}) for k, v in parts.items()})
    (D / "newcode" / "build_summary.json").write_text(json.dumps(stats, indent=1)); print(stats)


def stdlib():
    SRC = Path("/usr/lib/python3.14"); SKIP = {"test", "tests", "idlelib", "site-packages", "__pycache__", "lib2to3", "turtledemo"}
    out = []; nf = 0
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC).as_posix()
        if set(rel.split("/")[:-1]) & SKIP or rel.startswith("test"): continue
        src = P.normalise(f.read_bytes())
        if src is None: continue
        nf += 1; out += chunk_records(src, dict(file=rel), cap=3)
    (D / "stdlib").mkdir(exist_ok=True)
    with open(D / "stdlib" / "part-explore.jsonl", "w") as fh:
        for x in out: fh.write(json.dumps(x) + "\n")
    s = dict(files=nf, chunks=len(out)); (D / "stdlib" / "build_summary.json").write_text(json.dumps(s)); print(s)


if __name__ == "__main__":
    {"newcode": newcode, "stdlib": stdlib}[sys.argv[1]]()
