"""D3 round-2 held-out material (CHARTER-2.md, Material). Rule, fixed before any model read the material:
 code: every file of data/r2code/fetched.jsonl is normalised and chunked exactly as in round 1
   (build_material.chunk_records: pyindent.normalise, pyindent.chunks, at least 4 decision points). A chunk is
   dropped if its text hash occurs in any round-1 part (data/newcode, data/gen, data/stdlib) or any X1 part
   (X1-shared-parts/data/code, data/gen), or earlier in this build. All remaining chunks are held-out:
   data/r2code/part-held.jsonl.
 gen: round 1's frozen generator (gen_programs.G) with seeds 200000-203999, width [2, 3, 4, "mixed"][seed % 4],
   first attempt of 400-1,400 characters that parses, as in round 1. All held-out: data/r2gen/part-held.jsonl.
Usage: python r2_build.py code|gen"""
import ast, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyindent as P
from build_material import chunk_records
import gen_programs as GP

D = Path(__file__).resolve().parent.parent / "data"
X1 = Path(__file__).resolve().parents[2] / "X1-shared-parts" / "data"
H = lambda s: hashlib.sha256(s.encode()).hexdigest()


def old_hashes():
    hs = set()
    for f in list(D.glob("newcode/part-*.jsonl")) + list(D.glob("gen/part-*.jsonl")) + list(D.glob("stdlib/part-*.jsonl")) \
            + list(X1.glob("code/part-*.jsonl")) + list(X1.glob("gen/part-*.jsonl")):
        for l in open(f): hs.add(H(json.loads(l)["text"]))
    return hs


def code():
    rows = [json.loads(l) for l in open(D / "r2code" / "fetched.jsonl")]
    old = old_hashes(); seen = set(); out = []
    st = dict(files=len(rows), normalised=0, dup_old=0, dup=0, old_hashes=len(old))
    for r in sorted(rows, key=lambda r: r["key"]):
        src = P.normalise((D / "r2code" / "raw" / f"{r['key']}.py").read_bytes())
        if src is None: continue
        st["normalised"] += 1
        for c in chunk_records(src, dict(repo=r["repo"], path=r["path"], key=r["key"])):
            h = H(c["text"])
            if h in old: st["dup_old"] += 1; continue
            if h in seen: st["dup"] += 1; continue
            seen.add(h); out.append(c)
    with open(D / "r2code" / "part-held.jsonl", "w") as fh:
        for x in out: fh.write(json.dumps(x) + "\n")
    st.update(held_chunks=len(out), held_repos=len({x["repo"] for x in out}))
    (D / "r2code" / "build_summary.json").write_text(json.dumps(st, indent=1)); print(st)


def gen():
    out = []
    for seed in range(200000, 204000):
        width = [2, 3, 4, "mixed"][seed % 4]
        for attempt in range(20):
            src = GP.G(seed * 100 + attempt, width).program()
            if not (400 <= len(src) <= 1400): continue
            try: ast.parse(src)
            except SyntaxError: continue
            break
        else:
            continue
        out.append(dict(seed=seed, attempt=attempt, width=width, text=src))
    (D / "r2gen").mkdir(exist_ok=True)
    with open(D / "r2gen" / "part-held.jsonl", "w") as fh:
        for x in out: fh.write(json.dumps(x) + "\n")
    print(dict(held=len(out)))


if __name__ == "__main__":
    {"code": code, "gen": gen}[sys.argv[1]]()
