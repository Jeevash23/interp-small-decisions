"""D3 round 3 held-out material (CHARTER-3.md, Material). Rule, fixed in the charter before any build:
 code: every file of round3/data/code/fetched.jsonl, in key order, is normalised and chunked exactly as rounds 1
   and 2 (build_material.chunk_records: pyindent.normalise, pyindent.chunks, at least 4 decision points). A chunk
   is dropped if its text hash occurs in any earlier D3 part (data/newcode, gen, stdlib, r2code, r2gen, r2gen2) or
   X1 part (code, gen), or earlier in this build. All remaining chunks: round3/data/code/part-held.jsonl.
 gen: round 1's frozen generator (gen_programs.G) with seeds 300000-301999, width [2, 3, 4, "mixed"][seed % 4],
   first attempt of 400-1,400 characters that parses (as r2_build.gen). All held: round3/data/gen/part-held.jsonl.
 gen2: round 2's targeted generator (gen2.program) with seeds 10000-11499. All held: round3/data/gen2/part-held.jsonl.
The frozen modules are imported from D3-indentation/code (unchanged). Usage: python r3_build.py code|gen|gen2"""
import ast, hashlib, json, sys
from pathlib import Path
R3 = Path(__file__).resolve().parent.parent; D3 = R3.parent
sys.path.insert(0, str(D3 / "code"))
import pyindent as P
from build_material import chunk_records
import gen_programs as GP
import gen2 as G2

D = R3 / "data"; OLD = D3 / "data"; X1 = D3.parent / "X1-shared-parts" / "data"
H = lambda s: hashlib.sha256(s.encode()).hexdigest()


def old_hashes():
    hs = set()
    fs = [f for s in ("newcode", "gen", "stdlib", "r2code", "r2gen", "r2gen2") for f in (OLD / s).glob("part-*.jsonl")] \
        + list(X1.glob("code/part-*.jsonl")) + list(X1.glob("gen/part-*.jsonl"))
    for f in fs:
        for l in open(f): hs.add(H(json.loads(l)["text"]))
    return hs, len(fs)


def write(sub, rows):
    (D / sub).mkdir(parents=True, exist_ok=True)
    with open(D / sub / "part-held.jsonl", "w") as fh:
        for x in rows: fh.write(json.dumps(x) + "\n")


def code():
    rows = [json.loads(l) for l in open(D / "code" / "fetched.jsonl")]
    old, nfiles = old_hashes(); seen = set(); out = []
    st = dict(files=len(rows), normalised=0, dup_old=0, dup=0, old_hashes=len(old), old_part_files=nfiles)
    for r in sorted(rows, key=lambda r: r["key"]):
        src = P.normalise((D / "code" / "raw" / f"{r['key']}.py").read_bytes())
        if src is None: continue
        st["normalised"] += 1
        for c in chunk_records(src, dict(repo=r["repo"], path=r["path"], key=r["key"], commit=r["commit"], license=r["license"])):
            h = H(c["text"])
            if h in old: st["dup_old"] += 1; continue
            if h in seen: st["dup"] += 1; continue
            seen.add(h); out.append(c)
    write("code", out)
    st.update(held_chunks=len(out), held_repos=len({x["repo"] for x in out}),
              licences={l: sum(1 for x in out if x["license"] == l) for l in sorted({x["license"] for x in out})})
    (D / "code" / "build_summary.json").write_text(json.dumps(st, indent=1)); print(st)


def gen():
    out = []
    for seed in range(300000, 302000):
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
    write("gen", out); print(dict(gen_held=len(out)))


def gen2():
    out = [G2.program(s) for s in range(10000, 11500)]
    write("gen2", out); print(dict(gen2_held=len(out)))


if __name__ == "__main__":
    {"code": code, "gen": gen, "gen2": gen2}[sys.argv[1]]()
