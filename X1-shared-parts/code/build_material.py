"""X1 material and splits (CHARTER.md, Material). Rules fixed before any model read X1 material.
Prints counts only, never text.

 gen     D3's frozen generator (D3 code/gen_programs.py, class G, imported) with seeds 100000-103999 (D3 used
         0-3999), the same width cycle, length window and ast check as D3's main();
         part = int(sha256("X1-gen-" + seed), 16) % 2 -> 0 explore, 1 held.
 code    data/code/fetched.jsonl through D3's pyindent.normalise and pyindent.chunks with D3's build rule
         (chunks with >= 4 decision points; exact duplicates kept once, also against D3's two parts);
         part = int(sha256("X1-code-" + repo), 16) % 2.
 prose   data/prose/paragraphs.jsonl (D1's paragraph rule already applied);
         part = int(sha256("X1-prose-" + pageid), 16) % 2. A paragraph's 'caps_ok' flag is D5's rule
         (every character keeps its length under upper- and lower-casing, >= 60 alphabetic words).
 ord     D4's reserve frames (rankings, positions, anniversaries: the families D4's head tests used) and D4's
         reserve numbers (D4 common.number_part_ext == 'reserve'), 3 to 5 digits, teen (x11-x13) or ending in
         1-3 (non-teen) or ending in 4-9/0 (other); split by number:
         part = int(sha256("X1-ord-" + n), 16) % 2. D4's part-reserve.jsonl itself is not read.
Writes data/<source>/part-{explore,held}.jsonl and data/<source>/build_summary.json.
Usage: build_material.py gen|code|prose|ord"""
import ast, hashlib, importlib.util, json, re, sys
from pathlib import Path

X1 = Path(__file__).resolve().parent.parent
AREA = X1.parent
D = X1 / "data"
H = lambda s: int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2
PARTS = ("explore", "held")


def load_mod(name, path, add_path=None):
    if add_path: sys.path.insert(0, str(add_path))
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


def write(src, parts, extra=None):
    (D / src).mkdir(parents=True, exist_ok=True)
    for k, v in parts.items():
        with open(D / src / f"part-{k}.jsonl", "w") as fh:
            for x in v: fh.write(json.dumps(x) + "\n")
    s = {k: len(v) for k, v in parts.items()}; s.update(extra or {})
    (D / src / "build_summary.json").write_text(json.dumps(s, indent=1)); print(src, json.dumps(s))


def gen():
    g = load_mod("d3_gen", AREA / "D3-indentation" / "code" / "gen_programs.py")
    parts = {k: [] for k in PARTS}
    for seed in range(100000, 104000):
        width = [2, 3, 4, "mixed"][seed % 4]
        for attempt in range(20):
            src = g.G(seed * 100 + attempt, width).program()
            if not (400 <= len(src) <= 1400): continue
            try: ast.parse(src)
            except SyntaxError: continue
            break
        else:
            continue
        parts[PARTS[H(f"X1-gen-{seed}")]].append(dict(seed=seed, attempt=attempt, width=width, text=src))
    write("gen", parts)


def code():
    P = load_mod("d3_pyindent", AREA / "D3-indentation" / "code" / "pyindent.py")
    seen = set()
    for p in ("explore", "conf"):  # hashes only: D3's chunks are never printed or used
        for l in open(AREA / "D3-indentation" / "data" / "newcode" / f"part-{p}.jsonl"):
            seen.add(hashlib.sha256(json.loads(l)["text"].encode()).hexdigest())
    rows = [json.loads(l) for l in open(D / "code" / "fetched.jsonl")]
    parts = {k: [] for k in PARTS}; st = dict(files=len(rows), normalised=0, dup=0, dup_d3=0)
    n3 = len(seen)
    for r in sorted(rows, key=lambda r: r["key"]):
        src = P.normalise((D / "code" / "raw" / f"{r['key']}.py").read_bytes())
        if src is None: continue
        st["normalised"] += 1
        for i, c in enumerate(P.chunks(src)):
            if len(P.decision_points(c)) < 4: continue
            h = hashlib.sha256(c.encode()).hexdigest()
            if h in seen: st["dup"] += 1; continue
            seen.add(h)
            parts[PARTS[H(f"X1-code-{r['repo']}")]].append(dict(repo=r["repo"], path=r["path"], key=r["key"], idx=i, text=c))
    st["d3_chunk_hashes_excluded"] = n3
    st.update({f"{k}_repos": len({x["repo"] for x in v}) for k, v in parts.items()})
    write("code", parts, st)


def prose():
    def aligned(s): return all(len(c.upper()) == 1 and len(c.lower()) == 1 for c in s)
    rows = [json.loads(l) for l in open(D / "prose" / "paragraphs.jsonl")]
    parts = {k: [] for k in PARTS}
    for r in sorted(rows, key=lambda r: (r["pageid"], r["idx"])):
        r = dict(r, caps_ok=bool(aligned(r["text"]) and len(re.findall(r"[^\W\d_]+", r["text"])) >= 60))
        parts[PARTS[H(f"X1-prose-{r['pageid']}")]].append(r)
    write("prose", parts, {f"{k}_articles": len({x["pageid"] for x in v}) for k, v in parts.items()} |
          {f"{k}_caps_ok": sum(x["caps_ok"] for x in v) for k, v in parts.items()})


def ord_():
    d4 = load_mod("d4common", AREA / "D4-ordinal-suffixes" / "code" / "common.py")
    frames = [json.loads(l) for l in open(AREA / "D4-ordinal-suffixes" / "data" / "frames.jsonl")]
    frames = [f for f in frames if f["part"] == "reserve" and f["family"] in ("rankings", "positions", "anniversaries")]
    parts = {k: dict(frames=[dict(fid=f["fid"], family=f["family"], hi=f["hi"], text=f["text"]) for f in frames], numbers=[])
             for k in PARTS}
    cnt = {}
    for n in range(100, 100000):
        if d4.number_part_ext(n) != "reserve": continue
        cls = "teen" if n % 100 in (11, 12, 13) else "end123" if n % 10 in (1, 2, 3) else "other"
        k = PARTS[H(f"X1-ord-{n}")]
        parts[k]["numbers"].append(dict(n=n, digits=len(str(n)), cls=cls))
        cnt[f"{k}|{len(str(n))}|{cls}"] = cnt.get(f"{k}|{len(str(n))}|{cls}", 0) + 1
    (D / "ord").mkdir(parents=True, exist_ok=True)
    for k, v in parts.items(): (D / "ord" / f"part-{k}.json").write_text(json.dumps(v))
    s = dict(frames=len(frames), counts=dict(sorted(cnt.items())))
    (D / "ord" / "build_summary.json").write_text(json.dumps(s, indent=1)); print("ord", json.dumps(s))


if __name__ == "__main__":
    {"gen": gen, "code": code, "prose": prose, "ord": ord_}[sys.argv[1]]()
