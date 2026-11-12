"""D5: build the paragraph parts (CHARTER.md, "Material"). Reads the seven part files of D1's first
new-text batch (already filtered by D1's paragraph rule), keeps paragraphs whose every character keeps
its length under upper- and lower-casing and which have at least 60 alphabetic words, and splits them
by article: int(sha256("D5-caps-" + str(pageid)), 16) % 3 -> 0 explore, 1 conf, 2 reserve.
Writes data/part-{explore,conf,reserve}.jsonl (pageid, title, lastrevid, idx, text) and
data/build_summary.json (counts only). Prints counts only, never text."""
import hashlib, json, re
from pathlib import Path

D5 = Path(__file__).resolve().parent.parent
SRC = D5.parent / "D1-line-breaking" / "data" / "newtext"
PARTS = ["conf-pythia410m", "explore-pythia1b", "conf-pythia1b", "explore-qwen05b", "conf-qwen05b",
         "explore-olmo1b", "conf-olmo1b"]
NAMES = {0: "explore", 1: "conf", 2: "reserve"}


def aligned(s):
    return all(len(c.upper()) == 1 and len(c.lower()) == 1 for c in s)


def n_words(s):
    return len(re.findall(r"[^\W\d_]+", s))


def main():
    rows = []
    for p in PARTS:
        rows += [json.loads(l) for l in open(SRC / f"part-{p}.jsonl")]
    rows.sort(key=lambda r: (r["pageid"], r["idx"]))
    out = {k: [] for k in NAMES.values()}
    drop = {"length_change": 0, "short": 0}
    for r in rows:
        t = r["text"]
        if not aligned(t): drop["length_change"] += 1; continue
        if n_words(t) < 60: drop["short"] += 1; continue
        k = int(hashlib.sha256(f"D5-caps-{r['pageid']}".encode()).hexdigest(), 16) % 3
        out[NAMES[k]].append({x: r[x] for x in ("pageid", "title", "lastrevid", "idx", "text")})
    (D5 / "data").mkdir(exist_ok=True)
    summ = {"source_paragraphs": len(rows), "dropped": drop}
    for k, v in out.items():
        with open(D5 / "data" / f"part-{k}.jsonl", "w") as f:
            for r in v: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ[k] = {"paragraphs": len(v), "articles": len({r["pageid"] for r in v})}
    json.dump(summ, open(D5 / "data" / "build_summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
