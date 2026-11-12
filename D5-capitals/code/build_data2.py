"""D5 round 2: build the heading items (CHARTER-2.md, "Material"). Reads the D1 first-batch article
extracts (../D1-line-breaking/data/newtext/articles.jsonl; plain-text extracts with wiki-style headings
such as '== Early life ==') and, for every level-2 or level-3 heading whose previous paragraph has at
least 30 alphabetic words and whose next paragraph at least 20, writes one item:
  prev: the end of the previous paragraph, at most 600 characters, cut at a space
  heading: the heading text (without '=' marks)
  next: the start of the next paragraph, at most 300 characters, cut at a space
Reference-type sections are left out, and every character must keep its length under casing.
Split by article with the round-1 hash: int(sha256("D5-caps-" + str(pageid)), 16) % 3 ->
0 explore, 1 conf, 2 reserve. Writes data/headings-{explore,conf,reserve}.jsonl and
data/headings_summary.json. Prints counts only, never text."""
import hashlib, json, re
from pathlib import Path

D5 = Path(__file__).resolve().parent.parent
SRC = D5.parent / "D1-line-breaking" / "data" / "newtext" / "articles.jsonl"
NAMES = {0: "explore", 1: "conf", 2: "reserve"}
SKIP = {"references", "notes", "see also", "external links", "further reading", "bibliography", "sources",
        "citations", "footnotes"}
HEAD = re.compile(r"^(={2,3})\s*(.+?)\s*\1$")


def aligned(s):
    return all(len(c.upper()) == 1 and len(c.lower()) == 1 for c in s)


def n_words(s):
    return len(re.findall(r"[^\W\d_]+", s))


def tail(s, n):
    if len(s) <= n: return s
    k = s.find(" ", len(s) - n)
    return s[k + 1:] if k >= 0 else s[-n:]


def head(s, n):
    if len(s) <= n: return s
    k = s.rfind(" ", 0, n)
    return s[:k] if k > 0 else s[:n]


def items(extract):
    lines = [l.strip() for l in extract.split("\n")]
    out = []
    for j, l in enumerate(lines):
        m = HEAD.match(l)
        if not m or m.group(1) not in ("==", "==="): continue
        h = m.group(2).strip()
        if h.lower() in SKIP or not re.search(r"[^\W\d_]", h): continue
        prev = next((x for x in reversed(lines[:j]) if x), "")
        nxt = next((x for x in lines[j + 1:] if x), "")
        if not prev or HEAD.match(prev) or not nxt or HEAD.match(nxt): continue
        if n_words(prev) < 30 or n_words(nxt) < 20: continue
        it = {"level": len(m.group(1)), "heading": h, "prev": tail(prev, 600), "next": head(nxt, 300)}
        if not all(aligned(it[k]) for k in ("heading", "prev", "next")): continue
        out.append(it)
    return out


def main():
    arts = [json.loads(l) for l in open(SRC)]
    out = {k: [] for k in NAMES.values()}
    for a in sorted(arts, key=lambda a: a["pageid"]):
        k = int(hashlib.sha256(f"D5-caps-{a['pageid']}".encode()).hexdigest(), 16) % 3
        for i, it in enumerate(items(a.get("extract") or "")):
            out[NAMES[k]].append({"pageid": a["pageid"], "title": a["title"], "lastrevid": a["lastrevid"], "idx": i, **it})
    summ = {}
    for k, v in out.items():
        with open(D5 / "data" / f"headings-{k}.jsonl", "w") as f:
            for r in v: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summ[k] = {"items": len(v), "articles": len({r["pageid"] for r in v}),
                   "multiword_headings": sum(n_words(r["heading"]) >= 2 for r in v)}
    json.dump(summ, open(D5 / "data" / "headings_summary.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
