"""X1 prose material (CHARTER.md, Material): a fourth batch of English Wikipedia articles created 2025-26.
Selection rule as D1-X's fetch_newtext3.py (logevents of page creations, summary filter, length >= 6,000,
plain-text extracts, D1's paragraph filter, at most 3 paragraphs per article), except:
 1. Starting points are the 11th and the 25th of each month, Jan 2025 to Oct 2026 (44 starts); up to 22
    pages kept per start.
 2. Pages in any of D1's three earlier batches (D1 data/newtext, data/newtext2, xmodel/data/newtext3) are
    excluded, so no article was read by D1, D1-X or D5 (D5's paragraphs come from data/newtext).
 3. No split here: build_material.py splits by article hash. Extracts are cached one article at a time.
Writes data/prose/{pages.json, extracts.jsonl, articles.jsonl, paragraphs.jsonl, fetch_summary.json}.
Prints counts only. Licence of the text: CC BY-SA 4.0."""
import json, re, time, urllib.request, urllib.error, urllib.parse, datetime
from pathlib import Path

X1 = Path(__file__).resolve().parent.parent
OUT = X1 / "data" / "prose"; OUT.mkdir(parents=True, exist_ok=True)
D1 = X1.parent / "D1-line-breaking"
PREV = set()
for f in (D1 / "data" / "newtext" / "articles.jsonl", D1 / "data" / "newtext2" / "articles.jsonl",
          D1 / "xmodel" / "data" / "newtext3" / "articles.jsonl"):
    PREV |= {json.loads(l)["pageid"] for l in open(f)}
API = "https://en.wikipedia.org/w/api.php"
UA = "X1-interp-research/1.0 (non-commercial interpretability research script)"
NPER = 22
SKIP = re.compile(r"redirect|split|merg|cop(y|ied)|move|translat|from \[\[|draft", re.I)


def get(params, tries=8):
    params = dict(params, format="json", formatversion="2", maxlag="5")
    url = API + "?" + urllib.parse.urlencode(params); err = None
    for k in range(tries):
        time.sleep(1.0)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                d = json.loads(r.read().decode())
            if d.get("error", {}).get("code") == "maxlag": time.sleep(10); continue
            return d
        except urllib.error.HTTPError as e:
            err = e; time.sleep(float(e.headers.get("Retry-After") or 30 * (k + 1)))
        except Exception as e:
            err = e; time.sleep(5 * (k + 1))
    raise err


def paragraphs(text):
    """D1's paragraph rule (unchanged)."""
    out = []
    for line in text.split("\n"):
        s = re.sub(r"\s+", " ", line.strip())
        if not s or s.startswith("=") or not (600 <= len(s) <= 1800): continue
        if "@" in s or "{\\displaystyle" in s or any(ord(c) > 0x24F for c in s): continue
        out.append(s)
        if len(out) == 3: break
    return out


def main():
    pf = OUT / "pages.json"
    if pf.exists():
        pages = json.loads(pf.read_text())
    else:
        starts = [f"{y}-{m:02d}-{d:02d}T00:00:00Z" for y, m in [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 11)]
                  for d in (11, 25)]
        pages, seen = [], set()
        for st in starts:
            kept, cont = 0, {}
            while kept < NPER:
                r = get(dict(action="query", list="logevents", letype="create", lenamespace=0, lestart=st, ledir="newer", lelimit=500, **cont))
                evs = [e for e in r["query"]["logevents"] if not SKIP.search(e.get("comment", "")) and e.get("pageid")]
                for i in range(0, len(evs), 50):
                    chunk = evs[i:i + 50]
                    info = get(dict(action="query", prop="info", pageids="|".join(str(e["pageid"]) for e in chunk)))
                    ok = {p["pageid"]: p for p in info["query"]["pages"] if not p.get("missing") and not p.get("redirect")
                          and p.get("ns") == 0 and p.get("length", 0) >= 6000}
                    for e in chunk:
                        p = ok.get(e["pageid"])
                        if p and p["pageid"] not in seen and p["pageid"] not in PREV and kept < NPER:
                            seen.add(p["pageid"]); kept += 1
                            pages.append(dict(pageid=p["pageid"], title=p["title"], created=e["timestamp"], lastrevid=p["lastrevid"],
                                              length=p["length"], start=st))
                    if kept >= NPER: break
                if "continue" not in r: break
                cont = {"lecontinue": r["continue"]["lecontinue"]}
            print(st, kept, len(pages), flush=True)
        pf.write_text(json.dumps(pages))
    cache = OUT / "extracts.jsonl"
    done = {json.loads(l)["pageid"]: json.loads(l) for l in open(cache)} if cache.exists() else {}
    arts = []
    with open(cache, "a") as cf:
        for q, p in enumerate(pages):
            if p["pageid"] not in done:
                r = get(dict(action="query", prop="extracts", explaintext=1, exsectionformat="wiki", pageids=p["pageid"]))
                a = dict(p, extract=r["query"]["pages"][0].get("extract", "")); cf.write(json.dumps(a) + "\n"); cf.flush(); done[p["pageid"]] = a
            arts.append(done[p["pageid"]])
            if q % 100 == 0: print("extracts", q, flush=True)
    fetched = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    n = 0
    with open(OUT / "articles.jsonl", "w") as fa, open(OUT / "paragraphs.jsonl", "w") as fp:
        for a in arts:
            fa.write(json.dumps({k: a[k] for k in ("pageid", "title", "created", "lastrevid", "length", "start")}) + "\n")
            for q, s in enumerate(paragraphs(a["extract"])):
                fp.write(json.dumps(dict(pageid=a["pageid"], title=a["title"], lastrevid=a["lastrevid"], idx=q, text=s)) + "\n"); n += 1
    s = dict(fetched_utc=fetched, n_articles=len(arts), n_paragraphs=n, excluded_previous=len(PREV))
    (OUT / "fetch_summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s))


if __name__ == "__main__":
    main()
