"""X1 natural-code material (CHARTER.md, Material). D3's file rule (D3 code/fetch_newcode.py: SKIP set,
1,500-40,000-byte .py files, up to 4 per repository chosen by a sha256 order, read-only gh API calls,
raw files from raw.githubusercontent.com, nothing executed), with these changes:
 1. Query: language:Python created:>=2026-10-01 fork:false archived:false stars:>=10, sorted by stars,
    up to 5 pages of 100.
 2. Repositories already used by D3 (D3 data/newcode/fetched.jsonl) are skipped.
 3. File order key sha256("X1-file-" + repo + path); stop after 700 files or 25 MB.
Writes data/code/raw/<key>.py, data/code/fetched.jsonl, data/code/fetch_summary.json."""
import hashlib, json, sys, time, urllib.request
from pathlib import Path

X1 = Path(__file__).resolve().parent.parent
D3C = X1.parent / "D3-indentation" / "code"
sys.path.insert(0, str(D3C))
import importlib.util
_spec = importlib.util.spec_from_file_location("d3_fetch", D3C / "fetch_newcode.py")
d3f = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(d3f)   # reuses D3's SKIP and gh()

D = X1 / "data" / "code"; RAW = D / "raw"; RAW.mkdir(parents=True, exist_ok=True)
Q = "language:Python created:>=2026-10-01 fork:false archived:false stars:>=10"
MAXF, MAXB, PERREPO = 700, 25_000_000, 4
D3REPOS = {json.loads(l)["repo"] for l in open(X1.parent / "D3-indentation" / "data" / "newcode" / "fetched.jsonl")}


def main():
    repos = []
    for page in range(1, 6):
        res = d3f.gh("search/repositories", q=Q, sort="stars", order="desc", per_page=100, page=page)
        items = res.get("items", [])
        repos += [dict(full_name=i["full_name"], created_at=i["created_at"], stars=i["stargazers_count"], branch=i["default_branch"],
                       license=(i.get("license") or {}).get("spdx_id")) for i in items]
        if len(items) < 100: break
        time.sleep(2.5)
    n_listed = len(repos); repos = [r for r in repos if r["full_name"] not in D3REPOS]
    print("repos", n_listed, "after excluding D3", len(repos), file=sys.stderr)
    out = open(D / "fetched.jsonl", "w"); nf = nb = 0; used = 0
    for r in repos:
        if nf >= MAXF or nb >= MAXB: break
        try:
            tree = d3f.gh(f"repos/{r['full_name']}/git/trees/{r['branch']}", recursive="1")
        except Exception as e:
            print("tree fail", r["full_name"], str(e)[:100], file=sys.stderr); continue
        cands = [t for t in tree.get("tree", []) if t["type"] == "blob" and t["path"].endswith(".py")
                 and 1500 <= t.get("size", 0) <= 40000 and not (set(t["path"].split("/")[:-1]) & d3f.SKIP)]
        cands.sort(key=lambda t: hashlib.sha256(f"X1-file-{r['full_name']}{t['path']}".encode()).hexdigest())
        got = 0
        for t in cands[:PERREPO]:
            url = f"https://raw.githubusercontent.com/{r['full_name']}/{r['branch']}/{t['path']}"
            try:
                data = urllib.request.urlopen(url, timeout=30).read()
            except Exception as e:
                print("raw fail", url, str(e)[:80], file=sys.stderr); continue
            key = hashlib.sha256(f"{r['full_name']}/{t['path']}".encode()).hexdigest()[:16]
            (RAW / f"{key}.py").write_bytes(data)
            out.write(json.dumps(dict(key=key, repo=r["full_name"], path=t["path"], created_at=r["created_at"], stars=r["stars"],
                                      license=r["license"], bytes=len(data), sha256=hashlib.sha256(data).hexdigest())) + "\n")
            nf += 1; nb += len(data); got += 1
        used += got > 0
    out.close()
    s = dict(query=Q, fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), repos_listed=n_listed,
             repos_after_d3_exclusion=len(repos), repos_used=used, files=nf, bytes=nb)
    (D / "fetch_summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s))


if __name__ == "__main__":
    main()
