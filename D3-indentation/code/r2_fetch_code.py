"""D3 round-2 held-out material (CHARTER-2.md, Material): Python files from public GitHub repositories created on
or after 2026-09-01 with 5 to 9 stars, excluding every repository listed in round 1's or X1's fetch. A copy of
round 1's fetch_newcode.py with only the query, the exclusion and the output folder changed. Read-only GitHub API calls through the gh CLI; files fetched from raw.githubusercontent.com.
Nothing fetched is executed. Rule, fixed before the fetch:
 1. Repository search: language:Python, created>=2026-09-01, fork:false, archived:false, stars 5..9, sorted
    by stars, up to 5 pages of 100 (GitHub's cap is 1,000 results).
 2. Per repository: the default branch's recursive tree; candidate files end in .py, are 1,500-40,000 bytes,
    and have no path part in SKIP (vendored, generated, migrations, virtual envs, build output, notebooks
    checkpoints). Up to 4 files per repository, chosen by the order of sha256("D3-file-" + repo + path).
 3. Stop after 1,200 files or 40 MB in total, whichever comes first.
Raw files go to data/r2code/raw/<sha256(repo/path)[:16]>.py; the listing to data/r2code/fetched.jsonl.
Usage: python r2_fetch_code.py
"""
import hashlib, json, subprocess, sys, time, urllib.request
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "data" / "r2code"
X1 = Path(__file__).resolve().parents[2] / "X1-shared-parts" / "data" / "code" / "fetched.jsonl"
R1 = Path(__file__).resolve().parent.parent / "data" / "newcode" / "fetched.jsonl"
EXCL = {json.loads(l)["repo"] for f in (R1, X1) for l in open(f)}
RAW = D / "raw"; RAW.mkdir(parents=True, exist_ok=True)
SKIP = {"vendor", "vendored", "third_party", "thirdparty", "_vendor", "external", "extern", "migrations",
        "site-packages", "node_modules", "venv", ".venv", "env", "build", "dist", ".ipynb_checkpoints",
        "generated", "gen", "proto", "protos", "__pycache__"}
Q = "language:Python created:>=2026-09-01 fork:false archived:false stars:5..9"
MAXF, MAXB, PERREPO = 1200, 40_000_000, 4


def gh(path, **params):
    args = ["gh", "api", "-X", "GET", path] + sum([["-f", f"{k}={v}"] for k, v in params.items()], [])
    for t in range(4):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0: return json.loads(r.stdout)
        time.sleep(5 * (t + 1))
    raise RuntimeError(r.stderr[:500])


def main():
    repos = []
    for page in range(1, 6):
        res = gh("search/repositories", q=Q, sort="stars", order="desc", per_page=100, page=page)
        items = res.get("items", [])
        repos += [dict(full_name=i["full_name"], created_at=i["created_at"], stars=i["stargazers_count"],
                       branch=i["default_branch"], license=(i.get("license") or {}).get("spdx_id")) for i in items]
        if len(items) < 100: break
        time.sleep(2.5)  # search API: 30 requests per minute
    print("repos", len(repos), file=sys.stderr)
    out = open(D / "fetched.jsonl", "w"); nf = nb = 0; nrepo_used = 0
    nex = len(repos); repos = [r for r in repos if r["full_name"] not in EXCL]; print("after exclusion", len(repos), "of", nex, file=sys.stderr)
    for r in repos:
        if nf >= MAXF or nb >= MAXB: break
        try:
            tree = gh(f"repos/{r['full_name']}/git/trees/{r['branch']}", recursive="1")
        except Exception as e:
            print("tree fail", r["full_name"], str(e)[:100], file=sys.stderr); continue
        cands = [t for t in tree.get("tree", []) if t["type"] == "blob" and t["path"].endswith(".py")
                 and 1500 <= t.get("size", 0) <= 40000 and not (set(t["path"].split("/")[:-1]) & SKIP)]
        cands.sort(key=lambda t: hashlib.sha256(f"D3-file-{r['full_name']}{t['path']}".encode()).hexdigest())
        got = 0
        for t in cands[:PERREPO]:
            url = f"https://raw.githubusercontent.com/{r['full_name']}/{r['branch']}/{t['path']}"
            try:
                data = urllib.request.urlopen(url, timeout=30).read()
            except Exception as e:
                print("raw fail", url, str(e)[:80], file=sys.stderr); continue
            key = hashlib.sha256(f"{r['full_name']}/{t['path']}".encode()).hexdigest()[:16]
            (RAW / f"{key}.py").write_bytes(data)
            out.write(json.dumps(dict(key=key, repo=r["full_name"], path=t["path"], created_at=r["created_at"],
                                      stars=r["stars"], license=r["license"], bytes=len(data),
                                      sha256=hashlib.sha256(data).hexdigest())) + "\n")
            nf += 1; nb += len(data); got += 1
        nrepo_used += got > 0
    out.close()
    s = dict(query=Q, fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), repos_listed=nex, repos_after_exclusion=len(repos), excluded_names=len(EXCL),
             repos_used=nrepo_used, files=nf, bytes=nb)
    (D / "fetch_summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s))


if __name__ == "__main__":
    main()
