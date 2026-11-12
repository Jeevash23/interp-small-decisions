"""D3 round 3 natural-code material (CHARTER-3.md, Material): permissively licensed Python files from public
GitHub repositories created on or after 2026-09-01. Read-only GitHub API calls through the gh CLI; files read
from raw.githubusercontent.com at a recorded commit. Nothing fetched is executed. Rule (fixed in CHARTER-3.md):
 1. One repository search per licence (mit, apache-2.0, bsd-3-clause, bsd-2-clause, cc0-1.0, unlicense):
    language:Python created:>=2026-09-01 fork:false archived:false stars:>=3 license:<L>, sorted by stars,
    up to 5 pages of 100.
 2. Drop repositories used by D3 round 1, D3 round 2 or X1; order the rest by sha256("D3R3-repo-" + name).
 3. Per repository: licence endpoint must return one of ALLOWED; head commit of the default branch is recorded;
    recursive tree at that commit; candidates .py, 1,500-40,000 bytes, no path part in round 2's SKIP; up to 4
    per repository by sha256("D3R3-file-" + repo + path). Files whose text names another licence are dropped
    (BAD_LIC). Licence text and a root NOTICE file (if any) are saved to licences/.
 4. Stop after 1,000 kept files or 30 MB.
Writes round3/data/code/raw/<key>.py, fetched.jsonl, dropped.jsonl, licences/, fetch_summary.json.
Usage: python -I r3_fetch.py"""
import base64, hashlib, json, re, subprocess, sys, time, urllib.request
from pathlib import Path

R3 = Path(__file__).resolve().parent.parent
D3 = R3.parent
X1 = D3.parent / "X1-shared-parts"
D = R3 / "data" / "code"; RAW = D / "raw"; LIC = D / "licences"
RAW.mkdir(parents=True, exist_ok=True); LIC.mkdir(parents=True, exist_ok=True)
PREV = [D3 / "data" / "newcode" / "fetched.jsonl", D3 / "data" / "r2code" / "fetched.jsonl", X1 / "data" / "code" / "fetched.jsonl"]
EXCL = {json.loads(l)["repo"] for f in PREV for l in open(f)}
SKIP = {"vendor", "vendored", "third_party", "thirdparty", "_vendor", "external", "extern", "migrations",
        "site-packages", "node_modules", "venv", ".venv", "env", "build", "dist", ".ipynb_checkpoints",
        "generated", "gen", "proto", "protos", "__pycache__"}
LICS = ["mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "cc0-1.0", "unlicense"]
ALLOWED = {"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "CC0-1.0", "Unlicense"}
BAD_LIC = re.compile(r"GNU General Public License|GNU Lesser General Public|GNU Affero|Mozilla Public License|"
                     r"\b(?:LGPL|AGPL|GPL-)", re.I)
SPDX = re.compile(r"SPDX-License-Identifier:\s*([^\s*#\"']+(?:\s+(?:OR|AND|WITH)\s+[^\s*#\"']+)*)")
Q = "language:Python created:>=2026-09-01 fork:false archived:false stars:>=3 license:{}"
MAXF, MAXB, PERREPO = 1000, 30_000_000, 4
H = lambda s: hashlib.sha256(s.encode()).hexdigest()


def gh(path, **params):
    args = ["gh", "api", "-X", "GET", path] + sum([["-f", f"{k}={v}"] for k, v in params.items()], [])
    for t in range(4):
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode == 0: return json.loads(r.stdout)
        if "404" in r.stderr or "Not Found" in r.stderr: raise FileNotFoundError(path)
        time.sleep(5 * (t + 1))
    raise RuntimeError(r.stderr[:500])


def raw(repo, sha, path):
    return urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}", timeout=30).read()


def spdx_ok(text):
    for m in SPDX.finditer(text):
        ids = set(re.split(r"\s+(?:OR|AND|WITH)\s+", m.group(1).strip("()")))
        if not ids <= ALLOWED: return False
    return True


def main():
    repos = {}
    for L in LICS:
        for page in range(1, 6):
            res = gh("search/repositories", q=Q.format(L), sort="stars", order="desc", per_page=100, page=page)
            items = res.get("items", [])
            for i in items:
                repos.setdefault(i["full_name"], dict(full_name=i["full_name"], created_at=i["created_at"],
                                 stars=i["stargazers_count"], branch=i["default_branch"], query_license=L))
            time.sleep(2.5)  # search API: 30 requests per minute
            if len(items) < 100: break
    nlisted = len(repos)
    order = sorted((r for r in repos.values() if r["full_name"] not in EXCL), key=lambda r: H("D3R3-repo-" + r["full_name"]))
    print("listed", nlisted, "after exclusion", len(order), file=sys.stderr, flush=True)
    out = open(D / "fetched.jsonl", "w"); drop = open(D / "dropped.jsonl", "w")
    nf = nb = nrepo = 0; why = {}
    def dropped(kind, **kw):
        why[kind] = why.get(kind, 0) + 1; drop.write(json.dumps(dict(reason=kind, **kw)) + "\n")
    for r in order:
        if nf >= MAXF or nb >= MAXB: break
        name = r["full_name"]
        try:
            lic = gh(f"repos/{name}/license")
            spdx = (lic.get("license") or {}).get("spdx_id")
            if spdx not in ALLOWED: dropped("repo_licence", repo=name, spdx=spdx); continue
            sha = gh(f"repos/{name}/commits/{r['branch']}")["sha"]
            tree = gh(f"repos/{name}/git/trees/{sha}", recursive="1")
        except Exception as e:
            dropped("api_fail", repo=name, err=str(e)[:120]); continue
        cands = [t for t in tree.get("tree", []) if t["type"] == "blob" and t["path"].endswith(".py")
                 and 1500 <= t.get("size", 0) <= 40000 and not (set(t["path"].split("/")[:-1]) & SKIP)]
        cands.sort(key=lambda t: H(f"D3R3-file-{name}{t['path']}"))
        got = 0
        for t in cands[:PERREPO]:
            try:
                data = raw(name, sha, t["path"])
            except Exception as e:
                dropped("raw_fail", repo=name, path=t["path"], err=str(e)[:80]); continue
            txt = data.decode("utf-8", "replace")
            if BAD_LIC.search(txt) or not spdx_ok(txt):
                dropped("file_licence", repo=name, path=t["path"]); continue
            key = H(f"{name}/{t['path']}")[:16]
            (RAW / f"{key}.py").write_bytes(data)
            out.write(json.dumps(dict(key=key, repo=name, path=t["path"], commit=sha, license=spdx,
                                      licence_file=lic.get("path"), created_at=r["created_at"], stars=r["stars"],
                                      bytes=len(data), sha256=hashlib.sha256(data).hexdigest())) + "\n")
            nf += 1; nb += len(data); got += 1
        if got:
            nrepo += 1; stem = name.replace("/", "__")
            (LIC / f"{stem}.LICENSE.txt").write_bytes(base64.b64decode(lic.get("content", "")))
            for t in tree.get("tree", []):
                if t["type"] == "blob" and t["path"] in ("NOTICE", "NOTICE.txt", "NOTICE.md"):
                    try: (LIC / f"{stem}.{t['path']}").write_bytes(raw(name, sha, t["path"]))
                    except Exception as e: dropped("notice_fail", repo=name, err=str(e)[:80])
        out.flush(); drop.flush()
    out.close(); drop.close()
    s = dict(query=Q, licences=LICS, fetched_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), repos_listed=nlisted,
             repos_after_exclusion=len(order), excluded_names=len(EXCL), repos_used=nrepo, files=nf, bytes=nb, dropped=why)
    (D / "fetch_summary.json").write_text(json.dumps(s, indent=1)); print(json.dumps(s))


if __name__ == "__main__":
    main()
