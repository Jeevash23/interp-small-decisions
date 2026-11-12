"""Fetch again the GitHub source files of D3's first two rounds and of X1, which this release does not ship.

    python tools/refetch_old_code.py --check      # report which files are present and match; no network
    python tools/refetch_old_code.py              # download the missing files (network), keeping only exact matches

Each folder's fetched.jsonl lists every source file the studies fetched (repository, path, licence and SHA-256). This
script downloads each file from raw.githubusercontent.com (the repository's default branch, as the original fetch
did) into the folder's raw/ subfolder as raw/<key>.py, the name the build scripts read, and keeps it only if its
SHA-256 matches; a file changed or deleted since is reported, not kept. The licences are the repositories' own: many
of these files carry none, so check a repository's terms before you use its code.

With every file of a folder present, the study's own build script rebuilds the parts, and the result can be checked
against the frozen SHA-256 in FROZEN.md and against the chunk digest lists (part-*.chunks.jsonl):
  D3 round 1:  cd D3-indentation/code && python build_material.py newcode
  D3 round 2:  cd D3-indentation/code && python r2_build.py code
  X1:          cd X1-shared-parts/code && python build_material.py code
These scripts rewrite their part files and build summaries; run them in a copy if you want to keep the release as
shipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
FOLDERS = ["D3-indentation/data/newcode", "D3-indentation/data/r2code", "X1-shared-parts/data/code"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="no network: report what is present")
    ap.add_argument("--folder", choices=FOLDERS, action="append", help="only these folders (default: all three)")
    a = ap.parse_args()
    for folder in a.folder or FOLDERS:
        d = RELEASE / folder
        rows = [json.loads(l) for l in (d / "fetched.jsonl").read_text().splitlines() if l.strip()]
        raw = d / "raw"
        counts = {"present": 0, "fetched": 0, "changed upstream": 0, "missing upstream": 0, "absent": 0}
        for r in rows:
            target = raw / f"{r['key']}.py"
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == r["sha256"]:
                counts["present"] += 1
                continue
            if a.check:
                counts["absent"] += 1
                continue
            url = (f"https://raw.githubusercontent.com/{r['repo']}/HEAD/"
                   + urllib.parse.quote(r["path"]))
            try:
                data = urllib.request.urlopen(url, timeout=30).read()
            except Exception as e:  # noqa: BLE001
                counts["missing upstream"] += 1
                print(f"missing: {r['repo']} {r['path']} ({str(e)[:60]})", file=sys.stderr)
                continue
            if hashlib.sha256(data).hexdigest() != r["sha256"]:
                counts["changed upstream"] += 1
                print(f"changed since the fetch: {r['repo']} {r['path']}", file=sys.stderr)
                continue
            raw.mkdir(exist_ok=True)
            target.write_bytes(data)
            counts["fetched"] += 1
            time.sleep(0.05)
        print(f"{folder}: {len(rows)} files: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
