"""Check every file named in the studies' freeze records (their FROZEN.md tables) against its frozen SHA-256.

    python tools/check_frozen_tables.py     # after tools/unpack.py; exit status 1 if any file fails

Each study's FROZEN.md lists, section by section, the files frozen or sealed at each step with their SHA-256. A file
passes when it is byte for byte the frozen one, when release-changes.json records exactly the change made for the
release (the frozen digest as the original), or when it is a file the release leaves out on purpose and
release-changes.json lists it with the frozen digest (tools/release_paths.py has the rule).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_paths as rp  # noqa: E402


def main():
    tables = rp.changes().get("freeze_tables", [])
    total, bad = {}, []
    for t in tables:
        text = (rp.RELEASE / t["path"]).read_text()
        rows = re.findall(r"^\| (\S+)(?: \([^)]*\))? \| ([0-9a-f]{64}) \|", text, re.M)
        counts = {}
        for key, digest in rows:
            rel = os.path.normpath(f"{t['base']}/{key}")
            s = rp.status(rp.RELEASE / rel, digest, rel)
            counts[s] = counts.get(s, 0) + 1
            total[s] = total.get(s, 0) + 1
            if s not in rp.OK:
                bad.append(f"{t['path']}: {key}: {s}")
        print(f"{t['path']}: {len(rows)} rows: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))
    print("total: " + ", ".join(f"{n} {k}" for k, n in sorted(total.items())))
    for b in bad:
        print("FAILED", b)
    sys.exit(1 if bad or not tables else 0)


if __name__ == "__main__":
    main()
