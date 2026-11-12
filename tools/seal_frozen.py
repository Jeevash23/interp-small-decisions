"""Make the files named in the studies' freeze records read-only, as they were in the studies' own folders.

    python tools/seal_frozen.py            # chmod a-w on every shipped file a FROZEN.md table names
    python tools/seal_frozen.py --unseal   # make them writable again

The studies froze a document by recording its SHA-256 in FROZEN.md and making the file read-only; their loaders
and one paper checker (D3's "PROTOCOL-2.md is read-only on disk") test the read-only bit. Version control and
archives do not keep it, so this restores it after a checkout. Nothing else is changed.
"""
from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_paths as rp  # noqa: E402


def main():
    unseal = "--unseal" in sys.argv
    n = 0
    for t in rp.changes().get("freeze_tables", []):
        text = (rp.RELEASE / t["path"]).read_text()
        for key in re.findall(r"^\| (\S+)(?: \([^)]*\))? \| [0-9a-f]{64} \|", text, re.M):
            p = rp.RELEASE / os.path.normpath(f"{t['base']}/{key}")
            if not p.exists():
                continue
            mode = p.stat().st_mode
            new = (mode | stat.S_IWUSR) if unseal else (mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            if new != mode:
                os.chmod(p, new)
                n += 1
    print(f"{'unsealed' if unseal else 'sealed (read-only)'}: {n} files changed")


if __name__ == "__main__":
    main()
