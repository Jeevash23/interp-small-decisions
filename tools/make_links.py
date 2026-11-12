"""Create the links that D3 round 3's frozen scripts read their material through.

    python tools/make_links.py            # create every link in data/LINKS.json (an existing correct link is kept)
    python tools/make_links.py --remove   # remove them

Round 3 ran byte-identical copies of the frozen round-1, round-2 and X1 scripts on new material. Those scripts open
their material under the earlier rounds' file names (for example data/newcode/part-conf.jsonl), so round 3 gave each
such name a relative link to its new part. Version control and archives do not always keep links, so the release
lists them in data/LINKS.json and this script makes them. Nothing else is changed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]


def main():
    remove = "--remove" in sys.argv
    links = json.loads((RELEASE / "data" / "LINKS.json").read_text())["links"]
    made = kept = removed = 0
    for rel, rec in sorted(links.items()):
        p = RELEASE / rel
        if not str(p.resolve().parent).startswith(str(RELEASE.resolve())):
            raise SystemExit(f"refusing a link outside the release: {rel}")
        if remove:
            if p.is_symlink():
                p.unlink()
                removed += 1
            continue
        if p.is_symlink() and os.readlink(p) == rec["target"]:
            kept += 1
            continue
        if p.exists() or p.is_symlink():
            raise SystemExit(f"{rel} exists and is not the expected link; move it away first")
        if not (RELEASE / rec["resolves_to"]).exists():
            raise SystemExit(f"{rel}: its target {rec['resolves_to']} is missing (run tools/unpack.py first)")
        p.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(rec["target"], p)
        made += 1
    print(f"links: {removed} removed" if remove else f"links: {made} made, {kept} already present")


if __name__ == "__main__":
    main()
