"""The SHA-256 of the frozen original of a file in this release.

    from frozen_digest import sha256_as_frozen

A few frozen files were changed for the release (personal paths in shell scripts, personal wording in records) and
a few were not shipped (unread sealed material, see SEALED-MATERIAL.json). release-changes.json records each change
with the original's SHA-256 and each unshipped file with its frozen SHA-256. This function returns:
- for a file that is byte for byte the original: its own SHA-256;
- for a file changed for the release: the original's SHA-256, but only if the file is exactly the release copy that
  release-changes.json records (otherwise its own SHA-256, which then fails any comparison with the freeze);
- for a file that is not shipped: the SHA-256 that release-changes.json records for it (if it is not listed there,
  FileNotFoundError).
The paper checker of D3 compares this value with its study's FROZEN.md (see RELEASE-CHANGES.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_paths as rp  # noqa: E402


def sha256_as_frozen(path) -> str:
    path = Path(path)
    rel = rp.release_relative(path)
    ch = rp.changes()
    if not path.exists():
        entry = ch.get("frozen_files_not_shipped", {}).get(rel)
        if entry is None:
            raise FileNotFoundError(f"{path}: not in this release and not listed as a frozen file left out")
        return entry["original_sha256"]
    got = rp.sha256(path)
    rec = ch.get("changed_files", {}).get(rel)
    if rec and rec.get("release_sha256") == got:
        return rec["original_sha256"]
    return got
