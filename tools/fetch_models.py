"""Download the model checkpoints the studies used into the Hugging Face cache, at the recorded revisions.

    python tools/fetch_models.py --list               # each model, its revision, size and which studies used it
    python tools/fetch_models.py qwen3-0.6b-base      # download one model (needs network access) and check its files
    python tools/fetch_models.py --all                # every model in models/models.json (about 28 GB)
    python tools/fetch_models.py --check NAME ...     # check files already in the cache (no network)

The study code loads each model by its Hub name (for example Qwen/Qwen3-0.6B-Base) through the Hugging Face cache
(HF_HOME or ~/.cache/huggingface). Each file is checked against the SHA-256 recorded in models/models.json when the
studies ran. Gemma-4-E2B is a gated model: accept its licence on the Hub and log in (huggingface-cli login) first.
The OLMo-2 training checkpoints are listed separately in models.json; only the exploratory checkpoint analyses read
them, and their outputs are included in this release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
SPEC = json.loads((RELEASE / "models" / "models.json").read_text())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def snapshot(name, offline):
    from huggingface_hub import snapshot_download
    s = SPEC["models"][name]
    return Path(snapshot_download(s["repo"], revision=s["revision"], local_files_only=offline))


def check(name, folder):
    bad = []
    for rel, meta in SPEC["models"][name]["files"].items():
        p = folder / rel
        if not p.exists():
            bad.append(f"{rel}: missing")
        elif sha256(p) != meta["sha256"]:
            bad.append(f"{rel}: SHA-256 differs")
    print(f"{name}: {len(SPEC['models'][name]['files'])} files, " + ("all match" if not bad else "; ".join(bad)))
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.list:
        for n, s in SPEC["models"].items():
            gb = sum(f["bytes"] for f in s["files"].values()) / 1e9
            print(f"{n:18} {s['repo']:30} {s['revision'][:12]}  {gb:5.2f} GB  {s['used_by']}")
        c = SPEC.get("olmo_checkpoints")
        if c:
            print(f"\nOLMo-2 training checkpoints ({c['repo']}, branches): {', '.join(c['revisions'])}\n{c['about']}")
        return
    names = list(SPEC["models"]) if a.all else a.names
    unknown = [n for n in names if n not in SPEC["models"]]
    if unknown or not names:
        sys.exit(f"unknown or no model names: {unknown}; see --list")
    ok = True
    for n in names:
        ok &= check(n, snapshot(n, offline=a.check))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
