"""Compare the paper's regenerated numbers and figure data with the values the manuscript was built with.

    python tools/compare_numbers.py        # after `make numbers`; exit status 1 on any difference

paper/scripts/make_numbers.py rewrites paper/numbers.tex (every value in the text, as LaTeX macros), the figure data
paper/fig-*.dat and paper/checks/numbers.json. SHA256SUMS holds the digests of the copies shipped with the release,
which are the files the manuscript was built with; this script recomputes the digests of the regenerated files.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]


def main():
    want = {}
    for line in (RELEASE / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        if rel == "paper/numbers.tex" or rel == "paper/checks/numbers.json" or (rel.startswith("paper/fig-") and rel.endswith(".dat")):
            want[rel] = digest
    bad = []
    for rel, digest in sorted(want.items()):
        got = hashlib.sha256((RELEASE / rel).read_bytes()).hexdigest()
        if got != digest:
            bad.append(rel)
    n = sum(1 for l in (RELEASE / "paper/numbers.tex").read_text().splitlines() if l.startswith("\\expandafter"))
    print(f"{len(want)} generated files compared with the shipped copies ({n} values in paper/numbers.tex): "
          + ("all identical" if not bad else f"{len(bad)} differ: {', '.join(bad)}"))
    sys.exit(1 if bad or len(want) < 3 else 0)


if __name__ == "__main__":
    main()
