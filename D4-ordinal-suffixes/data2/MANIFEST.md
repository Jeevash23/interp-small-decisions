# D4 round-2 material manifest

Built by `code/r2_build_material.py`. Only counts were taken before freezing.

| Family | explore2 frames | held2 frames |
|---|---|---|
| rankings | 7 | 7 |
| positions | 7 | 7 |
| anniversaries | 5 | 5 |
| dates | 5 | 5 |

Numbers (rules in `code/r2common.py`, CHARTER-2.md): 2 digits all read before; 3-5 digits held2 items use
round-1 conf- and reserve-class numbers (read before, new frames); 6-7 digits round-1 reserve class are held2
(never read); 8-10 digits split by SHA-256 of `D4R2-num-<N>` mod 10, 0-5 explore2, 6-9 held2.
