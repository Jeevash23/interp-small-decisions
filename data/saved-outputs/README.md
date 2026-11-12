# Saved outputs

The studies' per-run outputs that the paper's numbers come from, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code and the paper's number scripts read it, and checks each SHA-256 against `data/PACKED-FILES.json`. The members are byte for byte the studies' own files.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `d3-outputs-conf.tar.xz` | 12 | 4.8 | 19.3 | 12 saved outputs of D3 (indentation), folder conf |
| `d3-outputs-main-1.tar.xz` | 3 | 36.9 | 169.4 | 3 saved outputs of D3 (indentation), folder main-1 |
| `d3-outputs-main-2.tar.xz` | 6 | 43.0 | 197.0 | 6 saved outputs of D3 (indentation), folder main-2 |
| `d3-outputs-main-3.tar.xz` | 21 | 4.8 | 21.9 | 21 saved outputs of D3 (indentation), folder main-3 |
| `d3-outputs-r2.tar.xz` | 32 | 54.0 | 198.6 | 32 saved outputs of D3 (indentation), folder r2 |
| `d3-outputs-v1-before-2230-fix-1.tar.xz` | 4 | 40.0 | 179.7 | 4 saved outputs of D3 (indentation), folder v1-before-2230-fix-1 |
| `d3-outputs-v1-before-2230-fix-2.tar.xz` | 3 | 8.3 | 38.2 | 3 saved outputs of D3 (indentation), folder v1-before-2230-fix-2 |
| `d3-round3-outputs-conf.tar.xz` | 6 | 0.0 | 0.1 | 6 saved outputs of D3 (indentation), folder conf |
| `d3-round3-outputs-main-1.tar.xz` | 3 | 37.1 | 170.6 | 3 saved outputs of D3 (indentation), folder main-1 |
| `d3-round3-outputs-main-2.tar.xz` | 4 | 42.9 | 199.7 | 4 saved outputs of D3 (indentation), folder main-2 |
| `d3-round3-outputs-main-3.tar.xz` | 5 | 27.2 | 128.8 | 5 saved outputs of D3 (indentation), folder main-3 |
| `d3-round3-outputs-r2.tar.xz` | 7 | 49.3 | 180.9 | 7 saved outputs of D3 (indentation), folder r2 |
| `d3-round3-outputs-x1-conf.tar.xz` | 2 | 0.0 | 0.2 | 2 saved outputs of D3 (indentation), folder x1-conf |
| `d4-outputs-conf.tar.xz` | 12 | 0.3 | 2.6 | 12 saved outputs of D4 (ordinal suffixes), folder conf |
| `d4-outputs-main.tar.xz` | 27 | 1.6 | 13.5 | 27 saved outputs of D4 (ordinal suffixes), folder main |
| `d4-outputs-r2.tar.xz` | 7 | 0.0 | 0.0 | 7 saved outputs of D4 (ordinal suffixes), folder r2 |
| `d4-outputs-r2conf.tar.xz` | 9 | 0.0 | 0.2 | 9 saved outputs of D4 (ordinal suffixes), folder r2conf |
| `d4-outputs-r2post.tar.xz` | 1 | 0.0 | 0.0 | 1 saved outputs of D4 (ordinal suffixes), folder r2post |
| `d5-outputs-main-1.tar.xz` | 1 | 13.4 | 158.0 | 1 saved outputs of D5 (capitals), folder main-1 |
| `d5-outputs-main-2.tar.xz` | 1 | 12.8 | 152.7 | 1 saved outputs of D5 (capitals), folder main-2 |
| `d5-outputs-main-3.tar.xz` | 1 | 10.2 | 127.9 | 1 saved outputs of D5 (capitals), folder main-3 |
| `d5-outputs-main-4.tar.xz` | 30 | 11.0 | 130.5 | 30 saved outputs of D5 (capitals), folder main-4 |
| `d5-outputs-r2.tar.xz` | 24 | 4.6 | 15.4 | 24 saved outputs of D5 (capitals), folder r2 |
| `x1-outputs-conf.tar.xz` | 10 | 3.2 | 48.2 | 10 saved outputs of X1 (shared parts), folder conf |
| `x1-outputs-main.tar.xz` | 8 | 0.9 | 1.2 | 8 saved outputs of X1 (shared parts), folder main |

Total packed size: 406.5 MB.
