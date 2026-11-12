# D5 freeze record

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER.md | d8afc0a091acef248e25cd7e1c29983fea5c1c997717bc25fca3c05603275b14 | 2026-11-07 21:02 +0530 |
| code/build_data.py | 938d8400f4ae60957e3c968a939407fb5b466c9c86054310439b805829d7902e | 2026-11-07 21:02 +0530 |
| data/part-explore.jsonl | 32872185257f374c98e48f9f4887e9c1c8dc3e1b015ff2f4d076320008b79209 | 2026-11-07 21:02 +0530 |
| data/part-conf.jsonl | cc805b829bcf712056d1a203f4af256d393c694bacc47bcdb27bf31c5c94adb4 | 2026-11-07 21:02 +0530 |
| data/part-reserve.jsonl | 0869e1abf78a7be531f74e76093470ff800216d06546e28bc01943d009522745 | 2026-11-07 21:02 +0530 |
| data/build_summary.json | a2319329f902fdfeedb3c0762c44fb393c2ec6220d9ddb330f1a01e7d5b8a570 | 2026-11-07 21:02 +0530 |

Frozen before any model was run on D5 material. The split files were written by `code/build_data.py`, which prints counts only; no script has read them for results. `conf` is held out for `PROTOCOL.md` and `reserve` for a later round; the loader enforces both. Changes to the charter are dated amendments in `AMENDMENTS.md`.

## Confirmation protocol (frozen 2026-11-08 05:48 +0530)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL.md | 720cdb6aaa45dc1162cc6f2f9a2da6135d789ba7c2322129e6cd9b7937de57cb | 2026-11-08 05:48 +0530 |
| code/d5common.py | e25b9939c1c901ef9183cf0037696bb846aa78a77dee8a5376892db0b627fd88 | 2026-11-08 05:48 +0530 |
| code/run.sh | 12150cd2aa8e304fc8cc3a823103b641bc5329e1418663d6497037ad602d96ff | 2026-11-08 05:48 +0530 |
| code/conf_gate.py | 26ab586fa1656e39336aa453f55a524feee61a7c7caf34e02b1643a35101e6cb | 2026-11-08 05:48 +0530 |
| code/conf_exp01_lens.py | 03a927cdab917e16bdb8e9e548130cefdac57722f340f71d3df40543f27152c3 | 2026-11-08 05:48 +0530 |
| code/conf_exp02_direction.py | 1ddf870e9b39946b6d5c1740f858cb74ab93fbbce70eef662b37c0eef326bc4b | 2026-11-08 05:48 +0530 |
| code/conf_exp04_switch.py | a3430b6bc2d163e4d9a56e01fceb757ed2149f3e5a29a1ecb263779aff63c813 | 2026-11-08 05:48 +0530 |
| code/conf_exp06_head.py | 52e9456c97295db38a9d50c71a2b729465ab7ad570a64a2b71a5031c3a194497 | 2026-11-08 05:48 +0530 |
| code/conf_exp07_ablate.py | f3d28d4998ff21dee9687de7f7d2ff4b745bb56592ed53b1d32f796527743ede | 2026-11-08 05:48 +0530 |
| code/conf_exp08_errors.py | cb6ebf1d50567bd64ce84538b2f117dcbaf62a6070401db247e0c712f2aa93af | 2026-11-08 05:48 +0530 |
| code/conf_verdict.py | 4c82e3c39f86a96699303db6abacab3841ce467a98ae1413d61259f2358693cf | 2026-11-08 05:48 +0530 |
| outputs/exp01_gemma_means.pt | 2745c19f8e166c7a56d4cde2ce73d12c261c6a0b79dabebf7207699c3a227e19 | 2026-11-08 05:48 +0530 |
| outputs/exp01_qwen3-17b_means.pt | be6b3bf565d30b2b6e5de8f1c3b6869686eb8f202fcfd08461c3d1c9dcbbf70e | 2026-11-08 05:48 +0530 |
| ../D1-line-breaking/xmodel/code/xm3_common.py | cb96c1b1eeca68c50fa3f4c4690fd93e98e4d72f3a97791ddb2c51b0d52dd309 | 2026-11-08 05:48 +0530 |
| ../D1-line-breaking/xmodel/code/xm_common.py | b16c86dccd6a32d9a294c7b458a22fbf10bd5bcc7fdafba0c8c699360a8ea24d | 2026-11-08 05:48 +0530 |

Frozen before any script read `data/part-conf.jsonl` for results. The exploration means are inputs (the case directions), not results of the confirmation run. D1-X's loader files are digested as used, not changed by D5.

## Round 2 charter (frozen 2026-11-08 17:02 +0530)

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER-2.md | aaa0c4e68dd7ad7bf4cf239dbc45ec75d065e626d6ff5d368d0f11397843c5da | 2026-11-08 17:02 +0530 |
| code/build_data2.py | 2ad0d75c19b4f688c720d089bb6ae4c4e0add4945165a255e692d5a286e62fb3 | 2026-11-08 17:02 +0530 |
| data/headings-explore.jsonl | 1b5c5bae27a70d5cee2059c185fd96dd3d569b2437aab6bf4b6a20fcdd7f6207 | 2026-11-08 17:02 +0530 |
| data/headings-conf.jsonl | fdc4a26daee5a07c8418937113a0e86a694deed2ef6f8bccb5dde6823aaf1dea | 2026-11-08 17:02 +0530 |
| data/headings-reserve.jsonl | d3cdc3e7ed17539c0b0554fb7db8a786586888e5a01598c9ae7da13629562335 | 2026-11-08 17:02 +0530 |
| data/headings_summary.json | ccc6405cfeb5c7f6dcc1805070f5b6616c523e32c1d5c40452a5640c12362e44 | 2026-11-08 17:02 +0530 |
| data/part-reserve.jsonl | 0869e1abf78a7be531f74e76093470ff800216d06546e28bc01943d009522745 | 2026-11-08 17:02 +0530 |

Frozen before any round-2 measurement. The heading files were written by `code/build_data2.py`, which prints counts only. `data/part-reserve.jsonl` is unchanged since its round-1 freeze (same digest) and has been read by no model or script; with `data/headings-reserve.jsonl` it is held out for `PROTOCOL-2.md`, which `code/d5common2.py` enforces. Changes to this charter are dated amendments in `AMENDMENTS-2.md`.

## Round 2 confirmation protocol (frozen 2026-11-08 20:26 +0530)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-2.md | dd7f101130e96660d0947fe8395a2d9768c7a45576ed9e60f46f0b1dccdc92e7 | 2026-11-08 20:26 +0530 |
| code/d5common2.py | 8dde4630d3d981d9dc4c14758e0847644f245256abc3e8b7ccd154810a647e29 | 2026-11-08 20:26 +0530 |
| code/run2.sh | 88563b5d5ddb5523463fcf00c0a2fcc69d72ed1f49e4940cf6b89fd76073ae2f | 2026-11-08 20:26 +0530 |
| code/r2_conf.sh | 01dbb205cdc31165419de222f1a3c626388e06871060bdb3fbbc597f8e47c4bc | 2026-11-08 20:26 +0530 |
| code/r2_exp03_sets.py | 6f97c976cf920b92de93fc2883515341a376637a9183c645bc438790f3ab390b | 2026-11-08 20:26 +0530 |
| code/r2_exp04_other.py | 002a25cb42304a112312418bef10429f50e6a5d8d97b70290cfc1dfa4285a5b0 | 2026-11-08 20:26 +0530 |
| code/r2_exp05_edit.py | db4eab2626adf84d4b1871da435516e4091cf441aafd2c9c55ea1e82b97ae155 | 2026-11-08 20:26 +0530 |
| code/r2_exp06_attn.py | 455c994fb02c055a371044c40398769833428445fa6f599e72883008fed08573 | 2026-11-08 20:26 +0530 |
| code/r2_verdict.py | 423275b19e26a23509e5b7fa4584baf062258851bcb67980f165303ddb18af61 | 2026-11-08 20:26 +0530 |
| code/r2_sets_qwen.json | a92662fd15cc4d1772e5310db691b7402f636f58f1fbb7786835324beda0b26a | 2026-11-08 20:26 +0530 |
| code/r2_sets_gemma.json | 29d46dc0feff55ca65c7b18b61c057735a7f14f3a11b9e32f810a0916ba1749a | 2026-11-08 20:26 +0530 |
| code/r2_sets_qwen_l17h6.json | fd7257235b32c9fa96fa25746667931eafbfaedcc83b472a58720c7b96669589 | 2026-11-08 20:26 +0530 |
| outputs/r2/explore_verdicts2_check.json | 001a170fd943806928baee5b3a512ca4f5dbd2e952e84b58420063c6ff54ec12 | 2026-11-08 20:26 +0530 |

Frozen before any model or script read the held-out reserve paragraphs or reserve heading items. Round 1's frozen files used here (`code/d5common.py`, the exploration means, D1-X's loaders) are unchanged (digests as in round 1's section). `outputs/r2/explore_verdicts2_check.json` is the verdict script run on the exploration outputs, the source of the bracketed exploration values in PROTOCOL-2.md. (A first attempt at 20:26 to write this record failed on a shell quoting error and was removed; a first start of `code/r2_conf.sh` at 20:26 exited at once, file not found, before reading anything.)
