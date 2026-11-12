# D4 freeze record

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER.md | ce602455294779258175a444a38a6c893673fbce1055c553912987514b94a665 | 2026-11-07 21:01 +0530 |

Frozen before any model was run on D4 material. Only the tokenisers were run beforehand (the tokenisation facts in the charter). Later changes to the charter are dated amendments in `AMENDMENTS.md`, never silent edits. Frozen files are made read-only.

## Material (`data/`)

| File | SHA-256 | Frozen |
|---|---|---|
| code/build_material.py | d27654ce3ebdf9d8e9aa242a91293b07da3210974bf082d2a7e8c95d29aa2e39 | 2026-11-07 21:02 +0530 |
| data/frames.jsonl | 0cb3ab1cf7dd0277d953423baad3432add24fb734dd61405627f6ce6223b926d | 2026-11-07 21:02 +0530 |
| data/MANIFEST.md | 91ec131e976df04e315a56a333e38e6cf961a7abb2242dd6a2013dbf563da60b | 2026-11-07 21:02 +0530 |
| data/numbers.json | a1173fad04dcbf0cc37209a836ec5e017a8eb2e00c8f7b49243eac8999897074 | 2026-11-07 21:02 +0530 |
| data/part-conf.jsonl | 0a6c06e3139f5c45449638766b1decb72ba43070021fae1faa1f3190a786ab54 | 2026-11-07 21:02 +0530 |
| data/part-explore.jsonl | 8a8ac086601f76c556f1787ea7c20943f102fbd67fb7d810a65faa62762cacc1 | 2026-11-07 21:02 +0530 |
| data/part-reserve.jsonl | 729004f4b3416685ca41c288d3878eb866a36d4ae3bac2f61271efc85711b99a | 2026-11-07 21:02 +0530 |
| data/SHA256SUMS | 82390695747b82269a73bf14d5a7b26afa7f74b6616da9de8efd1c4574bee1e8 | 2026-11-07 21:02 +0530 |

Frozen as soon as the parts were built, before any model read them; only item counts were taken (`data/MANIFEST.md`). `part-conf` and `part-reserve` are read only under a frozen `PROTOCOL.md`; the loader in `code/common.py` refuses them until `PROTOCOL.md` exists and is read-only.

## Amendment 1 (`AMENDMENTS.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| AMENDMENTS.md | d783f19e3788af591340e8c5b5c29bb8f45480246db4af48808e9ca27929bb4d | 2026-11-07 21:29 +0530 |

Frozen before any model read a number above 9,999.

Note added 2026-11-07 21:52 +0530: the heading of Amendment 1 in `AMENDMENTS.md` says "about 21:50 IST"; that time is wrong. The file was saved and frozen at 21:29 IST (file timestamp and the table above). The frozen file is left unchanged.

## Confirmation protocol (`PROTOCOL.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL.md | 5fd0ded833c005bd5ca364c1a143242d827db0a3c2c57baf6e27776dcaf84c6e | 2026-11-08 04:16 +0530 |
| code/common.py | c518ef04838cc75337bf6011cbf14211532f3a84717d211053347f5abde37116 | 2026-11-08 04:16 +0530 |
| code/hooks.py | a9d8b1848a922f8a0dca9d03df20d852f902d1711c8811c75c12baa030b686ee | 2026-11-08 04:16 +0530 |
| code/teen_effect.py | 75772e6d423678b10540874418c4028bb1c50202a580f80d3f146a5afebcb957 | 2026-11-08 04:16 +0530 |
| code/conf01_boundary.py | 01b1ffbb670e151ddd81371b5913e355901392d5ca4a890cdc25e4b9917f1d56 | 2026-11-08 04:16 +0530 |
| code/exp05_ablate.py | c68397d6e859accc094783b7c90b1cb36300e8e10ddc7703e95b4ba7ee29fda6 | 2026-11-08 04:16 +0530 |
| code/exp04_scan.py | 743ceb12670a58418627589d3fea1ae3c08c216ec45a63d64af4e8fb3061e8e8 | 2026-11-08 04:16 +0530 |
| code/exp01_patch_pairs.py | 073a45c5648edaf7eead6cd5d9fbd7d6be8e3b153c7e9a24e7e97a1986ff02ca | 2026-11-08 04:16 +0530 |
| code/run.sh | 45bab7512a246341a857121bcd4aabc2337823c92125fdaf859967ace41eba2e | 2026-11-08 04:16 +0530 |
| code/chain_conf.sh | fec46334f27300a353b2ac57bb50483395dfdd3857da02686a4800364f1f6d0f | 2026-11-08 04:16 +0530 |
| code/conf_evaluate.py | adfaf617eb8af0fa213ca8cf7decf8b27e14b9bc2d9c13537d70fe6efba99b8b | 2026-11-08 04:16 +0530 |
| code/check_report.py | 9760f31826e24d45a8dac52a63f0072b8655b19b08b2f93188774e60ea8673c5 | 2026-11-08 04:16 +0530 |

Frozen before any model read a confirmation frame or a confirmation number above 99. `code/chain_conf.sh` runs each confirmation job once.

## Round 2 charter (`CHARTER-2.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER-2.md | 5f4137bae307a4ea6159ffba904185994206a0ada85432271f95f79436084f9e | 2026-11-08 16:54 +0530 |

Frozen before any model was run on round-2 material.

Note added 2026-11-08 16:56 +0530: `CHARTER-2.md` says it was written "about 17:05 IST"; that time is wrong. It was written and frozen at 16:54 IST (`date` and the table above). The frozen file is left unchanged.

## Round 2 material (`data2/`)

| File | SHA-256 | Frozen |
|---|---|---|
| code/r2_build_material.py | b74fe134b9e7d6cc414f39a68e87989eeefd283f3296975dda9468fd340c510a | 2026-11-08 17:01 +0530 |
| code/r2common.py | d502fee93fcecdf00589a9dccd14d8890383a7ae9c475ffa1b7b2dacc56d00ca | 2026-11-08 17:01 +0530 |
| data2/frames2.jsonl | fd5831171747233f250d41a715466e7dbf45abd9d07248f0c74bd3790a7b3072 | 2026-11-08 17:01 +0530 |
| data2/MANIFEST.md | 7c2437556cdda8c9acd528379ae3549d68ba47ba42074328cf82fffc372c34b2 | 2026-11-08 17:01 +0530 |
| data2/SHA256SUMS | 3089b72dfbe7c7f2e253a16e11af0b55c2226594e776e3e2f629dc73ecf51ec7 | 2026-11-08 17:01 +0530 |

Frozen before any model read a round-2 frame. Built 16:55; the local machine restarted at about 16:57, and on resuming (17:01) the digests were checked and a rebuild gave identical bytes. `code/r2common.py` refuses held2 frames and held-out numbers until `PROTOCOL-2.md` exists and is read-only.

## Round 2 confirmation protocol (`PROTOCOL-2.md`), frozen 9 Nov 2026, 09:34 IST

Read-only from this time. `AMENDMENTS-2.md` (Amendment 1) is listed for the record; it stays writable for later amendments.

| File | SHA-256 |
|---|---|
| PROTOCOL-2.md | ca7641945d6a4db6a464896c2b9d83c61e14704915b27c27e1af0f16e3d529f7 |
| AMENDMENTS-2.md | bf79cc557b1a7527ff44aaa79e549e8f814710c59f75c29226e9d7cfbd392e6b |
| code/r2lib.py | 85db49f2f6bbab32341cc3ce48790778c7648a5d18f54eeb07ab6f2e902def2a |
| code/r2conf_run.py | b8acc462ab25c6ddefba5475f93f24d5cfd0715787c988e23612efe3cc699fe3 |
| code/r2conf_evaluate.py | 2481b7d59635d1ab174ae1355dc8a5c99413ad9635ccdc55b1abbd13f8bef743 |
| code/r2chain_conf.sh | e930bdd27a8c442f8f159ec4ccb267903bafb4ff63958a99e3542dd65da4ba58 |
| code/run.sh | 45bab7512a246341a857121bcd4aabc2337823c92125fdaf859967ace41eba2e |
| code/r2conf_sets.json | 7f1c3356db502914eb995649b10c3663c6b022096c3a6d0c15441590b1861f60 |
| code/r2common.py | d502fee93fcecdf00589a9dccd14d8890383a7ae9c475ffa1b7b2dacc56d00ca |
