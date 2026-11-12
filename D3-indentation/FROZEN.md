# D3 freeze record

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER.md | c1c018a009beca3177e2ef119f67924d453ece04d8b957aa4bacf51226da0d12 | 2026-11-07 21:04 +0530 |

The charter was frozen before any material was built or any model was run. Later changes to it are dated amendments in `AMENDMENTS.md`, never silent edits.

Correction, 21:16: the freeze time first written here (21:12) was wrong; the file timestamps show the charter frozen at 21:04 IST. The digest is unchanged.


## Material and held-out parts

| File | SHA-256 | Sealed |
|---|---|---|
| data/gen/part-explore.jsonl | f03a89c580d4c092ca7fc95a2613c1a6c53e88ed3c5aac3c4369e19cb90a5ed5 | 2026-11-07 21:17 +0530 |
| data/gen/part-conf.jsonl | 5f39278aa638de7eea8d8af62425565f40c65d7f0bdc2e78e1a68729f8ce05e8 | 2026-11-07 21:17 +0530 |
| code/gen_programs.py | f4c581094143da4a8e669c12fba63a7379a34076ea30ec543b013ffb9000b7b8 | 2026-11-07 21:17 +0530 |
| data/newcode/fetched.jsonl | e13e8bcba8fc813957d0944fd30adf61dc963f8f62f43e6c2c0efee1836d2658 | 2026-11-07 21:36 +0530 |
| data/newcode/fetch_summary.json | db0668029b8b6b5c070c11668b7e6ada2123de884b07142b748b2c9371b3137d | 2026-11-07 21:36 +0530 |
| data/newcode/build_summary.json | f07abf2d60d6e2cd8905fe29eafcea48ac4d4229c19df5e97591c2a03cdff3e7 | 2026-11-07 21:36 +0530 |
| data/newcode/part-explore.jsonl | 25234eeeba01470a95cddcb78486ccccd8d9b1b033f67f315803dc0243a56f73 | 2026-11-07 21:36 +0530 |
| data/newcode/part-conf.jsonl | 978b49d0e7ae581349e3fdb16a574f1797e5eeef4955f89cf5ec36edbbab3ea6 | 2026-11-07 21:36 +0530 |
| data/newcode/SHA256SUMS | 0a41e9b810f790788baa643a4eba2f5bb36d3d8dc4af0a7f23cebb270f050b21 | 2026-11-07 21:36 +0530 |
| code/fetch_newcode.py | 526c5bad74ed87ad06a11894f1beb179c0caf3c4e02bf38c7bfca5b2f9a14299 | 2026-11-07 21:36 +0530 |
| code/build_material.py | ad72bf136c9d01b6d7ab65bf7668d38a8e9665781202420e14060de061f01349 | 2026-11-07 21:36 +0530 |
| code/pyindent.py | e5b57c3140bdb8d7b1da8ba2529fe8b04be19371b3f44d1356e94cdd9b1acd13 | 2026-11-07 21:36 +0530 |
| data/stdlib/part-explore.jsonl | 1583dab1e236cd94402e40e5cfe6f534f0d41484c1d2adaa1bb804fa8a62b3f5 | 2026-11-07 21:36 +0530 |

Generated programs: 1,992 explore, 2,008 conf (split by seed hash). Natural code: 1,201 files fetched 7 Nov 2026 (11.6 MB) from 314 GitHub repositories created on or after 1 Sep 2026. 710 files survived normalisation; they gave 2,991 chunks, split by repository hash: explore 1,385 chunks (123 repositories), conf 1,606 chunks (123 repositories). The raw files are digested together in `data/newcode/SHA256SUMS`. Standard-library chunks (1,127) are for exploration only.

The `conf` parts are sealed: read-only on disk, and `code/common.load_chunks` refuses them until `PROTOCOL.md` exists and is read-only. No script has read a conf part for results. The generated parts were sealed at 21:17, three minutes after a 30-program smoke test on the explore part (NOTEBOOK.md, 21:14). The natural-code parts were sealed straight after the build, before any model read them. `code/pyindent.py` is digested as the version that built the material; any later change to it is logged.


## Confirmation protocol (`PROTOCOL.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL.md | 1c0691108db0336008506c5aa7b1f6635f8dec102598abd1a147914e89f97673 | 2026-11-08 06:31 +0530 |
| AMENDMENTS.md | deed7353f39817b440ba9f6304f4f38edf359f277124dc08d95475872c323158 | 2026-11-08 06:31 +0530 |
| code/c01_ablate.py | 2649cebbcb035e93f9e71c741e291f56fc73c8c5354baa9528a532e794a33afb | 2026-11-08 06:31 +0530 |
| code/c02_pairs.py | 8a1b3eaa2b276c915bcfbc0688a97ac3daa5aea26d2c5ccc47578f561ea9b4e9 | 2026-11-08 06:31 +0530 |
| code/c03_verdicts.py | e37136fb8dd603365f05bde2e39c8787a0d27dc9d1aaaf390a44c761e3f6947c | 2026-11-08 06:31 +0530 |
| code/common.py | e90eff35831d101b8bd31f71bd16299e2733cabee7b7b98fa5d19437219c51f0 | 2026-11-08 06:31 +0530 |
| code/pyindent.py | e5b57c3140bdb8d7b1da8ba2529fe8b04be19371b3f44d1356e94cdd9b1acd13 | 2026-11-08 06:31 +0530 |
| code/run.sh | d8b0a23ce710994a200a7af30099b62b7304e503a09b21fbedea86e75b13edd5 | 2026-11-08 06:31 +0530 |
| code/gemma_load.py | ec9d83733effcb9eda390e25e6ee02ee46117f6888cb8aab89fb0e16b52ad922 | 2026-11-08 06:31 +0530 |
| data/newcode/part-conf.jsonl | 978b49d0e7ae581349e3fdb16a574f1797e5eeef4955f89cf5ec36edbbab3ea6 | 2026-11-08 06:31 +0530 |
| data/gen/part-conf.jsonl | 5f39278aa638de7eea8d8af62425565f40c65d7f0bdc2e78e1a68729f8ce05e8 | 2026-11-08 06:31 +0530 |

Frozen on 8 Nov 2026 before any script read a sealed confirmation part. The two `part-conf` digests match the seal of 7 Nov (21:17 and 21:36 above). `code/common.py` differs from the material-build version by Amendment 1 (the blank-line fix), the vectorised `RootMasses` (same arithmetic), the bfloat16 model entry and the Gemma loader import. `code/pyindent.py` is unchanged since the material seal. Pipeline smoke tests of c01–c03 ran on exploration material only (`outputs/conf_smoke/`, not results). No `outputs/conf/` directory existed at freeze time.


## Round 2 charter (`CHARTER-2.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER-2.md | 2a725c1a68f8436fe78636bf873569930b48be9b7e11e00dc4cd46470f7d6cd8 | 2026-11-08 16:54 +0530 |

Frozen (made read-only) at 16:54 IST, before any round-2 material was built or any round-2 run. The charter's own text says "about 17:05"; that was an estimate written before the freeze, and the true freeze time is 16:54 (from `date`).

## Round 2 held-out material

| File | SHA-256 | Sealed |
|---|---|---|
| data/r2gen2/part-held.jsonl (1,510 targeted programs) | beb447d25b348e75bf684c82d63b2c336c9bdf58ecadb297ecb6b2011db04025 | 2026-11-08 17:02 +0530 |
| data/r2gen/part-held.jsonl (4,000 programs, seeds 200000–203999) | c7f9083b85a9adc4adf5acb31d3741c89b1d7d598d9a402323d34fcfa8ff5e79 | 2026-11-08 17:02 +0530 |
| code/gen2.py | 384138ae0452e1774c38dce3f268c8273751653f796fc23f32c8d8b87e1ef730 | 2026-11-08 17:02 +0530 |
| code/r2_build.py | 5ad4b31c2678b524c31ad190389ce6a14cca46560336c170b037b223b327ca6e | 2026-11-08 17:02 +0530 |
| data/r2gen2/part-explore.jsonl (1,490 programs; exploration, digested for the record) | 7b3c85557cb89f07778fbe3e9ebd94f7b286a3a43bb426aaa9d4c162baececcd | 2026-11-08 17:02 +0530 |

Both generated held-out parts were sealed (read-only) before any model read any round-2 material. The local machine restarted at about 16:57, during the first build; `gen2.py` was rerun at 17:01 and gave byte-identical files. The natural-code part follows once the fetch finishes.

| File | SHA-256 | Sealed |
|---|---|---|
| data/r2code/part-held.jsonl (2,712 chunks, 225 repositories) | 04d551b6b47829d23751e96e43be3857755768a22e90120901984f933fdb38a5 | 2026-11-08 17:19 +0530 |
| data/r2code/fetched.jsonl (1,202 files, 335 repositories, 10.9 MB) | b6a1e33debe58b1ad26415c13dd341b13dbfb6033af6dc59e3885805999f292f | 2026-11-08 17:19 +0530 |
| code/r2_fetch_code.py | 86d6244531d7a3c2a8a1255caddb44ff02fdcb539744f4524d45a16a810e0db3 | 2026-11-08 17:19 +0530 |

The natural-code held-out part was sealed straight after the build, before any model read it. Query: `language:Python created:>=2026-09-01 fork:false archived:false stars:5..9`. 500 repositories were listed and none was in round 1's or X1's lists. One chunk matched an older hash and was dropped; 175 exact duplicates within the build were dropped.

## Round 2 confirmation protocol (`PROTOCOL-2.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-2.md | 885e0b117de9f3d3a01bd34d200ff4c04dff5c2647ce13141c59bf7307daf99c | 2026-11-09 09:55 +0530 |
| AMENDMENTS-2.md | 748a408f8c4d6e73e19301db2606de99d03051dbe7b13d93ea7dff455aad12a5 | 2026-11-09 09:55 +0530 |
| code/r2c01_header.py | beffca84288d35aefd7d6f469ed0f7230986b8de944896eaa4db82d1c1670954 | 2026-11-09 09:55 +0530 |
| code/r2c02_gen2.py | b9198f56de061e5ee79a94468ddf9bf58886a30c22440b7c0ba5d6a0678637fe | 2026-11-09 09:55 +0530 |
| code/r2c03_group.py | 2f731490b882a679c79c2e255bfdfae9e281a3a28d7da56555e24819c381231d | 2026-11-09 09:55 +0530 |
| code/r2c04_verdicts.py | b70dd9f45dd2028db0b8b5110b04698a6e21b565e2eef5a7e701c0afc253ff0c | 2026-11-09 09:55 +0530 |
| code/r2common.py | 5c3ec1088c1bfcdf89523fe964a32c48b5b63f0f329c9b5db6579b6f55a88e3d | 2026-11-09 09:55 +0530 |
| code/common.py | e90eff35831d101b8bd31f71bd16299e2733cabee7b7b98fa5d19437219c51f0 | 2026-11-09 09:55 +0530 |
| code/pyindent.py | e5b57c3140bdb8d7b1da8ba2529fe8b04be19371b3f44d1356e94cdd9b1acd13 | 2026-11-09 09:55 +0530 |
| code/gen2.py | 384138ae0452e1774c38dce3f268c8273751653f796fc23f32c8d8b87e1ef730 | 2026-11-09 09:55 +0530 |
| code/run.sh | d8b0a23ce710994a200a7af30099b62b7304e503a09b21fbedea86e75b13edd5 | 2026-11-09 09:55 +0530 |
| code/r2q_conf.sh | e34d4f2095fe8ec3f342791db31456f8adc27833eff915fcdadf6540356c3cfb | 2026-11-09 09:55 +0530 |

Frozen (made read-only) before any script read a sealed round-2 part; no `outputs/r2/conf/` directory existed at freeze time. `code/gen2.py` matches its seal of 8 Nov 17:02. `code/common.py`, `code/pyindent.py` and `code/run.sh` are unchanged since round 1's protocol freeze. `AMENDMENTS-2.md` holds R2-1 and R2-2 at freeze time; later round-2 amendments go in a new dated file. Smoke tests (`outputs/r2/conf_smoke/`) and calibration runs (`outputs/r2/conf_calib_*/`) read exploration material only.

## Round 3 charter (`CHARTER-3.md`, replication on shareable material)

| File | SHA-256 | Frozen |
|---|---|---|
| CHARTER-3.md | f52bb1b546930a458b3b99da1c31e2fb3889e10aa9bb83cfaa40520ed2f1ab6d | 2026-11-10 20:19 +0530 |

Frozen (read-only) before any round-3 material was fetched or generated and before any round-3 run. Plan: `ROUND-3-PLAN.md`.

Correction, 20:19: the charter's first line gives its writing time as about 20:35 IST; that was written ahead of the clock. The true freeze time is 20:19 IST (from `date`). The digest is unchanged.

## Round 3 held-out material

| File | SHA-256 | Sealed |
|---|---|---|
| round3/data/gen/part-held.jsonl | 53a3a562ad3acaa712315b117f824b4bcb1f0873e832c30297a067c203dc2c10 | 2026-11-10 20:20 +0530 |
| round3/data/gen2/part-held.jsonl | 7ee28719cffffb122e8d2bdd551fe5daf7537ea7a721689afe3dc3d01d6664eb | 2026-11-10 20:20 +0530 |
| round3/code/r3_build.py | 0b497dd19de5dfa3731234b033969e94c7364723d21a22d91530a82ce166d3dd | 2026-11-10 20:20 +0530 |
| round3/code/r3_fetch.py | f1157d6ed1ac795d8f6adb15a31f60b713862bd900d3459d29a4b6e6f8656298 | 2026-11-10 20:20 +0530 |

Generated parts (2,000 programs from `gen_programs.G`, seeds 300000–301999; 1,500 from `gen2.program`, seeds 10000–11499) sealed (read-only) straight after the build, before any model read them. The natural-code part follows once the fetch finishes.

| File | SHA-256 | Sealed |
|---|---|---|
| round3/data/code/part-held.jsonl | 5b5b036ad3c0da8f78b7cbe062596967e3822cd88c31422a16b75a27c3f447e5 | 2026-11-10 20:58 +0530 |
| round3/data/code/fetched.jsonl | 23089f48314028f4715cdd7dfb691be666e37c558ebbd0b01ae5902f7e1658a2 | 2026-11-10 20:58 +0530 |
| round3/data/code/dropped.jsonl | a5a10a5389fb6186141b014dd47f21efc6188c91aaa0e530bc3894e75dca2027 | 2026-11-10 20:58 +0530 |
| round3/data/code/fetch_summary.json | 8ce78a59e642f7c412aa8d9f327ead27f7031be8eeebd4de843a8b8c0b99a534 | 2026-11-10 20:58 +0530 |
| round3/data/code/build_summary.json | 373f8773263c4ac067a619260516223510d49b96e5c5ffa7a0c0b06f197d4990 | 2026-11-10 20:58 +0530 |
| round3/data/code/SHA256SUMS | 181ad831bc234698af1240065137065cb2f9d8318ca1dd44e24fadd5527313a4 | 2026-11-10 20:58 +0530 |

The natural-code part was built and sealed (read-only) straight after the fetch (finished 20:58 IST, 1,000 files, 9,460,326 bytes, 274 repositories, all with licence MIT, Apache-2.0, BSD-2/3-Clause, CC0-1.0 or Unlicense by GitHub's licence endpoint), before any model read it. The raw files are digested together in `round3/data/code/SHA256SUMS`; licence texts and notices are in `round3/data/code/licences/`.

## Round 3 confirmation protocol (`PROTOCOL-3.md`)

| File | SHA-256 | Frozen |
|---|---|---|
| PROTOCOL-3.md | 43a309455c7378d609568e83187da50839e6095f026dfdfc98c2b2aa4a14c374 | 2026-11-10 21:37:43 +0530 |
| round3/code/r3x_c01.py | c979d9a8316f76662bd119ccad85118db4522a3242a38cef7ed0f4b1c672aae8 | 2026-11-10 21:37:43 +0530 |
| round3/code/r3x_l11h9.py | bd0c995a10e0f2857bff4717ead2517ebac3dff0348d57c0e927e8f5aad2fc20 | 2026-11-10 21:37:43 +0530 |
| round3/code/r3_pairs.py | 764d9b2f1fb5e6b7766eddec8606d7ccf7e46e7a62dc71638990495d0c451c85 | 2026-11-10 21:37:43 +0530 |
| round3/code/r3_verdicts.py | 5dad50f4c0055a1ae3d6983d8cc7318a24f918791957624a2889e35fe93e16cc | 2026-11-10 21:37:43 +0530 |
| round3/code/r3q_conf.sh | c1eb692a7ebb7f1fd530d8c8e812499a1ae1a0c1430153b857a03b41e72fa31c | 2026-11-10 21:37:43 +0530 |
| round3/code/r3_freeze.sh | bcc9ff2ef4d5b84acbc8c4d9d2ecaea626e0a589814918dab8f55d09089f9416 | 2026-11-10 21:37:43 +0530 |
| round3/code/e01_behaviour.py | d3c3d225c3a5bb9710fd397d2090b91dbcdf05a4cc43e2c853ad887f21391a91 | 2026-11-10 21:37:43 +0530 |
| round3/code/gate_summary.py | 5bdd073e1b20907969f7b2e551fa1b3311f2271e5b4a94201228987ff864cfad | 2026-11-10 21:37:43 +0530 |
| round3/code/common.py | e90eff35831d101b8bd31f71bd16299e2733cabee7b7b98fa5d19437219c51f0 | 2026-11-10 21:37:43 +0530 |
| round3/code/pyindent.py | e5b57c3140bdb8d7b1da8ba2529fe8b04be19371b3f44d1356e94cdd9b1acd13 | 2026-11-10 21:37:43 +0530 |
| round3/code/gemma_load.py | ec9d83733effcb9eda390e25e6ee02ee46117f6888cb8aab89fb0e16b52ad922 | 2026-11-10 21:37:43 +0530 |
| round3/code/c01_ablate.py | 2649cebbcb035e93f9e71c741e291f56fc73c8c5354baa9528a532e794a33afb | 2026-11-10 21:37:43 +0530 |
| round3/code/c02_pairs.py | 8a1b3eaa2b276c915bcfbc0688a97ac3daa5aea26d2c5ccc47578f561ea9b4e9 | 2026-11-10 21:37:43 +0530 |
| round3/code/c03_verdicts.py | e37136fb8dd603365f05bde2e39c8787a0d27dc9d1aaaf390a44c761e3f6947c | 2026-11-10 21:37:43 +0530 |
| round3/code/r2common.py | 5c3ec1088c1bfcdf89523fe964a32c48b5b63f0f329c9b5db6579b6f55a88e3d | 2026-11-10 21:37:43 +0530 |
| round3/code/r2c01_header.py | beffca84288d35aefd7d6f469ed0f7230986b8de944896eaa4db82d1c1670954 | 2026-11-10 21:37:43 +0530 |
| round3/code/r2c02_gen2.py | b9198f56de061e5ee79a94468ddf9bf58886a30c22440b7c0ba5d6a0678637fe | 2026-11-10 21:37:43 +0530 |
| round3/code/r2c03_group.py | 2f731490b882a679c79c2e255bfdfae9e281a3a28d7da56555e24819c381231d | 2026-11-10 21:37:43 +0530 |
| round3/code/r2c04_verdicts.py | b70dd9f45dd2028db0b8b5110b04698a6e21b565e2eef5a7e701c0afc253ff0c | 2026-11-10 21:37:43 +0530 |
| round3/code/run.sh | d8b0a23ce710994a200a7af30099b62b7304e503a09b21fbedea86e75b13edd5 | 2026-11-10 21:37:43 +0530 |
| round3/code/r2q_conf.sh | e34d4f2095fe8ec3f342791db31456f8adc27833eff915fcdadf6540356c3cfb | 2026-11-10 21:37:43 +0530 |
| round3/x1/outputs/means_qwen3-06b-bf16_IND_dec.pt | 7275b22dd7e24deb58a394d86c3bc1c4e03dc1a009134d10737bfbd7a63a3c50 | 2026-11-10 21:37:43 +0530 |
| ../X1-shared-parts/code/c01_confirm.py | 0983678339d40e6c1086d7908badbee5e777a1206f7d8485c719f8112a1e519a | 2026-11-10 21:37:43 +0530 |
| ../X1-shared-parts/code/x1common.py | d50c8fa55fe11f9f374ba60d0f85f1d31358b9d8fb732a614843b1a8c403a036 | 2026-11-10 21:37:43 +0530 |

Frozen (read-only) before any model read round-3 material; no `round3/outputs/conf/` existed. The copies in `round3/code/` other than the r3 scripts and the two gate scripts have the same digests as their earlier freezes above; the means file and X1's two scripts match `X1-shared-parts/FROZEN.md`. After this entry the slot links and the copies `round3/PROTOCOL.md` and `round3/PROTOCOL-2.md` were created (PROTOCOL-3.md, How the frozen scripts are run).
