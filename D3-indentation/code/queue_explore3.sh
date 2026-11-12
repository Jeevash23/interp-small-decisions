#!/usr/bin/env bash
# D3 exploration queue 3 (Qwen3-0.6B-Base): finish e02, direct logit attribution, group ablations.
cd "$(dirname "$0")"
./run.sh e02_ablate.py qwen3-06b-bf16 newcode 160 --time-limit 1000
./run.sh e07_dla.py qwen3-06b-bf16 newcode 250
./run.sh e07_dla.py qwen3-06b-bf16 gen 250
./run.sh e06_group.py qwen3-06b-bf16 newcode 120 OPEN=L19H3,L20H14,L22H9,L22H1,L20H15,L21H9 PREV=L21H6,L22H8,L1H10,L3H9 KW=L1H0,L5H6,L3H3,L3H2 DED=L23H2,L17H3,L17H4,L17H5,L18H6 --draws 12
echo EXPLORE3_DONE
