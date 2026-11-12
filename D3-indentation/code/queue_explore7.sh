#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh e10_width_ablate.py qwen3-06b-bf16 240 80 H25=L25H9 STEP=L25H9,L23H3,L27H15 WRITE=L23H2,L22H8 MLP26=mlp26 MLPS=mlp22,mlp23,mlp26 --draws 8
echo EXPLORE7_DONE
