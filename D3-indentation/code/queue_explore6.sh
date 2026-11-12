#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh e07b_dla_split.py qwen3-06b-bf16 newcode 250
./run.sh e07b_dla_split.py qwen3-06b-bf16 gen 400
echo EXPLORE6_DONE
