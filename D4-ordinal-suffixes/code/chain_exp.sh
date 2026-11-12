#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh exp01_patch.py qwen3-06b --n 160 --digits 2,3,4
./run.sh exp02_heads.py qwen3-06b --digits 3 --n 96
./run.sh exp02_heads.py qwen3-06b --digits 2 --n 96
./run.sh exp02_heads.py olmo2-1b --digits 4 --n 96
./run.sh exp02_heads.py qwen3-06b --digits 4 --n 96
./run.sh exp01_patch.py olmo2-1b --n 160 --digits 3,4
echo done > ../outputs/chain_exp.done
