#!/usr/bin/env bash
# D3 exploration queue 1 (Qwen3-0.6B-Base, main model from 23:05). Each step <= ~18 min of lock; e02/e04 resume.
cd "$(dirname "$0")"
M=qwen3-06b
./run.sh e03_attn.py $M newcode 300
./run.sh e03_attn.py $M gen 300
for i in 1 2 3; do ./run.sh e02_ablate.py $M newcode 160 --time-limit 1000; done
for i in 1 2 3 4; do ./run.sh e04_pairs.py $M 120 --time-limit 1000; done
MEM=7G ./run.sh e05_probe.py $M newcode 5000
echo EXPLORE1_DONE
