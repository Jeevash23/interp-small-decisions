#!/usr/bin/env bash
# D3 exploration queue 2 (Qwen3-0.6B-Base in bfloat16 for scans). Each step <= ~18 min of lock; e02/e04 resume.
cd "$(dirname "$0")"
M=qwen3-06b-bf16
for i in 1 2 3; do ./run.sh e02_ablate.py $M newcode 160 --time-limit 1000; grep -q "^done" ../outputs/e02_ablate_qwen306bbf16_newcode_160_timelimit_1000.log && ! grep -q "time limit" ../outputs/e02_ablate_qwen306bbf16_newcode_160_timelimit_1000.log && break; done
for i in 1 2 3 4; do ./run.sh e04_pairs.py $M 120 --time-limit 1000; grep -q "^secs" ../outputs/e04_pairs_qwen306bbf16_120_timelimit_1000.log && ! grep -q "time limit" ../outputs/e04_pairs_qwen306bbf16_120_timelimit_1000.log && break; done
MEM=7G ./run.sh e05_probe.py qwen3-06b newcode 5000
echo EXPLORE2_DONE
