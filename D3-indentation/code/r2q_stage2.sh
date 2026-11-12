#!/usr/bin/env bash
# 9 Nov: question 5 follow-up. Download two stage-2 checkpoints (ingredient 1 at 3B and 51B tokens), then scan them.
cd "$(dirname "$0")"
V="${PYTHON:-python3}"
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" r2_dl_olmo_ckpts.py stage2-ingredient1-step1000-tokens3B stage2-ingredient1-step23852-tokens51B > ../outputs/r2_dl_stage2.log 2>&1
MEM=7G ./run.sh r2e09_stage2.py 200 olmo1b-s2i1-3B
MEM=7G ./run.sh r2e09_stage2.py 200 olmo1b-s2i1-51B
