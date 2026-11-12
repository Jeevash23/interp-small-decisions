#!/usr/bin/env bash
# 9 Nov 12:10: question 5 follow-up 2. Ingredient 1 keeps L15H9 to the end of stage 2, yet the released model uses L13H1.
# Download the final checkpoints of ingredients 2 and 3, scan them, and test whether the released weights are their mean.
cd "$(dirname "$0")"
V="${PYTHON:-python3}"
systemd-run --user --scope -q -p MemoryMax=3G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" r2_dl_olmo_ckpts.py stage2-ingredient2-step23852-tokens51B stage2-ingredient3-step23852-tokens51B > ../outputs/r2_dl_stage2b.log 2>&1
systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=1G -p TasksMax=256 -- "$V" r2e10_merge.py > ../outputs/r2e10_merge.log 2>&1
MEM=7G ./run.sh r2e09_stage2.py 200 olmo1b-s2i2-51B
MEM=7G ./run.sh r2e09_stage2.py 200 olmo1b-s2i3-51B
