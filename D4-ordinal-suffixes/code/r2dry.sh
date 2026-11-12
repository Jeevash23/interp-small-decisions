#!/usr/bin/env bash
# Dry run of the round-2 confirmation pipeline on exploration items (outputs/r2dry/), one lock hold, one cage.
cd "$(dirname "$0")"
V="${PYTHON:-python3}"
while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt 6 ]; do sleep 30; done
exec flock $HOME/.cache/d1-gpu.lock systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=1G -p TasksMax=256 -- timeout 1200 bash -c "
export PYTHONUNBUFFERED=1
'$V' r2conf_run.py qwen --dry && '$V' r2conf_run.py qwen2 --dry && '$V' r2conf_run.py olmo --dry
" > ../outputs/r2dry_run.log 2>&1
