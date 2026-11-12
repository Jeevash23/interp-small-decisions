#!/usr/bin/env bash
# Run one D4 model job under the shared GPU lock and inside the memory cage (DECISIONS.md, 7 Nov 21:00):
#   ./run.sh [-m <GB>] <script.py> [args...]   -> ../outputs/<script>_<args>.log
# Waits until at least 6 GB of RAM is available, then takes ~/.cache/d1-gpu.lock (blocking).
# Each job is capped at 20 minutes so the three studies take turns.
cd "$(dirname "$0")"
MEM=6
if [ "$1" = "-m" ]; then MEM=$2; shift 2; fi
V="${PYTHON:-python3}"
name="$(basename "$1" .py)_$(echo "${@:2}" | sed 's/--//g; s/ /_/g')"
while [ "$(free -g | awk '/^Mem:/ {print $7}')" -lt 6 ]; do sleep 30; done
exec flock $HOME/.cache/d1-gpu.lock systemd-run --user --scope -q -p MemoryMax=${MEM}G -p MemorySwapMax=1G -p TasksMax=256 -- timeout 1200 "$V" "$@" > "../outputs/${name}.log" 2>&1
