#!/usr/bin/env bash
# Run one D5 script under the shared-local machine rules (DECISIONS.md, 7 Nov 21:00): wait for at least 6 GB of
# RAM available, hold ~/.cache/d1-gpu.lock, run inside a memory cage. Log in ../outputs/<script>_<args>.log.
# The run itself is limited to 20 minutes (timeout inside the cage; waiting for the lock is not counted).
# Usage: ./run.sh [-m <GB>] <script.py> [args...]
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MEM=7
if [ "$1" = "-m" ]; then MEM="$2"; shift 2; fi
V="${PYTHON:-python3}"
name="$(basename "$1" .py)_$(echo "${@:2}" | tr ' /' '__' | tr -d '.-')"
log="../outputs/${name}.log"
mkdir -p ../outputs
exec 8>$HOME/.cache/d5-queue.lock
flock 8
flock $HOME/.cache/d1-gpu.lock bash -c '
  while [ "$(free -g | awk "/^Mem:/{print \$7}")" -lt 6 ]; do echo "waiting for RAM $(date +%T)" >&2; sleep 30; done
  exec systemd-run --user --scope -q -p MemoryMax='"$MEM"'G -p MemorySwapMax=1G -p TasksMax=256 -- timeout 1200 "$0" "$@"' "$V" "$@" > "$log" 2>&1
echo "exit $? $(date +%T)" >> "$log"
