#!/usr/bin/env bash
# D5 round 2: run several scripts in ONE hold of the shared GPU lock (local machine rules: at least 6 GB RAM
# available, memory cage, whole hold limited to 20 minutes). Each argument is one quoted command line
# "script.py args...". Each run's log goes to ../outputs/<script>_<args>.log as in run.sh.
# Usage: ./run2.sh [-m GB] "a.py x y" "b.py z" ...
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MEM=8
if [ "$1" = "-m" ]; then MEM="$2"; shift 2; fi
V="${PYTHON:-python3}"
exec 8>$HOME/.cache/d5-queue.lock
flock 8
flock $HOME/.cache/d1-gpu.lock bash -c '
  while [ "$(free -g | awk "/^Mem:/{print \$7}")" -lt 6 ]; do echo "waiting for RAM $(date +%T)" >&2; sleep 30; done
  V="$1"; MEM="$2"; shift 2; end=$(( $(date +%s) + 1200 ))
  for c in "$@"; do
    left=$(( end - $(date +%s) )); [ $left -lt 60 ] && { echo "hold limit reached before: $c" >&2; break; }
    set -- $c; name="$(basename "$1" .py)_$(echo "${@:2}" | tr " /" "__" | tr -d ".-")"
    systemd-run --user --scope -q -p MemoryMax=${MEM}G -p MemorySwapMax=1G -p TasksMax=256 -- timeout $left "$V" "$@" > "../outputs/${name}.log" 2>&1
    echo "exit $? $(date +%T)" >> "../outputs/${name}.log"
  done' _ "$V" "$MEM" "$@"
echo "chain done $(date +%T)"
