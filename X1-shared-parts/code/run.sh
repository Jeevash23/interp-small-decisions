#!/usr/bin/env bash
# Run one X1 script that loads model weights: waits for >= 6 GB available RAM, takes the shared GPU lock,
# and runs inside a memory cage. Usage: MEM=7G ./run.sh <script.py> [args...]   Log: ../outputs/logs/<name>.log
cd "$(dirname "$0")"
V="${PYTHON:-python3}"
mkdir -p ../outputs/logs
name="${LOGNAME_X1:-$(basename "$1" .py)_$(echo "${@:2}" | tr ' /' '__' | tr -d '.-')}"
MEM="${MEM:-6G}"
while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 6 ]; do sleep 30; done
exec 9>$HOME/.cache/d1-gpu.lock
flock 9
while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 6 ]; do sleep 30; done
echo "start $(date +%H:%M:%S)" >> "../outputs/logs/${name}.log"
systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=1G -p TasksMax=256 -- "$V" "$@" >> "../outputs/logs/${name}.log" 2>&1
echo "exit $? $(date +%H:%M:%S)" >> "../outputs/logs/${name}.log"
