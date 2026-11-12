#!/usr/bin/env bash
# reruns of the 3-digit Qwen3 scans with the ratio-of-sums metric (exp04_scan.py from 22:26)
cd "$(dirname "$0")"
while systemctl --user is-active -q d4-exp04; do sleep 30; done
for k in intoteen outteen; do ./run.sh exp04_scan.py qwen3-06b --digits 3 --kind $k --n 96; done
echo done > ../outputs/chain_exp04b.done
