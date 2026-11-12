#!/usr/bin/env bash
cd "$(dirname "$0")"
for d in 3 4 2; do for k in intoteen outteen last; do ./run.sh exp04_scan.py qwen3-06b --digits $d --kind $k --n 96; done; done
for k in intoteen outteen; do ./run.sh exp04_scan.py olmo2-1b --digits 4 --kind $k --n 96; ./run.sh exp04_scan.py olmo2-1b --digits 3 --kind $k --n 96; done
echo done > ../outputs/chain_exp04.done
