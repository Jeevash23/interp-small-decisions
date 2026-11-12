#!/usr/bin/env bash
# R2 exploration chain B: after chain A; E05 hops, Qwen3 three-digit two-route comparison.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
while systemctl --user is-active -q d4r2-chain-a2; do sleep 20; done
./run.sh r2e05_qwen_hops.py --digits 2,3
./run.sh r2e05_qwen_hops.py --digits 4
touch ../outputs/r2/chain_b.done
