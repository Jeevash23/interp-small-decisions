#!/usr/bin/env bash
# R2 exploration chain A: small jobs, each its own lock hold (run.sh), unbuffered logs.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
./run.sh r2e01_qwen_path.py --digits 2
./run.sh r2e01_qwen_path.py --digits 3
./run.sh r2e03_olmo_downstream.py --digits 3,4,7
./run.sh r2e01_qwen_path.py --digits 4
./run.sh r2e02_qwen_twodigit.py --digits 2
./run.sh r2e03_olmo_downstream.py --digits 5,8,9,10
touch ../outputs/r2/chain_a.done
