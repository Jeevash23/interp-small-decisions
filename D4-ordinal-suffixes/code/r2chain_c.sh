#!/usr/bin/env bash
# R2 exploration chain C: after chain B; E07 (OLMo-2 signal form), then question 4 (other models).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
while systemctl --user is-active -q d4r2-chain-b; do sleep 20; done
./run.sh r2e07_olmo_format.py
./run.sh r2e08_qwen_backup.py
./run.sh -m 8 r2e04_other_models.py smollm2-17b
./run.sh -m 8 r2e04_other_models.py qwen3-17b --gate
./run.sh -m 8 r2e04_other_models.py gemma4-e2b
touch ../outputs/r2/chain_c.done
