#!/usr/bin/env bash
# Re-queue E04 for SmolLM2 after the crash at 21:03 (M.last unset before the attention pass; fixed 21:05).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
while systemctl --user is-active -q d4r2-chain-c3; do sleep 20; done
mv ../outputs/r2e04_other_models_smollm2-17b.log ../outputs/r2e04_other_models_smollm2-17b_CRASHED.log 2>/dev/null
./run.sh -m 8 r2e04_other_models.py smollm2-17b
