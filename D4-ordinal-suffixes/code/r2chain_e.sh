#!/usr/bin/env bash
# Re-queue E04 for Gemma-4-E2B after the crash at 21:17 (per-layer head size; fixed 21:19). Runs after chain D.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
while pgrep -f "r2conf_run.py .* --dry" >/dev/null; do sleep 20; done
mv ../outputs/r2e04_other_models_gemma4-e2b.log ../outputs/r2e04_other_models_gemma4-e2b_CRASHED.log 2>/dev/null
./run.sh -m 8 r2e04_other_models.py gemma4-e2b
