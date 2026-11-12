#!/usr/bin/env bash
# Smoke run of r2conf_run.py "other" on exploration items (--dry), small sizes, before the PROTOCOL-2 freeze.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 R2_PER=8 R2_NR=2
for m in qwen3-17b smollm2-17b gemma4-e2b; do ./run.sh -m 8 r2conf_run.py other "$m" --dry; done
echo "smoke_other done $(date)"
