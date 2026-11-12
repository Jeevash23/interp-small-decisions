#!/usr/bin/env bash
cd "$(dirname "$0")"
MEM=7G ./run.sh r2e07_hdrscan.py x1code 400 olmo1b
./run.sh r2e07_hdrscan.py x1code 400 qwen3-06b-bf16
