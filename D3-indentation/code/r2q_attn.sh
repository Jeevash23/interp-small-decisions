#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh r2e01_attn.py x1code 400
./run.sh r2e01_attn.py gen2 600
./run.sh r2e01_attn.py r1gen 300
