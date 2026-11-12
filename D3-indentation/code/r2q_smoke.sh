#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh r2c01_header.py qwen3-06b-bf16 L23H2 --smoke
./run.sh r2c02_gen2.py qwen3-06b-bf16 L23H2 --smoke
