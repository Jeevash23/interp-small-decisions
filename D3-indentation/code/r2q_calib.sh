#!/usr/bin/env bash
# 9 Nov: calibration runs on exploration material (exploratory), to set PROTOCOL-2 thresholds before freezing.
cd "$(dirname "$0")"
MEM=7G ./run.sh r2c01_header.py qwen3-06b-bf16 L23H2 --calib x1code 250 --time-limit 1000
MEM=7G ./run.sh r2c02_gen2.py qwen3-06b-bf16 L23H2 --calib 400 --time-limit 1000
MEM=7G ./run.sh r2c01_header.py olmo1b L13H1 --calib x1code 250 --time-limit 1000
MEM=7G ./run.sh r2c02_gen2.py olmo1b L13H1 --calib 400 --time-limit 1000
MEM=7G ./run.sh r2c03_group.py olmo1b DEDO=L13H1,L10H12,L11H0,L15H4 --calib 300 --time-limit 1000
