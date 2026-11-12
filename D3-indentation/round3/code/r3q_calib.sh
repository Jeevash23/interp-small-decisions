#!/usr/bin/env bash
# D3 round 3 calibration (CHARTER-3.md, Calibration): old material only, exploratory. Runs under the GPU lock.
cd "$(dirname "$0")"
MEM=7G ./run.sh r3x_l11h9.py --calib
MEM=7G ./run.sh r3_pairs.py --calib
echo "calibration finished $(date +%H:%M)"
