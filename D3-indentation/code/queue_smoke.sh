#!/usr/bin/env bash
# Pipeline smoke tests of the confirmation scripts on EXPLORATION material (outputs/conf_smoke/).
cd "$(dirname "$0")"
while systemctl --user is-active -q d3-explore9; do sleep 60; done
./run.sh c01_ablate.py newcode --smoke
./run.sh c01_ablate.py gen --smoke
./run.sh c02_pairs.py --smoke
echo SMOKE_DONE
