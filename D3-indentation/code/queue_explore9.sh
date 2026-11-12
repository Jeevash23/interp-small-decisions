#!/usr/bin/env bash
# Two-route check with disjoint groups (DED vs TERMR), exploration parts only.
cd "$(dirname "$0")"
while systemctl --user is-active -q d3-explore8; do sleep 60; done
M=qwen3-06b-bf16
./run.sh e08_pairs_group.py $M 120 DED=L23H2,L17H3,L17H4,L17H5,L18H6 TERMR=L11H9,L5H6,L17H6,L16H2,L6H5
./run.sh e06_group.py $M newcode 200 DED=L23H2,L17H3,L17H4,L17H5,L18H6 TERMR=L11H9,L5H6,L17H6,L16H2,L6H5 --draws 10
./run.sh e06_group.py $M gen 200 TERMR=L11H9,L5H6,L17H6,L16H2,L6H5 --draws 10
echo EXPLORE9_DONE
