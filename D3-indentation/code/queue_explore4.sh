#!/usr/bin/env bash
# D3 exploration queue 4: group patching on the return/assignment pairs, after e04 (queue 2) finishes.
cd "$(dirname "$0")"
while systemctl --user is-active -q d3-explore2; do sleep 60; done
./run.sh e08_pairs_group.py qwen3-06b-bf16 120 TOP6=L17H4,L11H9,L5H6,L17H6,L16H2,L6H5 TOP12=L17H4,L11H9,L5H6,L17H6,L16H2,L6H5,L3H3,L16H3,L8H9,L15H3,L5H8,L11H14 EARLY=L3H3,L5H6,L6H5,L3H2,L1H0 OPEN=L19H3,L20H14,L22H9,L22H1,L20H15,L21H9
echo EXPLORE4_DONE
