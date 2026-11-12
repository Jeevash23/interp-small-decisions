#!/usr/bin/env bash
# D3 round 2, question 4: exploration in OLMo-2-1B (passed the frozen gate on natural code, 17:16).
cd "$(dirname "$0")"
while systemctl --user is-active -q d3r2-gate; do sleep 20; done
MEM=7G ./run.sh r2e01_attn.py x1code 300 olmo1b
for i in 1 2 3 4 5 6; do
  MEM=7G ./run.sh e02_ablate.py olmo1b newcode 160 --time-limit 1000
  log=../outputs/e02_ablate_olmo1b_newcode_160_timelimit_1000.log
  cp "$log" "${log%.log}_pass$i.log"
  grep -q "time limit" "$log" || break
done
echo QUEUE_DONE
