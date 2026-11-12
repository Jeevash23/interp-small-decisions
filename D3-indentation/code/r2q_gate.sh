#!/usr/bin/env bash
# D3 round 2, question 4: round 1's frozen gate (e01_behaviour.py) for the second-model candidates, in charter order.
cd "$(dirname "$0")"
for job in "olmo1b newcode" "olmo1b gen" ; do
  set -- $job
  for i in 1 2 3 4 5 6 7 8; do
    MEM=7G ./run.sh e01_behaviour.py $1 $2 --time-limit 1000
    log="../outputs/e01_behaviour_$(echo "$1_$2___timelimit_1000" | tr -d '.-' | sed 's/___/_/').log"
    cp "$log" "${log%.log}_pass$i.log"
    grep -q '"points"' "$log" && ! grep -q "time limit" "$log" && break
    grep -q "Traceback" "$log" && break
  done
done
echo QUEUE_DONE
