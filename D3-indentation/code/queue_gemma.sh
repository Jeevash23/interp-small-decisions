#!/usr/bin/env bash
cd "$(dirname "$0")"
for i in 1 2 3 4 5 6 7 8; do
  MEM=8G ./run.sh e01_behaviour.py gemma4-e2b newcode --time-limit 1000
  log=../outputs/e01_behaviour_gemma4e2b_newcode_timelimit_1000.log; cp $log ${log%.log}_pass$i.log
  grep -q '"points"' $log && ! grep -q "time limit" $log && break
  grep -q Traceback $log && break
done
echo GEMMA_DONE
