#!/usr/bin/env bash
# D3 confirmation run (PROTOCOL.md, frozen 06:31): each step resumes until done; lock holds <= ~18 min.
cd "$(dirname "$0")"
for src in newcode gen; do
  for i in $(seq 1 10); do
    ./run.sh c01_ablate.py $src --time-limit 1000
    log=../outputs/c01_ablate_${src}_timelimit_1000.log; cp $log ${log%.log}_pass$i.log
    grep -q '"done": true' $log && break
    grep -q Traceback $log && break
  done
done
for i in $(seq 1 6); do
  ./run.sh c02_pairs.py --time-limit 1000
  log=../outputs/c02_pairs_timelimit_1000.log; cp $log ${log%.log}_pass$i.log
  grep -q '"runs": 42' $log && break
  grep -q Traceback $log && break
done
echo CONF_DONE
