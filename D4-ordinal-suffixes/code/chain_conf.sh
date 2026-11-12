#!/usr/bin/env bash
# PROTOCOL.md section 5: the confirmation runs, once each. Refuses to start unless PROTOCOL.md is read-only.
cd "$(dirname "$0")"
[ -w ../PROTOCOL.md ] && { echo "PROTOCOL.md is not frozen"; exit 1; }
[ -e ../outputs/conf/chain_conf.started ] && { echo "confirmation already started once; see PROTOCOL.md section 5"; exit 1; }
mkdir -p ../outputs/conf; date '+%F %T' > ../outputs/conf/chain_conf.started
for m in olmo2-1b gpt2 pythia-1b; do ./run.sh conf01_boundary.py $m --part conf --per 200; done
./run.sh exp05_ablate.py qwen3-06b --heads 0.2,1.6 --digits 2,3,4 --per 96 --random 20 --pool 0-2 --part conf
for d in 3 4; do for k in intoteen last; do
  ./run.sh exp04_scan.py qwen3-06b --digits $d --kind $k --n 96 --what head --part conf
done; done
./run.sh exp05_ablate.py olmo2-1b --heads 0.5,0.0,0.9 --digits 3,4,5,7 --per 96 --random 20 --pool 0-2 --part conf
date '+%F %T' > ../outputs/conf/chain_conf.done
