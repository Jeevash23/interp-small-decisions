#!/usr/bin/env bash
# D3 round 3: resume r2c03 (OLMo-2 DED-O group ablation) after r2q_conf.sh's pass cap (12) was reached at 11:40 on
# 11 Nov with 14 of 22 runs saved. Same command, same script, resumes from its saved record; no change to the computation.
cd "$(dirname "$0")"
log="../outputs/r2c03_group_olmo1b_DEDO=L13H1,L10H12,L11H0,L15H4_timelimit_1000.log"
for i in $(seq 1 20); do
  MEM=7G ./run.sh r2c03_group.py olmo1b DEDO=L13H1,L10H12,L11H0,L15H4 --time-limit 1000
  grep -q "time limit" "$log" || { echo "r2c03 done $(date +%H:%M)"; exit 0; }
done
echo "r2c03 still unfinished"
