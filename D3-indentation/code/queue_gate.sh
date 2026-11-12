#!/usr/bin/env bash
# D3 gate queue (rerun after the 22:30 measure fix): natural code and generated programs for the four charter models.
# Each step holds the GPU lock for at most ~18 minutes (--time-limit 1000 plus load); e01 resumes where it stopped.
cd "$(dirname "$0")"
for job in "pythia-410m newcode" "qwen3-06b newcode" "gemma4-e2b newcode" "pythia-410m gen" "qwen3-06b gen" "gemma4-e2b gen" "gpt2 newcode" "gpt2 gen"; do
  set -- $job
  mem=6G; [ "$1" = gemma4-e2b ] && mem=8G
  for i in 1 2 3 4 5 6 7 8; do
    MEM=$mem ./run.sh e01_behaviour.py $1 $2 --time-limit 1000
    log="../outputs/e01_behaviour_$(echo "$1_$2___timelimit_1000" | tr -d '.-' | sed 's/___/_/').log"
    cp "$log" "${log%.log}_pass$i.log"
    grep -q '"points"' "$log" && ! grep -q "time limit" "$log" && break
    grep -q "Traceback" "$log" && break
  done
done
echo QUEUE_DONE
