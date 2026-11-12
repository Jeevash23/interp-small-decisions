#!/usr/bin/env bash
# D3 round 3 confirmation queue (PROTOCOL-3.md, Run plan). Each run.sh call is one GPU lock hold of at most about
# 1,000 s of work inside the memory cage; a call is repeated while its script stops at the time limit (the scripts
# resume from their saved records). Order: E (L11H9), X (X1 indentation side), D3 round 1, D3 round 2, gate.
cd "$(dirname "$0")"
rep() {  # rep <max passes> <script.py> [args...]   (log name as run.sh makes it)
  local n=$1; shift
  local log="../outputs/$(basename "$1" .py)_$(echo "${@:2}" | tr ' /' '__' | tr -d '.-').log"
  for i in $(seq 1 "$n"); do
    MEM=7G ./run.sh "$@"
    tail -1 "$log" | grep -q '^exit 0' || { echo "error in $*: $(tail -3 "$log")"; return 1; }
    grep -q "time limit" "$log" || { echo "done $* $(date +%H:%M)"; return 0; }
  done
  echo "still unfinished after $n passes: $*" >&2; return 1
}
echo "start $(date '+%F %H:%M')"
rep 8 r3x_l11h9.py --time-limit 1000
rep 4 r3_pairs.py --time-limit 1000
rep 10 r3x_c01.py qwen3-06b-bf16 IND --time-limit 1000
rep 10 c01_ablate.py newcode --time-limit 1000
rep 6 c01_ablate.py gen --time-limit 1000
rep 4 c02_pairs.py --time-limit 1000
./r2q_conf.sh
for m in qwen3-06b olmo1b pythia-410m gemma4-e2b gpt2; do
  for s in newcode gen; do rep 8 e01_behaviour.py $m $s --time-limit 1000; done
done
echo "all round-3 runs finished $(date '+%F %H:%M')"
