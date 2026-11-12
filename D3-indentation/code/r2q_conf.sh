#!/usr/bin/env bash
# D3 round 2 confirmation (PROTOCOL-2.md). Each run.sh call is one GPU lock hold with at most 1,000 s of work;
# a call is repeated while its script stops at the time limit (the scripts resume from their saved records).
cd "$(dirname "$0")"
rep() {  # rep <log name> <max passes> <args...>
  local log="../outputs/$1.log" n=$2; shift 2
  for i in $(seq 1 "$n"); do
    MEM=7G ./run.sh "$@"
    grep -q "time limit" "$log" || return 0
  done
  echo "still unfinished after $n passes: $*" >&2; return 1
}
rep r2c01_header_qwen306bbf16_L23H2_timelimit_1000 8 r2c01_header.py qwen3-06b-bf16 L23H2 --time-limit 1000
rep r2c02_gen2_qwen306bbf16_L23H2_timelimit_1000 6 r2c02_gen2.py qwen3-06b-bf16 L23H2 --time-limit 1000
rep r2c01_header_olmo1b_L13H1_timelimit_1000 10 r2c01_header.py olmo1b L13H1 --time-limit 1000
rep r2c02_gen2_olmo1b_L13H1_timelimit_1000 6 r2c02_gen2.py olmo1b L13H1 --time-limit 1000
rep r2c03_group_olmo1b_DEDO=L13H1,L10H12,L11H0,L15H4_timelimit_1000 12 r2c03_group.py olmo1b DEDO=L13H1,L10H12,L11H0,L15H4 --time-limit 1000
echo "all confirmation runs finished $(date +%H:%M)"
