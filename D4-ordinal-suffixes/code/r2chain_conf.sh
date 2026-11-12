#!/usr/bin/env bash
# Round-2 confirmation (PROTOCOL-2.md): each task once, in order, each its own lock hold (run.sh: RAM gate, shared
# GPU lock, memory cage, 20-minute cap). Refuses to start unless PROTOCOL-2.md is read-only, or if it has started before.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
[ -w ../PROTOCOL-2.md ] && { echo "PROTOCOL-2.md is not frozen"; exit 1; }
[ -e ../outputs/r2conf/chain_conf.started ] && { echo "already started once"; exit 1; }
mkdir -p ../outputs/r2conf; date "+%F %T %z" > ../outputs/r2conf/chain_conf.started
./run.sh r2conf_run.py qwen
./run.sh r2conf_run.py qwen2
./run.sh -m 8 r2conf_run.py olmo
for m in $(python3 -c "import json;print(' '.join(json.load(open('r2conf_sets.json'))))"); do
  ./run.sh -m 8 r2conf_run.py other "$m"
done
date "+%F %T %z" > ../outputs/r2conf/chain_conf.done
