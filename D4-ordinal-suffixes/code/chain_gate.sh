#!/usr/bin/env bash
cd "$(dirname "$0")"
for m in qwen3-06b pythia-410m olmo2-1b smollm2-17b gpt2; do ./run.sh gate01_behaviour.py $m; done
./run.sh -m 8 gate01_behaviour.py gemma4-e2b
echo chain done > ../outputs/chain_gate.done
