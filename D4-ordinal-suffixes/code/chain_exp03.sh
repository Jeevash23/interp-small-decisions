#!/usr/bin/env bash
cd "$(dirname "$0")"
for m in olmo2-1b pythia-1b gpt2 qwen3-06b smollm2-17b; do ./run.sh exp03_length_sep.py $m; done
./run.sh -m 8 exp03_length_sep.py gemma4-e2b
echo done > ../outputs/chain_exp03.done
