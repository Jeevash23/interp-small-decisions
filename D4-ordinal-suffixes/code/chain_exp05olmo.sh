#!/usr/bin/env bash
# exploratory: OLMo-2 layer-0 heads that carry the previous token (exp04 4-digit scans) - mean ablation at the last position
cd "$(dirname "$0")"
while [ ! -f ../outputs/chain_exp04b.done ]; do sleep 30; done
./run.sh exp05_ablate.py olmo2-1b --heads 0.5,0.0,0.9 --digits 3,4,5,7 --per 96 --random 20 --pool 0-2
echo done > ../outputs/chain_exp05olmo.done
