#!/usr/bin/env bash
# pipeline check of conf01 on exploration material (not a result of the protocol)
cd "$(dirname "$0")"
for m in gpt2 pythia-1b olmo2-1b; do ./run.sh conf01_boundary.py $m --part explore; done
echo done > ../outputs/chain_conf01x.done
