#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh e09_falsekw.py qwen3-06b-bf16 250 READ=L1H0,L3H3,L3H2,L5H6,L6H5,L17H4
./run.sh e09_falsekw.py pythia-410m 250
MEM=8G ./run.sh e09_falsekw.py gemma4-e2b 250
echo EXPLORE5_DONE
