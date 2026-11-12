#!/usr/bin/env bash
# D5 round 2 confirmation runs (PROTOCOL-2.md, Section 3), each chain one hold of the shared GPU lock
# (code/run2.sh: RAM gate, memory cage, 20-minute hold limit). Run once, after PROTOCOL-2.md is frozen.
cd "$(dirname "$0")"
./run2.sh -m 10 "r2_exp03_sets.py qwen3-17b reserve 0 300 r2_sets_qwen.json 2 13 20 conf" \
                "r2_exp03_sets.py qwen3-17b reserve 0 300 r2_sets_qwen_l17h6.json 2 13 20 l17h6" \
                "r2_exp06_attn.py qwen3-17b reserve 0 200 L19H0,L17H6,L18H1,L18H0,L0H3"
./run2.sh -m 10 "r2_exp05_edit.py qwen3-17b reserve 0 300 4 15" \
                "r2_exp04_other.py qwen3-17b reserve 0 264 0 250" \
                "r2_exp04_other.py qwen3-17b reserve 264 264 250 250"
./run2.sh -m 10 "r2_exp03_sets.py gemma reserve 0 250 r2_sets_gemma.json 2 13 20 conf" \
                "r2_exp06_attn.py gemma reserve 0 200 L24H6,L14H6,L14H7,L14H1,L15H2"
./run2.sh -m 10 "r2_exp05_edit.py gemma reserve 0 200 4 15"
./run2.sh -m 10 "r2_exp04_other.py gemma reserve 0 264 0 250"
./run2.sh -m 10 "r2_exp04_other.py gemma reserve 264 264 250 250"
echo "r2 confirmation chains done $(date +%T)"
