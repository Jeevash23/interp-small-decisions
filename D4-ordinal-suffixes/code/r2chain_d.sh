#!/usr/bin/env bash
# R2 chain D: after chain C; dry run of the confirmation pipeline, then question 5 (OLMo-2 checkpoints on disk).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
while systemctl --user is-active -q d4r2-chain-c3 || systemctl --user is-active -q d4r2-chain-c2b; do sleep 20; done
rm -rf ../outputs/r2dry; ./r2dry.sh
./run.sh -m 8 r2e06_olmo_ckpts.py stage1-step300-tokens1B stage1-step10000-tokens21B stage1-step40000-tokens84B
./run.sh -m 8 r2e06_olmo_ckpts.py stage1-step200000-tokens420B stage1-step480000-tokens1007B stage1-step1907359-tokens4001B
touch ../outputs/r2/chain_d.done
