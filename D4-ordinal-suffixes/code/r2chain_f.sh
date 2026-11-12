#!/usr/bin/env bash
# Re-run E06 for the four checkpoints lost to the OOM at 21:40 on 8 Nov (model not freed between revisions;
# fixed 9 Nov 09:2x). One revision per job. Exploratory and descriptive only (AMENDMENTS-2.md, Amendment 1).
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
for rev in stage1-step10000-tokens21B stage1-step40000-tokens84B stage1-step480000-tokens1007B stage1-step1907359-tokens4001B; do
  ./run.sh -m 8 r2e06_olmo_ckpts.py "$rev"
done
echo "chain_f done $(date)"
