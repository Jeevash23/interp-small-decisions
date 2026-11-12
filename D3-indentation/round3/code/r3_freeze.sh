#!/usr/bin/env bash
# D3 round 3: freeze PROTOCOL-3.md and its scripts (digests to FROZEN.md, read-only), then create the slot links and
# the read-only copies of PROTOCOL.md and PROTOCOL-2.md that the frozen scripts check. Run once, by hand.
set -euo pipefail
cd "$(dirname "$0")/../.."     # D3-indentation
for f in round3/data/newcode round3/data/r2code round3/data/r2gen2 round3/outputs/conf round3/x1/outputs/conf; do
  if [ -e "$f" ]; then echo "exists before freeze: $f"; exit 1; fi
done
X1=../X1-shared-parts
chmod a-w PROTOCOL-3.md round3/code/r3x_c01.py round3/code/r3x_l11h9.py round3/code/r3_pairs.py round3/code/r3_verdicts.py \
  round3/code/r3q_conf.sh round3/code/r3_freeze.sh
t=$(date "+%Y-%m-%d %H:%M:%S %z")
{
  echo; echo "## Round 3 confirmation protocol (\`PROTOCOL-3.md\`)"; echo
  echo "| File | SHA-256 | Frozen |"; echo "|---|---|---|"
  for f in PROTOCOL-3.md round3/code/{r3x_c01.py,r3x_l11h9.py,r3_pairs.py,r3_verdicts.py,r3q_conf.sh,r3_freeze.sh,e01_behaviour.py,gate_summary.py} \
           round3/code/{common.py,pyindent.py,gemma_load.py,c01_ablate.py,c02_pairs.py,c03_verdicts.py,r2common.py,r2c01_header.py,r2c02_gen2.py,r2c03_group.py,r2c04_verdicts.py,run.sh,r2q_conf.sh} \
           round3/x1/outputs/means_qwen3-06b-bf16_IND_dec.pt $X1/code/c01_confirm.py $X1/code/x1common.py; do
    echo "| $f | $(sha256sum "$f" | cut -d' ' -f1) | $t |"
  done
  echo
  echo "Frozen (read-only) before any model read round-3 material; no \`round3/outputs/conf/\` existed. The copies in \`round3/code/\` other than the r3 scripts and the two gate scripts have the same digests as their earlier freezes above; the means file and X1's two scripts match \`X1-shared-parts/FROZEN.md\`. After this entry the slot links and the copies \`round3/PROTOCOL.md\` and \`round3/PROTOCOL-2.md\` were created (PROTOCOL-3.md, How the frozen scripts are run)."
} >> FROZEN.md
cd round3
mkdir -p data/newcode data/r2code data/r2gen2
ln -s ../code/part-held.jsonl data/newcode/part-conf.jsonl
ln -s ../code/part-held.jsonl data/newcode/part-explore.jsonl
ln -s part-held.jsonl data/gen/part-conf.jsonl
ln -s part-held.jsonl data/gen/part-explore.jsonl
ln -s ../code/part-held.jsonl data/r2code/part-held.jsonl
ln -s ../gen2/part-held.jsonl data/r2gen2/part-held.jsonl
ln -s ../../../data/code/part-held.jsonl x1/data/code/part-held.jsonl
ln -s ../../../data/gen/part-held.jsonl x1/data/gen/part-held.jsonl
cp ../PROTOCOL.md PROTOCOL.md; cp ../PROTOCOL-2.md PROTOCOL-2.md; chmod a-w PROTOCOL.md PROTOCOL-2.md
cmp PROTOCOL.md ../PROTOCOL.md && cmp PROTOCOL-2.md ../PROTOCOL-2.md
for l in data/newcode/part-conf.jsonl data/gen/part-conf.jsonl data/r2code/part-held.jsonl data/r2gen2/part-held.jsonl x1/data/code/part-held.jsonl x1/data/gen/part-held.jsonl; do
  echo "$l -> $(readlink -f $l | sed 's|.*/round3/||') $(sha256sum < $l | cut -c1-12)"
done
echo "frozen $t"
