#!/usr/bin/env bash
# 9 Nov: OLMo exploration finish-up. Each run.sh call is one lock hold (each well under 20 minutes).
cd "$(dirname "$0")"
MEM=7G ./run.sh e06_group.py olmo1b newcode 160 DEDO=L13H1,L10H12,L11H0,L15H4 --draws 10
MEM=7G ./run.sh r2c01_header.py olmo1b L13H1 --smoke
MEM=7G ./run.sh r2c02_gen2.py olmo1b L13H1 --smoke
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b-1B
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b-21B
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b-84B
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b-420B
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b-1007B
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b-4001B
MEM=7G ./run.sh r2e08_ckpt.py 200 olmo1b
