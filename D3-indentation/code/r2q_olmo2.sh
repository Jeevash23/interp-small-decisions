#!/usr/bin/env bash
cd "$(dirname "$0")"
MEM=7G ./run.sh r2e05_steer.py x1code 200 olmo1b L13H1
MEM=7G ./run.sh r2e05_steer.py gen2 300 olmo1b L13H1
