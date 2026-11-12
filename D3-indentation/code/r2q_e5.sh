#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh r2e05_steer.py x1code 200
./run.sh r2e05_steer.py gen2 300
./run.sh r2e05_steer.py r1code 200
