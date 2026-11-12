#!/usr/bin/env bash
cd "$(dirname "$0")"
./run.sh r2e03_term.py 300 terminal
./run.sh r2e02_knock.py x1code 150
./run.sh r2e02_knock.py gen2 300
./run.sh r2e03_term.py 150 plain
