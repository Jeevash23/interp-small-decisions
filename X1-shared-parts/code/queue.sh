#!/usr/bin/env bash
# Repeat one resumable X1 job in lock holds of at most ~18 minutes until its last line reports done.
# Usage: queue.sh <logname> <script.py> [args...]   (each hold releases the GPU lock; 20 s gap between holds)
cd "$(dirname "$0")"
name="$1"; shift
for i in $(seq 1 30); do
  LOGNAME_X1="$name" ./run.sh "$@" --time-limit 1080
  if tail -3 "../outputs/logs/${name}.log" | grep -q '"done": true'; then echo "queue done $(date +%H:%M:%S)" >> "../outputs/logs/${name}.log"; exit 0; fi
  if tail -1 "../outputs/logs/${name}.log" | grep -qv '^exit 0'; then echo "queue stopped on error" >> "../outputs/logs/${name}.log"; exit 1; fi
  sleep 20
done
