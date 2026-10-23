#!/bin/bash
# Wait for night1's end marker, then run night3. Polls a file - never pgrep, which
# matches the shell wrapper carrying the script's own text and deadlocks.
cd /mnt/G/Projects/ULF-MICCAI
for i in $(seq 1 480); do
  grep -q 'NIGHT1 DONE' out/run_night1.log 2>/dev/null && break
  sleep 30
done
bash exp/run_night3.sh
