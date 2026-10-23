#!/bin/bash
cd /mnt/G/Projects/ULF-MICCAI
for i in $(seq 1 180); do
  grep -q 'SIMCTL DONE' out/run_simctl.log 2>/dev/null && break
  sleep 30
done
echo "---- three-map eval on the real-input pipeline ----" >> out/run_simctl.log
REAL_TAG=_loose /mnt/G/Projects/FlowPoC/.venv/bin/python -I -u exp/d3realraw.py >> out/run_simctl.log 2>&1
echo "CTLEVAL DONE" >> out/run_simctl.log
