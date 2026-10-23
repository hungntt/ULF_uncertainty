#!/bin/bash
cd /mnt/G/Projects/ULF-MICCAI
for i in $(seq 1 960); do
  grep -q 'NIGHT4 SWEEP DONE' out/run_night4.log 2>/dev/null && break
  sleep 30
done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "########## one fixed map across enhancers ##########" > out/run_night5.log
/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u exp/d3xarch.py >> out/run_night5.log 2>&1
echo "NIGHT5 DONE" >> out/run_night5.log
