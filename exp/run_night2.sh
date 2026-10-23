#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_night2.log; : > $LOG
echo "########## polarity sweep: is route B transferring, or inverting? ##########" >> $LOG
$PY exp/d3polarity.py >> $LOG 2>&1
echo "POLARITY DONE" >> $LOG
