#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_acq.log; : > $LOG
echo "########## simulating the two acquisition qualities ##########" >> $LOG
$PY exp/d3acq.py gen   --contrast T2 >> $LOG 2>&1 || { echo "GEN FAILED" >> $LOG; exit 1; }
echo "########## enhancer + QA map on simulated-standard input ##########" >> $LOG
$PY exp/d3acq.py train --contrast T2 --steps 9000 >> $LOG 2>&1 || { echo "TRAIN FAILED" >> $LOG; exit 1; }
echo "########## allocation policies ##########" >> $LOG
$PY exp/d3acq.py run   --contrast T2 >> $LOG 2>&1 || echo "RUN FAILED" >> $LOG
echo "ACQ DONE" >> $LOG
