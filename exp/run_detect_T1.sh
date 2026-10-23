#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_detect_T1.log; : > $LOG
echo "########## T1 detectability: labels + training ##########" >> $LOG
$PY exp/d3detect.py --contrast T1 --steps 8000 >> $LOG 2>&1
echo "########## T1 evaluation vs strong baselines ##########" >> $LOG
$PY exp/d3detect_eval.py --contrast T1 >> $LOG 2>&1
echo "DETECT T1 DONE" >> $LOG
