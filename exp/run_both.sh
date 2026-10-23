#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_both.log; : > $LOG
echo "########## 1. which input route preserves lesions in 3T T1? ##########" >> $LOG
$PY exp/d3pipe.py >> $LOG 2>&1
echo "########## 2. detectability trained on BOTH polarities (T2) ##########" >> $LOG
$PY exp/d3detect.py --contrast T2 --steps 8000 --tag _signed >> $LOG 2>&1
$PY exp/d3detect_eval.py --contrast T2 --tag _signed >> $LOG 2>&1
echo "BOTH DONE" >> $LOG
