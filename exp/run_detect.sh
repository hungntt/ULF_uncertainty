#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_detect.log; : > $LOG
echo "########## detectability map: labels + training (T2) ##########" >> $LOG
$PY exp/d3detect.py --contrast T2 --steps 8000 >> $LOG 2>&1
echo "########## evaluation against strong baselines ##########" >> $LOG
$PY exp/d3detect_eval.py --contrast T2 >> $LOG 2>&1
echo "DETECT DONE" >> $LOG
