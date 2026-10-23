#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_ood.log; : > $LOG
echo "########## out-of-distribution lesion families ##########" >> $LOG
$PY exp/d3ood.py --contrast T2 >> $LOG 2>&1
echo "########## brain-fraction usability number ##########" >> $LOG
$PY exp/d3brainfrac.py T2 >> $LOG 2>&1
echo "OOD DONE" >> $LOG
