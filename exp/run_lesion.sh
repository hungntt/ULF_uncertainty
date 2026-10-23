#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_lesion.log; : > $LOG
$PY exp/d3lesion.py --contrast T2 --n_slices 6 >> $LOG 2>&1
echo "LESION DONE" >> $LOG
