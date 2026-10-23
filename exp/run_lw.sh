#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_lw.log; : > $LOG
echo "########## polarity augmentation WITH a lesion-weighted loss ##########" >> $LOG
$PY exp/d3aug.py --aug 0.5 --lw 1.0 --tag _augw >> $LOG 2>&1
echo "########## sweep ##########" >> $LOG
$PY exp/d3polarity.py --cmodel bothtoT1_augw --label _augw >> $LOG 2>&1
echo "LW DONE" >> $LOG
