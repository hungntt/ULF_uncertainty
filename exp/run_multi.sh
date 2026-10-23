#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_multi.log; : > $LOG
echo "########## map trained ACROSS three enhancers ##########" >> $LOG
$PY exp/d3multi.py --steps 10000 --tag _multi >> $LOG 2>&1
echo "########## scored on all six, three of them held out ##########" >> $LOG
$PY exp/d3xarch.py --map_tag _multi --held "U-Net L1+SSIM|U-Net absolute|D1 calibrated corpus" >> $LOG 2>&1
echo "MULTI DONE" >> $LOG
