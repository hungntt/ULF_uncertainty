#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_d3v2.log; : > $LOG
echo "########## residual-shift wavelet diffusion, T2 ##########" >> $LOG
$PY exp/d3v2.py --contrast T2 --steps 14000 --gamma 1.0 >> $LOG 2>&1
echo "########## residual-shift wavelet diffusion, T1 ##########" >> $LOG
$PY exp/d3v2.py --contrast T1 --steps 14000 --gamma 1.0 >> $LOG 2>&1
echo "D3V2 TRAINING DONE" >> $LOG
