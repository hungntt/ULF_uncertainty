#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_d3v2c.log; : > $LOG
until [ -f out/d3v2_T2_v2.json ]; do sleep 60; done     # wait for the v2 analysis
echo "########## subject-swap counterfactual, collapsed posterior ##########" >> $LOG
$PY exp/d3v2_swap.py --contrast T2 --K 16 >> $LOG 2>&1
echo "########## subject-swap counterfactual, v2 posterior ##########" >> $LOG
$PY exp/d3v2_swap.py --contrast T2 --tag _v2 --K 16 >> $LOG 2>&1
echo "SWAP DONE" >> $LOG
