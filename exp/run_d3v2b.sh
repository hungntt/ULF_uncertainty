#!/bin/bash
# Second pass at the posterior. The first collapsed: after 14k steps the dispersion
# fell to 0.0010 against a 0.078 error, i.e. the network learned a near-deterministic
# map and the diffusion noise no longer changes the output. Three changes:
#   norm       diffuse in unit-variance subband space (LL..HH std differ ~15x here,
#              so one gamma was meaningless across bands)
#   gamma 2.0  the value the paper uses on its normalised data
#   cond_drop  zero the 64 mT conditioning on 10% of steps, so the network cannot
#              rely on the conditioning alone
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_d3v2b.log; : > $LOG
until [ -f out/d3v2_diff_T1.pt ] && [ ! -f out/.d3v2_analyze_running ]; do sleep 60; done
echo "########## v2 posterior: norm + gamma 2.0 + cond-drop, T2 ##########" >> $LOG
$PY exp/d3v2.py --contrast T2 --steps 14000 --gamma 2.0 --norm --cond_drop 0.1 --tag _v2 >> $LOG 2>&1
echo "V2 TRAINING DONE" >> $LOG
$PY exp/d3v2_analyze.py --contrast T2 --K 32 --J 8 --tag _v2 >> $LOG 2>&1
echo "V2 ANALYSIS DONE" >> $LOG
