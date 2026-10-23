#!/bin/bash
# Resolution-targeted re-acquisition: does targeting k-space coverage pay where
# targeting extra averages did not (0.7% ceiling)?
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_acq_res.log; : > $LOG
while pgrep -f "exp/d3cascade\.py" > /dev/null; do sleep 30; done
echo "########## simulating two resolution levels (k-space 0.50 vs 0.75) ##########" >> $LOG
$PY exp/d3acq.py gen_res --contrast T2 >> $LOG 2>&1 || { echo "GEN FAILED" >> $LOG; exit 1; }
echo "########## enhancer + QA map on the lower-resolution input ##########" >> $LOG
$PY exp/d3acq.py train --contrast T2 --steps 9000 --kind rstd >> $LOG 2>&1 || { echo "TRAIN FAILED" >> $LOG; exit 1; }
echo "########## allocation policies ##########" >> $LOG
$PY exp/d3acq.py run --contrast T2 --kind rstd --hi rhi >> $LOG 2>&1 || echo "RUN FAILED" >> $LOG
echo "ACQ-RES DONE" >> $LOG
