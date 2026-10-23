#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_qa.log; : > $LOG
for C in T2 T1; do
  echo "########## cross-fitting the enhancer, $C ##########" >> $LOG
  $PY exp/d3qa.py crossfit --contrast $C --steps 9000 >> $LOG 2>&1
  echo "########## training the QA map, $C ##########" >> $LOG
  $PY exp/d3qa.py qa --contrast $C --steps 9000 >> $LOG 2>&1
done
echo "QA TRAINING DONE" >> $LOG
