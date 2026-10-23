#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_qa2.log; : > $LOG
until grep -q "QA TRAINING DONE" out/run_qa.log 2>/dev/null; do sleep 60; done
for C in T2 T1; do
  echo "########## QA evaluation, $C ##########" >> $LOG
  $PY exp/d3qa_eval.py --contrast $C --alpha 0.10 >> $LOG 2>&1
  echo "########## unlabelled-deployment check, $C ##########" >> $LOG
  $PY exp/d3qa_deploy.py --contrast $C >> $LOG 2>&1
done
echo "QA EVAL DONE" >> $LOG
