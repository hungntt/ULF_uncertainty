#!/bin/bash
# Each contrast end-to-end, so T2 results land before T1 starts rather than after.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_qa.log; : > $LOG
for C in T2 T1; do
  echo -e "\n########## [$C] cross-fitting the enhancer (4 folds) ##########" >> $LOG
  $PY exp/d3qa.py crossfit --contrast $C --steps 9000 >> $LOG 2>&1 || { echo "CROSSFIT $C FAILED" >> $LOG; continue; }
  echo -e "\n########## [$C] training the QA map ##########" >> $LOG
  $PY exp/d3qa.py qa --contrast $C --steps 9000 >> $LOG 2>&1 || { echo "QA $C FAILED" >> $LOG; continue; }
  echo -e "\n########## [$C] evaluation vs unsupervised baselines ##########" >> $LOG
  $PY exp/d3qa_eval.py --contrast $C --alpha 0.10 >> $LOG 2>&1 || echo "EVAL $C FAILED" >> $LOG
  echo -e "\n########## [$C] unlabelled-deployment check (48 ULF-only subjects) ##########" >> $LOG
  $PY exp/d3qa_deploy.py --contrast $C >> $LOG 2>&1 || echo "DEPLOY $C FAILED" >> $LOG
  echo "=== $C COMPLETE ===" >> $LOG
done
echo "QA ALL DONE" >> $LOG
