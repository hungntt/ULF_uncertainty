#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_tail.log; : > $LOG
while pgrep -f "exp/run_all.sh|exp/run_nsubj.sh|exp/run_extra.sh|exp/run_d2extra.sh" > /dev/null; do sleep 60; done
for C in T2 T1; do
  echo -e "\n########## unet_noresid $C ##########" >> $LOG
  $PY exp/baselines.py --contrast $C --method unet_noresid --steps 12000 >> $LOG 2>&1
done
echo "TAIL DONE" >> $LOG
