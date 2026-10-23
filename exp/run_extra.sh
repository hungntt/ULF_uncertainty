#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_extra.log; : > $LOG
while pgrep -f "exp/run_all.sh" > /dev/null; do sleep 60; done
for C in T2 T1; do
  echo -e "\n\n########## mismatched pairs $C ##########" >> $LOG
  $PY exp/baselines.py --contrast $C --method unet_mismatched --steps 12000 >> $LOG 2>&1
done
echo "EXTRA DONE" >> $LOG
