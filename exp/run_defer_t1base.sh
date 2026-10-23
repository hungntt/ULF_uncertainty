#!/bin/bash
# The baseline suite runs T2 then T1. The three directions matter more than the
# second baseline contrast, so: once the T2 arm is complete, drop the suite's
# parent (its running python finishes, orphaned, and nothing else spawns) and
# re-queue the T1 baselines behind everything else.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_t1base.log; : > $LOG
until [ -f out/base_lohiresgan_T2.json ]; do
  pgrep -f "exp/run_baselines.sh" > /dev/null || { echo "suite already gone" >> $LOG; break; }
  sleep 30
done
sleep 20
pkill -f "exp/run_baselines.sh" && echo "dropped baseline suite parent" >> $LOG
while pgrep -f "exp/run_all.sh|exp/run_nsubj.sh|exp/run_extra.sh|exp/run_d2extra.sh|exp/run_tail.sh" > /dev/null; do sleep 60; done
$PY exp/baselines.py --contrast T1 --method identity >> $LOG 2>&1
for M in unet unet_ssim pix2pix lohiresgan; do
  echo -e "\n########## $M T1 ##########" >> $LOG
  $PY exp/baselines.py --contrast T1 --method $M --steps 12000 >> $LOG 2>&1
done
echo "T1 BASELINES DONE" >> $LOG
