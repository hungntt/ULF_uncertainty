#!/bin/bash
# Two more seeds per architecture: the headline rests on a +-0.048 spread from n=3.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_archseed2.log; : > $LOG
for S in 3 4; do
  for M in unet pix2pix; do
    echo -e "\n\n########## $M seed $S ##########" >> $LOG
    $PY exp/baselines.py --contrast T2 --method $M --steps 8000 --seed $S --tag ${M}_s$S >> $LOG 2>&1
  done
done
echo "ARCHSEED2 TRAIN DONE" >> $LOG
$PY exp/d3archseed.py >> $LOG 2>&1
echo "ARCHSEED2 DONE" >> $LOG
