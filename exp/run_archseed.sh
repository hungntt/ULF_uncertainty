#!/bin/bash
# Seed-control the architecture headline: is pix2pix's lesion advantage a property of
# the architecture, or of one lucky training run?
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_archseed.log; : > $LOG
for S in 1 2; do
  for M in unet pix2pix; do
    echo -e "\n\n########## $M seed $S ##########" >> $LOG
    $PY exp/baselines.py --contrast T2 --method $M --steps 8000 --seed $S --tag ${M}_s$S >> $LOG 2>&1
  done
done
echo "ARCHSEED TRAIN DONE" >> $LOG
$PY exp/d3archseed.py >> $LOG 2>&1
echo "ARCHSEED DONE" >> $LOG
