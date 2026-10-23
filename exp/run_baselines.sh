#!/bin/bash
PY=/mnt/G/Projects/FlowPoC/.venv/bin/python
cd /mnt/G/Projects/ULF-MICCAI
for C in T2 T1; do
  $PY -I exp/baselines.py --contrast $C --method identity
  for M in unet unet_ssim pix2pix lohiresgan; do
    echo "######## $M $C ########"
    $PY -I exp/baselines.py --contrast $C --method $M --steps 12000
  done
done
echo "ALL BASELINES DONE"
