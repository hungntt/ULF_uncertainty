#!/bin/bash
# Gate B with the corrected schedule, chained behind the running calib2 cache.
# Strictly serial: the card has ~13 GB free of 24 and one MT arm wants 11.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
while [ "$1" != "" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
LOG=out/run_ecomt3.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; date >> $LOG; }

say "calib2 fit (per-scale MIND, voxel-level trust)"
$PY exp/ecomt_calib2.py fit --contrast T2 --memfrac 0.55 >> $LOG 2>&1

say "B1 paired-only control (lam=0, same code path)"
$PY exp/ecomt.py train --contrast T2 --mode b1 >> $LOG 2>&1
for L in 3.0 1.0 10.0; do
  say "B2 vanilla Mean Teacher, lambda $L"
  $PY exp/ecomt.py train --contrast T2 --mode b2 --lam $L --tag b2_lam$L >> $LOG 2>&1
done
say "GATE B STAGE DONE"
