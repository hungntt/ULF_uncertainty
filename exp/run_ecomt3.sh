#!/bin/bash
# Stage 1 after the schedule fix. The first pass put a 2000-step supervised
# warm-up in front of a cohort whose supervised optimum arrives at step 2000, so
# checkpoint selection always returned the pre-consistency model and every arm
# reproduced B1 exactly. The consistency term now reaches full strength at step
# 1500, and B1 is produced by this same script with lam = 0 so the control is
# matched in sampler, optimiser, schedule and selection grid.
#
# Gate B first: if vanilla MT cannot clear the paired-only control at matched
# compute, the gated arms are not worth running.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_ecomt3.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; date >> $LOG; }

say "B1 paired-only control (lam=0, same code path)"
$PY exp/ecomt.py train --contrast T2 --mode b1 >> $LOG 2>&1

for L in 3.0 1.0 10.0; do
  say "B2 vanilla Mean Teacher, lambda $L"
  $PY exp/ecomt.py train --contrast T2 --mode b2 --lam $L --tag b2_lam$L >> $LOG 2>&1
done

say "second calibration pass: per-scale MIND, voxel-level trust"
$PY exp/ecomt_calib2.py cache --contrast T2 --bs 16 --memfrac 0.55 >> $LOG 2>&1
$PY exp/ecomt_calib2.py fit   --contrast T2 --memfrac 0.55 >> $LOG 2>&1
say "STAGE 1 DONE"
