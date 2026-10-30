#!/bin/bash
# Second runner for the 12000-step seed study (replaces run_seeds12k.sh's queue after the
# first two jobs started). One lane per instance, next to the D1 chain from run_seeds12k.sh;
# start a second instance when memory allows. A job is skipped if its checkpoint exists or another instance holds
# its lock, so the runner can be restarted safely.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOGD=out/seeds12k
JOBS=$LOGD/jobs_b.txt; : > $JOBS
for S in 1 2 3 4; do
  for M in pix2pix unet lohiresgan unet_ssim unet_noresid; do
    T=${M}_seed$S
    [ "$T" = lohiresgan_seed1 ] && continue          # already running from the first runner
    echo "[ -f out/base_${T}_T2.pt ] || ! mkdir $LOGD/lock_$T 2>/dev/null || $PY exp/baselines.py --contrast T2 --method $M --steps 12000 --seed $S --tag $T > $LOGD/$T.log 2>&1" >> $JOBS
  done
done
tr '\n' '\0' < $JOBS | xargs -0 -P 1 -I{} bash -c "{}"
echo "SEEDS12K_B DONE" > $LOGD/DONE_b
