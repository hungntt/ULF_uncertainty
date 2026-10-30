#!/bin/bash
# Seed study for all six enhancers at the ORIGINAL 12000-step budget. The earlier seed runs
# (tags *_s1..s4) used 8000 steps against seed 0's 12000, so budget and seed were confounded.
# Seed 0 of every enhancer is the existing Table-1 model; this adds seeds 1-4.
# Two lanes: a GAN job holds ~8 GB, and four lanes ran the 24 GB card out of memory.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOGD=out/seeds12k; mkdir -p $LOGD
JOBS=$LOGD/jobs.txt; : > $JOBS
# D1 chain first (longest): regenerate the deleted pseudo-pairs (deterministic), then 4 seeds
echo "$PY exp/d1.py pairs --contrast T2 --mode learned_cal > $LOGD/d1_pairs.log 2>&1; for S in 1 2 3 4; do $PY exp/d1.py train --contrast T2 --mode learned_cal --pre_steps 8000 --ft_steps 6000 --seed \$S --tag learned_cal_seed\$S > $LOGD/d1_seed\$S.log 2>&1; done" >> $JOBS
for S in 1 2 3 4; do
  for M in lohiresgan pix2pix unet unet_ssim unet_noresid; do
    echo "$PY exp/baselines.py --contrast T2 --method $M --steps 12000 --seed $S --tag ${M}_seed$S > $LOGD/${M}_seed$S.log 2>&1" >> $JOBS
  done
done
tr '\n' '\0' < $JOBS | xargs -0 -P 2 -I{} bash -c "{}"
echo "SEEDS12K DONE" > $LOGD/DONE
