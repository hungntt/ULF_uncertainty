#!/bin/bash
# Seed replication of the decisive comparisons. Every difference in the ladder
# is between -0.001 and -0.009 masked SSIM, and FINDINGS.md already warns that
# no result in this study carries a seed-variance estimate. The claim that
# matters - that a random gate of matched mean weight does as well as the
# evidence-based ones - cannot rest on one seed.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_ecomt_seeds.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; date >> $LOG; }
for S in 1 2; do
  for M in b1 b2 b2rand b5vox; do
    say "seed $S  $M"
    $PY exp/ecomt.py train --contrast T2 --mode $M --lam 1.0 --seed $S --tag ${M}_s$S >> $LOG 2>&1
  done
done
say "SEEDS DONE"
