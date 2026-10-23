#!/bin/bash
# Seeds 3-4, extending the decisive arms to five seeds each.
# Three seeds showed two things that need firmer ground: B1's own seed SD
# (0.0086) is larger than every effect in the ladder, and the voxel evidence
# gate beats a matched random gate on 3/3 seeds (+0.0019). An F test on 2 and 2
# df and a sign test on three seeds are both too thin to carry a claim.
# ecomtvox is added to check that the observability factor really is what drags
# the voxel gate down - it was the single worst component in the ladder.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_ecomt_seeds2.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; date >> $LOG; }
for S in 3 4; do
  for M in b1 b5vox b2rand ecomtvox; do
    say "seed $S  $M"
    $PY exp/ecomt.py train --contrast T2 --mode $M --lam 1.0 --seed $S --tag ${M}_s$S >> $LOG 2>&1
  done
done
# ecomtvox at the three original seeds, so its contrast is paired like the rest
for S in 1 2; do
  say "seed $S ecomtvox"
  $PY exp/ecomt.py train --contrast T2 --mode ecomtvox --lam 1.0 --seed $S --tag ecomtvox_s$S >> $LOG 2>&1
done
say "SEEDS2 DONE"
