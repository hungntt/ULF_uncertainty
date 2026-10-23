#!/bin/bash
# The gated ladder, at the dev-selected lambda = 1.0.
#
# Gate B failed: vanilla MT costs 0.0072 masked SSIM against the paired-only
# control, and more of it costs more (lam 3 -> -0.0161, lam 10 -> -0.0271). The
# ladder is still worth running because the trajectory says why: the consistency
# term almost removes the overfitting decay (B1 falls 0.046 dev SSIM from its
# peak to step 12000, lam=1 falls 0.0095) while capping the peak, because the
# pseudo-target pulls the student toward a teacher that is worse than the best
# student. Selective consistency is exactly the proposal's answer to that, so
# the question is whether gating recovers the loss. Arms are ordered with the
# most likely recoveries first.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_ecomt5.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; date >> $LOG; }

for M in ecomtvox b6 b5vox ecomt b5 b3 b2rand b4; do
  say "ladder $M, lambda 1.0"
  $PY exp/ecomt.py train --contrast T2 --mode $M --lam 1.0 >> $LOG 2>&1
  $PY exp/ecomt.py report --contrast T2 > out/ecomt_report_T2.md 2>&1
done
say "T2 LADDER DONE"
