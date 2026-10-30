#!/bin/bash
# Honest cohort-size sweep: the forward operator, the spectral calibration target and
# the enhancer all see the same n paired subjects. Without this, a low-n arm inherits
# an operator fitted on all 32 and the curve is optimistic at the left-hand end.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_nsubj_clean.log; : > $LOG
while pgrep -f "exp/run_nsubj.sh|exp/run_d2extra.sh|exp/run_tail.sh|exp/run_defer_t1base.sh" > /dev/null; do sleep 60; done
for N in 4 8 16; do
  echo -e "\n########## clean sweep n=$N ##########" >> $LOG
  $PY exp/d1.py forward --contrast T2 --steps 6000 --n_subj $N --ptag "_n$N" >> $LOG 2>&1
  $PY exp/d1.py pairs   --contrast T2 --mode learned_cal --n_subj $N --ptag "_n$N" >> $LOG 2>&1
  $PY exp/d1.py train   --contrast T2 --mode learned_cal --n_subj $N --ptag "_n$N" \
      --pre_steps 8000 --ft_steps 6000 --tag "cn${N}_learned_cal" >> $LOG 2>&1
  rm -f /mnt/G/Projects/ULF-MICCAI/datasets/npy/d1_pseudo_*_n$N.npy
done
echo "CLEAN SWEEP DONE" >> $LOG
