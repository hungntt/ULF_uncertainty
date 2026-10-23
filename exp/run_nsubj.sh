#!/bin/bash
# How many paired subjects does a ULF enhancer need, with and without corpus amplification?
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_nsubj.log; : > $LOG
while pgrep -f "exp/run_all.sh|exp/run_extra.sh" > /dev/null; do sleep 60; done
for N in 4 8 16 32; do
  for M in none learned_cal; do
    echo -e "\n\n########## n=$N mode=$M ##########" >> $LOG
    $PY exp/d1.py train --contrast T2 --mode $M --n_subj $N \
        --pre_steps 8000 --ft_steps 6000 --tag "n${N}_${M}" >> $LOG 2>&1
  done
done
echo "NSUBJ SWEEP DONE" >> $LOG
