#!/bin/bash
# Isolates the *input-side* cross-contrast gain: identical target, capacity and
# budget as the single-contrast runs; only the visible ULF contrasts differ.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_d2extra.log; : > $LOG
while pgrep -f "exp/run_all.sh|exp/run_nsubj.sh|exp/run_extra.sh" > /dev/null; do sleep 60; done
echo -e "\n########## both -> T1 ##########" >> $LOG
$PY exp/d2.py --use T1,T2 --out T1 --steps 10000 --tag bothtoT1 >> $LOG 2>&1
echo -e "\n########## both -> T2 ##########" >> $LOG
$PY exp/d2.py --use T1,T2 --out T2 --steps 10000 --tag bothtoT2 >> $LOG 2>&1
echo "D2 EXTRA DONE" >> $LOG
