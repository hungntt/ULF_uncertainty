#!/bin/bash
# Re-run the two single-contrast arms that were trained before the target-channel
# fix (the sampler returned both 3T contrasts, so a 1-channel head regressed
# against their mean via silent broadcasting).
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_d2fix.log; : > $LOG
# start as soon as run_all leaves the Direction 2 block, not after the whole queue
until [ -f out/d2_joint_drop.json ]; do sleep 45; done
for T in T1 T2; do
  echo -e "\n########## ${T}only (refit) ##########" >> $LOG
  $PY exp/d2.py --use $T --out $T --steps 10000 --tag ${T}only >> $LOG 2>&1
done
echo "D2 FIX DONE" >> $LOG
