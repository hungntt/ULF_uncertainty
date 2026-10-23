#!/bin/bash
# Seed variance on the headline detectability number. The labels are cached and
# shared, so this isolates TRAINING stochasticity: same pipeline, same stimuli,
# same test sites - only the map's initialisation and batch order change.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_night3.log; : > $LOG
for S in 1 2 3; do
  echo -e "\n\n########## detectability map, seed $S ##########" >> $LOG
  $PY exp/d3detect.py --contrast T2 --steps 8000 --seed $S --tag _s$S --labtag "" --resume >> $LOG 2>&1
  $PY exp/d3detect_eval.py --contrast T2 --tag _s$S >> $LOG 2>&1
done
echo "NIGHT3 DONE" >> $LOG
