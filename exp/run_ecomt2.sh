#!/bin/bash
# Second calibration pass, chained behind the main ladder: there is not enough
# spare card to run them together (a llama-server holds 10 GB throughout).
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
while [ "$1" != "" ] && kill -0 "$1" 2>/dev/null; do sleep 120; done
LOG=out/eco_calib2.log; : > $LOG
echo "main queue finished $(date)" >> $LOG
$PY exp/ecomt_calib2.py cache --contrast T2 --bs 16 --memfrac 0.55 >> $LOG 2>&1
$PY exp/ecomt_calib2.py fit   --contrast T2 --memfrac 0.55 >> $LOG 2>&1
echo "CALIB2 DONE" >> $LOG
