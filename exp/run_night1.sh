#!/bin/bash
# Overnight phase 1. Each stage writes its own JSON, so a failure costs one stage.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_night1.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; }

say "1. cross-pipeline CONTROL: identical lesion in both inputs (isolates pipeline from conspicuity)"
$PY exp/d3pipe.py --c1_frac 1.0 --tag _same >> $LOG 2>&1

say "2. detectability trained on BOTH polarities, T2 (sign bug fixed)"
$PY exp/d3detect.py --contrast T2 --steps 8000 --tag _signed >> $LOG 2>&1
$PY exp/d3detect_eval.py --contrast T2 --tag _signed >> $LOG 2>&1

say "3. lesion survival across enhancer architectures"
$PY exp/d3arch.py >> $LOG 2>&1

echo "NIGHT1 DONE" >> $LOG
