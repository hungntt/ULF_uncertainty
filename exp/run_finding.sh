#!/bin/bash
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_finding.log; : > $LOG
$PY exp/d3finding.py --contrast T2 >> $LOG 2>&1
echo "FINDING DONE" >> $LOG
