#!/bin/bash
# Does polarity-violating augmentation teach route C to read the 64 mT T1 channel?
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_night4.log; : > $LOG
echo "########## control: same code path, augmentation off ##########" >> $LOG
$PY exp/d3aug.py --aug 0.0 --tag _ctl >> $LOG 2>&1
echo "########## treatment: lesions at both T1 polarities ##########" >> $LOG
$PY exp/d3aug.py --aug 0.5 --tag _aug >> $LOG 2>&1
echo "########## polarity sweep with the augmented route C ##########" >> $LOG
$PY exp/d3polarity.py --cmodel bothtoT1_ctl --label _ctl >> $LOG 2>&1
$PY exp/d3polarity.py --cmodel bothtoT1_aug --label _aug >> $LOG 2>&1
echo "NIGHT4 SWEEP DONE" >> $LOG
