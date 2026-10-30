#!/bin/bash
# Evaluate seed models as they finish training: zoo metrics (GPU), then SynthSeg (CPU) on the
# exported volumes. Both stages skip work already done. Stops when all 39 models are scored.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
while true; do
  $PY exp/d3zoo.py --chunk 3 >> out/d3zoo_rolling.log 2>&1
  $PY exp/d3synthseg.py prepare >> out/d3synthseg_rolling.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'/mnt/G/Projects/ULF-MICCAI'); import exp.d3synthseg as s; s.run(threads=12)" >> out/d3synthseg_rolling.log 2>&1
  n=$(python3 -c "import json; print(len(json.load(open('out/d3zoo_T2.json'))))")
  ns=$(ls -d datasets/scratch/synthseg/seg/*/ | wc -l)
  echo "$(date) zoo $n/39 synthseg folders $ns/41" >> out/zoo_rolling_status.log
  [ "$n" -ge 39 ] && [ "$ns" -ge 41 ] && break
  sleep 600
done
$PY exp/d3synthseg.py score >> out/d3synthseg_rolling.log 2>&1
$PY exp/d3zoo_stats.py > out/d3zoo_stats_final.log 2>&1
echo "ROLLING DONE $(date)" >> out/zoo_rolling_status.log
