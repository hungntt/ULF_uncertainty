#!/bin/bash
# External-cohort replication (OpenNeuro ds006557 sessions HFC and HFE, Zenodo v5).
# GPU stages run first; SynthSeg (CPU) runs per cohort once its zoo volumes are exported.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/external; mkdir -p $LOG
# pack and qc were run by hand on 2026-10-28 (out/prep_external_qc.json)


# polarity and the map / closed-loop check: short GPU jobs
for S in ext_on_hfc ext_on_hfe ext_v5; do
  $PY exp/d3lesiontype.py $S > $LOG/lesiontype_$S.log 2>&1
done
$PY exp/d3external.py test ext_on_hfc ext_on_hfe ext_v5 > $LOG/d3external.log 2>&1 &

# the 39-model zoo on each cohort (two at a time on the shared GPU)
$PY exp/d3zoo.py --chunk 4 --split ext_v5     > $LOG/zoo_ext_v5.log 2>&1 &
$PY exp/d3zoo.py --chunk 4 --split ext_on_hfc > $LOG/zoo_ext_on_hfc.log 2>&1
wait
$PY exp/d3zoo.py --chunk 4 --split ext_on_hfe > $LOG/zoo_ext_on_hfe.log 2>&1 &

# SynthSeg on the exported volumes, then the statistics, cohort by cohort
for S in ext_v5 ext_on_hfc ext_on_hfe; do
  [ $S = ext_on_hfe ] && wait
  export ULF_SPLIT=$S
  $PY exp/d3synthseg.py prepare > $LOG/synthseg_$S.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'/mnt/G/Projects/ULF-MICCAI'); import exp.d3synthseg as s; s.run(threads=30, workers=3)" >> $LOG/synthseg_$S.log 2>&1
  $PY exp/d3synthseg.py score >> $LOG/synthseg_$S.log 2>&1
  $PY exp/d3zoo_stats.py > $LOG/stats_$S.log 2>&1
done
echo "EXTERNAL DONE $(date)" > $LOG/DONE
