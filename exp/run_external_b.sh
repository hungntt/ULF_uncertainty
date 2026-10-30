#!/bin/bash
# Continuation of run_external.sh (replaced after the GPU stages started):
#  - ext_on_hfe reuses ext_on_hfc's closed-loop quantities (byte-identical GE 3 T arrays) and
#    computes only SSIM/PSNR on its own acquired 64 mT input and the export for SynthSeg;
#  - SynthSeg and the statistics run per cohort as soon as that cohort's zoo is complete.
# Waits on the two zoo processes by PID, given as arguments.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/external
V5PID=$1; HFCPID=$2
seg() {   # $1 split, $2 threads
  export ULF_SPLIT=$1
  $PY exp/d3synthseg.py prepare > $LOG/synthseg_$1.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'/mnt/G/Projects/ULF-MICCAI'); import exp.d3synthseg as s; s.run(threads=$2, workers=2)" >> $LOG/synthseg_$1.log 2>&1
  $PY exp/d3synthseg.py score >> $LOG/synthseg_$1.log 2>&1
  $PY exp/d3zoo_stats.py > $LOG/stats_$1.log 2>&1
}
while kill -0 $V5PID 2>/dev/null; do sleep 30; done
seg ext_v5 16 &
while kill -0 $HFCPID 2>/dev/null; do sleep 30; done
$PY exp/d3zoo.py --chunk 8 --split ext_on_hfe --reuse_from ext_on_hfc > $LOG/zoo_ext_on_hfe.log 2>&1
wait
seg ext_on_hfc 30
seg ext_on_hfe 30
echo "EXTERNAL DONE $(date)" > $LOG/DONE
