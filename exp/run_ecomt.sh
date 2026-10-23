#!/bin/bash
# ECO-MT: calibration, the lambda choice on dev, then the baseline ladder.
# One GPU, strictly sequential. The locked 10 test subjects never enter a
# decision: lambda is picked on the 8 dev-val subjects, every arm selects its
# checkpoint on the same 8, and the test set is scored once at the end.
#
# Waits on the cascade job by PID rather than by pgrep pattern: a shell wrapper
# whose command line merely contains the other script's name matches the pattern
# and deadlocks the chain (see the note in run_final.sh).
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_ecomt.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; date >> $LOG; }

WAIT_PID=${1:-0}
while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
say "GPU released"

say "calibration T2 (inner 3-fold + operational teacher)"
$PY exp/ecomt.py calib --contrast T2 --steps 6000 >> $LOG 2>&1

for L in 0.3 1.0 3.0; do
  say "B2 vanilla Mean Teacher, lambda $L"
  $PY exp/ecomt.py train --contrast T2 --mode b2 --lam $L --tag b2_lam$L >> $LOG 2>&1
done
LAM=$($PY exp/ecomt.py pick --contrast T2)
say "lambda $LAM selected on dev-val"
cp out/eco_b2_lam${LAM}_T2.json out/eco_b2_T2.json
cp out/eco_b2_lam${LAM}_T2.pt   out/eco_b2_T2.pt

for M in ecomt b5 b6 b3 b2rand b4; do
  say "ladder $M, lambda $LAM"
  $PY exp/ecomt.py train --contrast T2 --mode $M --lam $LAM >> $LOG 2>&1
done
say "T2 report"
$PY exp/ecomt.py report --contrast T2 > out/ecomt_report_T2.md 2>&1

say "T1 headline arms"
$PY exp/ecomt.py calib --contrast T1 --steps 6000 >> $LOG 2>&1
for M in b2 ecomt; do
  say "T1 $M, lambda $LAM"
  $PY exp/ecomt.py train --contrast T1 --mode $M --lam $LAM >> $LOG 2>&1
done
$PY exp/ecomt.py report --contrast T1 > out/ecomt_report_T1.md 2>&1
say "ECO-MT QUEUE DONE"
