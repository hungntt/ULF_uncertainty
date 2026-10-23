#!/bin/bash
# Everything after the baseline suite. One GPU, strictly sequential.
# Ordered so the self-contained results land first.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_all.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; }
while pgrep -f "exp/baselines.py" > /dev/null; do sleep 30; done
say "baselines released the GPU"

# ---- Direction 2: cross-contrast fusion (cheapest, fully self-contained) ----
say "D2 T1-only -> T1";   $PY exp/d2.py --use T1    --out T1   --steps 10000 --tag T1only  >> $LOG 2>&1
say "D2 T2-only -> T2";   $PY exp/d2.py --use T2    --out T2   --steps 10000 --tag T2only  >> $LOG 2>&1
say "D2 joint -> both";   $PY exp/d2.py --use T1,T2 --out both --steps 10000 --tag joint   >> $LOG 2>&1
say "D2 cross T2 -> T1";  $PY exp/d2.py --use T2    --out T1   --steps 10000 --tag T2toT1  >> $LOG 2>&1
say "D2 cross T1 -> T2";  $PY exp/d2.py --use T1    --out T2   --steps 10000 --tag T1toT2  >> $LOG 2>&1
say "D2 joint + dropout"; $PY exp/d2.py --use T1,T2 --out both --steps 10000 --drop --tag joint_drop >> $LOG 2>&1

# ---- Direction 1: corpus amplification, T2 arm ----
say "D1 forward operator T2"; $PY exp/d1.py forward --contrast T2 --steps 6000 >> $LOG 2>&1
for M in handcrafted learned learned_cal; do
  say "D1 pairs T2 $M"; $PY exp/d1.py pairs --contrast T2 --mode $M >> $LOG 2>&1
done
say "D1 operator check T2"; $PY exp/d1_operator_check.py T2 >> $LOG 2>&1
for M in none handcrafted learned learned_cal; do
  say "D1 train T2 $M"; $PY exp/d1.py train --contrast T2 --mode $M --pre_steps 8000 --ft_steps 6000 >> $LOG 2>&1
done

# ---- Direction 3: risk control, T2 arm ----
for M in cqr gauss; do
  say "D3 $M T2"; $PY exp/d3.py --contrast T2 --mode $M --steps 10000 --alpha 0.1 >> $LOG 2>&1
done

# ---- T1 arms ----
say "D1 forward operator T1"; $PY exp/d1.py forward --contrast T1 --steps 6000 >> $LOG 2>&1
say "D1 pairs T1 learned_cal"; $PY exp/d1.py pairs --contrast T1 --mode learned_cal >> $LOG 2>&1
say "D1 operator check T1"; $PY exp/d1_operator_check.py T1 >> $LOG 2>&1
for M in none learned_cal; do
  say "D1 train T1 $M"; $PY exp/d1.py train --contrast T1 --mode $M --pre_steps 8000 --ft_steps 6000 >> $LOG 2>&1
done
for M in cqr gauss; do
  say "D3 $M T1"; $PY exp/d3.py --contrast T1 --mode $M --steps 10000 --alpha 0.1 >> $LOG 2>&1
done
say "ALL DIRECTIONS DONE"
