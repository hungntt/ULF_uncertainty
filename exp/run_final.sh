#!/bin/bash
# Everything still outstanding, in one sequential script.
# No cross-script pgrep waits: a shell wrapper whose command line merely *contains*
# another script's text matches those patterns and deadlocks the chain.
PY="/mnt/G/Projects/FlowPoC/.venv/bin/python -I -u"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /mnt/G/Projects/ULF-MICCAI
LOG=out/run_final.log; : > $LOG
say(){ echo -e "\n\n########## $* ##########" >> $LOG; }

say "D2 input-side control: both -> T1"
$PY exp/d2.py --use T1,T2 --out T1 --steps 10000 --tag bothtoT1 >> $LOG 2>&1
say "D2 input-side control: both -> T2"
$PY exp/d2.py --use T1,T2 --out T2 --steps 10000 --tag bothtoT2 >> $LOG 2>&1

say "architecture ablation: U-Net without the residual anchor (T2)"
$PY exp/baselines.py --contrast T2 --method unet_noresid --steps 12000 >> $LOG 2>&1

say "T1 baselines"
for M in unet unet_ssim pix2pix lohiresgan; do
  say "baseline $M T1"
  $PY exp/baselines.py --contrast T1 --method $M --steps 12000 >> $LOG 2>&1
done

say "confound-free cohort sweep (operator refitted per subset)"
for N in 4 8 16; do
  say "clean sweep n=$N"
  $PY exp/d1.py forward --contrast T2 --steps 6000 --n_subj $N --ptag "_n$N" >> $LOG 2>&1
  $PY exp/d1.py pairs   --contrast T2 --mode learned_cal --n_subj $N --ptag "_n$N" >> $LOG 2>&1
  $PY exp/d1.py train   --contrast T2 --mode learned_cal --n_subj $N --ptag "_n$N" \
      --pre_steps 8000 --ft_steps 6000 --tag "cn${N}_learned_cal" >> $LOG 2>&1
  rm -f /mnt/G/Projects/ULF-MICCAI/datasets/npy/d1_pseudo_*_n$N.npy
done

say "U-Net without residual anchor (T1)"
$PY exp/baselines.py --contrast T1 --method unet_noresid --steps 12000 >> $LOG 2>&1
echo "FINAL QUEUE DONE" >> $LOG
