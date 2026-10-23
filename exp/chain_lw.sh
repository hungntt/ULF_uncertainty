#!/bin/bash
cd /mnt/G/Projects/ULF-MICCAI
for i in $(seq 1 960); do
  grep -q 'ARCHSEED DONE' out/run_archseed.log 2>/dev/null && break
  sleep 30
done
bash exp/run_lw.sh
