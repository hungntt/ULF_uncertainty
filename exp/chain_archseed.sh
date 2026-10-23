#!/bin/bash
cd /mnt/G/Projects/ULF-MICCAI
for i in $(seq 1 960); do
  grep -q 'NIGHT5 DONE' out/run_night5.log 2>/dev/null && break
  sleep 30
done
bash exp/run_archseed.sh
