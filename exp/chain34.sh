#!/bin/bash
cd /mnt/G/Projects/ULF-MICCAI
for i in $(seq 1 720); do
  grep -q 'NIGHT3 DONE' out/run_night3.log 2>/dev/null && break
  sleep 30
done
bash exp/run_night4.sh
