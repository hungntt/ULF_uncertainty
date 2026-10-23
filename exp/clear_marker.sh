#!/bin/bash
cd /mnt/G/Projects/ULF-MICCAI
until [ -f out/d3v2_T2.json ]; do sleep 60; done
rm -f out/.d3v2_analyze_running
