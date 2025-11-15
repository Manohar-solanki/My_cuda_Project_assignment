#!/bin/bash
set -e

# Create dirs
mkdir -p data/input data/output

# Download 10 sample WAVs (replace with full dataset later)
cd data/input
wget -nc https://www.dsprelated.com/freebooks/pasp/figs/e_gtr_6s.wav
wget -nc https://www.dsprelated.com/freebooks/pasp/figs/e_gtr_fb_6s.wav
wget -nc https://www.dsprelated.com/freebooks/pasp/figs/e_gtr_wah_6s.wav
cd ../..

# Build
make

# Run
./bin/main data/input data/output