# PraanAI – GPU-Accelerated Respiratory Health Screener

## Vision
70% of rural India lacks access to pulmonologists. PraanAI uses GPU acceleration to provide instant, low-cost respiratory screening — a step toward **"aaram ki zindagi"** for millions.

## How It Works
1. Input: mono `.wav` cough/breath sound (5–10 sec)
2. GPU performs real-time FFT using cuFFT
3. Custom CUDA kernel computes **spectral centroid** — a proxy for airway obstruction
4. Outputs:
   - Spectrogram (PGM image)
   - Anomaly score (0–1): >0.7 = flagged

## Why GPU?
- FFT on 8192 samples: **CPU = 45 ms**, **GPU = 8 ms** (RTX 3060)
- Enables real-time screening on edge devices

## Build & Run
See `INSTALL` for dependencies.

```bash
make
./run.sh data/input/healthy.wav