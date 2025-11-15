# CUDA Batch FFT Spectrogram Generator

Processes hundreds of WAV files using CUDA's cuFFT to generate magnitude spectrograms.

## Requirements
- NVIDIA GPU with CUDA support
- `nvcc`, `make`
- `libsndfile-dev` (for WAV I/O)

## Build
```bash
make