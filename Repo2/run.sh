#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input.wav>"
    exit 1
fi

INPUT="$1"
OUTPUT="data/output/$(basename "$INPUT" .wav)_spectrogram.pgm"
LOG="data/output/$(basename "$INPUT" .wav)_log.txt"

mkdir -p data/output

./bin/praanai --input "$INPUT" --output "$OUTPUT" 2>&1 | tee "$LOG"
echo "Output saved to: $OUTPUT and $LOG"