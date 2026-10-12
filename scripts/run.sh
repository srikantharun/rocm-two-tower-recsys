#!/usr/bin/env bash
# Train then benchmark the two-tower recommender on whatever backend is present.
# Prints the detected backend (ROCm / CUDA / CPU) at the top of each command.
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">>> Training"
python -m src.train --epochs 10

echo
echo ">>> Benchmark"
python -m src.benchmark
