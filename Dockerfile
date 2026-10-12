# Runs the recommender on an AMD GPU using AMD's official ROCm PyTorch image.
# PyTorch is already built against ROCm inside this base image -- no torch install needed.
#
# Build:
#   docker build -t rocm-two-tower .
#
# Run (exposes the AMD GPU to the container):
#   docker run --rm \
#     --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined --group-add video \
#     rocm-two-tower
#
# Swap the tag for the ROCm version matching your driver (see https://hub.docker.com/r/rocm/pytorch).
FROM rocm/pytorch:latest

WORKDIR /app
COPY . /app

# Only the non-torch deps; torch ships with the base image.
RUN pip install --no-cache-dir numpy pytest

# Default: train on the AMD GPU, then benchmark.
CMD ["bash", "-lc", "python -m src.train --epochs 10 && python -m src.benchmark"]
