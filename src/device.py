"""Compute-backend detection that works identically on AMD ROCm and NVIDIA CUDA.

The key portability fact this repo demonstrates: a PyTorch ROCm build exposes AMD
GPUs through the *same* ``torch.cuda`` API and the same ``"cuda"`` device string as
NVIDIA. So the training/benchmark code never branches on vendor -- only this module
inspects the backend, purely for reporting.
"""

from __future__ import annotations

import platform

import torch


def get_device() -> torch.device:
    """Return the best available device.

    On a ROCm PyTorch build this returns ``cuda`` for an AMD GPU -- HIP maps the
    CUDA runtime API onto AMD hardware, so the string is still ``"cuda"``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def backend_name() -> str:
    """Human-readable backend: 'ROCm (HIP)', 'CUDA', or 'CPU'."""
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "ROCm (HIP)"
        if getattr(torch.version, "cuda", None):
            return "CUDA"
        return "GPU (unknown runtime)"
    return "CPU"


def device_info() -> dict[str, object]:
    """Collect backend/runtime details for logging and benchmark headers."""
    info: dict[str, object] = {
        "backend": backend_name(),
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def print_device_info() -> None:
    info = device_info()
    print("=" * 62)
    print(" Compute backend")
    print("=" * 62)
    for key, value in info.items():
        print(f"  {key:16}: {value}")
    print("=" * 62)


def synchronize() -> None:
    """Block until all queued GPU work finishes -- required for honest timing.

    A no-op on CPU. Works for both ROCm and CUDA via the shared ``torch.cuda`` API.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
