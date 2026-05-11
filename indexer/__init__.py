# indexer package

from __future__ import annotations

import logging
import os

import torch


logger = logging.getLogger(__name__)


def resolve_device(*env_vars: str) -> str:
    """Pick a torch device: env override (first non-empty match) → CUDA → Apple MPS → CPU."""
    candidates = env_vars or ("EMBED_DEVICE",)
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_device_cache(device: str) -> None:
    """Release cached memory on the active accelerator. No-op on CPU."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
