"""
Device Management Utilities

Centralized device detection and management for optimal hardware acceleration.
Supports CUDA, Apple Silicon MPS, and CPU with intelligent fallback strategies.
"""

import os
import torch
import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["auto", "cuda", "mps", "cpu"]

def setup_mps_environment():
    """Configure environment for optimal MPS performance."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def get_device(preference: DeviceType = "auto", allow_mps: bool = True) -> torch.device:
    """
    Get optimal device with MPS support.

    Args:
        preference: Device preference ("auto", "cuda", "mps", "cpu")
        allow_mps: Whether to allow MPS (set False for sparse tensor operations)

    Returns:
        torch.device for the selected device
    """
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif allow_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        setup_mps_environment()
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_sparse_safe_device() -> torch.device:
    """Get device safe for sparse tensor operations (excludes MPS)."""
    return get_device(allow_mps=False)

def warmup_mps(device: torch.device):
    """Warm up MPS backend to avoid first-use delays."""
    if device.type != "mps":
        return
    try:
        warmup_tensor = torch.randn(100, 100, device=device)
        _ = warmup_tensor @ warmup_tensor.T
        del warmup_tensor
        torch.mps.empty_cache()
        logger.info("MPS backend warmed up successfully")
    except Exception as e:
        logger.warning(f"MPS warmup failed: {e}")

def get_device_info() -> dict:
    """Get detailed device availability info."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "recommended": str(get_device())
    }