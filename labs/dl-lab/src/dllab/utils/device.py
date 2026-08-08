"""Selección de dispositivo (CUDA / MPS / CPU)."""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Devuelve el mejor dispositivo disponible.

    prefer: "auto" | "cuda" | "mps" | "cpu"
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    """Cadena legible con info del dispositivo, útil como primera celda de un notebook."""
    device = device or get_device()
    lines = [f"torch {torch.__version__} | device: {device}"]
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        lines.append(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
        lines.append(f"CUDA runtime: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
    return "\n".join(lines)
