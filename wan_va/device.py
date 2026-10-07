import os

import torch


def get_device(local_rank: int = 0, requested: str | None = None) -> torch.device:
    device_type = (requested or os.getenv("ZERO_WAM_DEVICE", "auto")).lower()
    if device_type == "auto":
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
        else:
            device_type = "cpu"

    if device_type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("Intel XPU was requested but is unavailable")
        return torch.device(f"xpu:{local_rank}")
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(f"cuda:{local_rank}")
    if device_type == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported ZERO_WAM_DEVICE={device_type!r}")


def set_device(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def distributed_backend(device: torch.device) -> str:
    if device.type == "xpu":
        return "xccl"
    if device.type == "cuda":
        return "nccl"
    return "gloo"