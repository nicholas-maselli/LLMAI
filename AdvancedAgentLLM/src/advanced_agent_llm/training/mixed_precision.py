from contextlib import nullcontext
from typing import ContextManager

import torch


def autocast_context(
    device: torch.device,
    precision: str,
) -> ContextManager[None]:
    if precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def create_gradient_scaler(
    device: torch.device,
    precision: str,
) -> torch.amp.GradScaler:
    enabled = device.type == "cuda" and precision == "fp16"
    return torch.amp.GradScaler(device.type, enabled=enabled)
