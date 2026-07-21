import torch
import torch.nn as nn


def create_adamw_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    fused: bool | None = None,
) -> torch.optim.AdamW:
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    if fused is None:
        fused = torch.cuda.is_available()

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=fused,
    )
