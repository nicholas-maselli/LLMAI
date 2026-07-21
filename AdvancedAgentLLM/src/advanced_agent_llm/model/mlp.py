import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_intermediate_size(
    hidden_size: int,
    multiplier: float,
    multiple_of: int = 256,
) -> int:
    unrounded_size = int(hidden_size * multiplier)
    return multiple_of * math.ceil(unrounded_size / multiple_of)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.up_projection = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
        )
        self.down_projection = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_projection(x))
        values = self.up_projection(x)
        return self.down_projection(gate * values)

