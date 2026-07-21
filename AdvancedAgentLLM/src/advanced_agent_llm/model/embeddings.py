import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_size: int, base: float = 10_000.0):
        super().__init__()

        if head_size % 2 != 0:
            raise ValueError("head_size must be even for rotary embeddings")

        inverse_frequencies = 1.0 / (
            base ** (torch.arange(0, head_size, 2).float() / head_size)
        )
        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        angles = position_ids.float().unsqueeze(-1) * self.inverse_frequencies
        return angles.cos().to(dtype), angles.sin().to(dtype)


def apply_rotary_position_embedding(
    vectors: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    even_dimensions = vectors[..., 0::2]
    odd_dimensions = vectors[..., 1::2]

    rotated_even = even_dimensions * cosine - odd_dimensions * sine
    rotated_odd = even_dimensions * sine + odd_dimensions * cosine

    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

