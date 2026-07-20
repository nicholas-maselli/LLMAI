import torch
import torch.nn as nn

from advanced_llm.config import ModelConfig
from advanced_llm.model.attention import GroupedQueryAttention
from advanced_llm.model.mlp import SwiGLU, calculate_intermediate_size
from advanced_llm.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.attention_norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_epsilon,
        )
        self.attention = GroupedQueryAttention(
            hidden_size=config.hidden_size,
            number_of_attention_heads=config.number_of_attention_heads,
            number_of_key_value_heads=config.number_of_key_value_heads,
            rope_base=config.rope_base,
        )

        intermediate_size = calculate_intermediate_size(
            hidden_size=config.hidden_size,
            multiplier=config.feed_forward_multiplier,
        )
        self.mlp_norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_epsilon,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x
