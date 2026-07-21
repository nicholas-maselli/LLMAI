import torch
import torch.nn as nn
import torch.nn.functional as F

from advanced_agent_llm.model.embeddings import (
    RotaryPositionEmbedding,
    apply_rotary_position_embedding,
)


class GroupedQueryAttentionProjections(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        number_of_attention_heads: int,
        number_of_key_value_heads: int,
    ):
        super().__init__()

        if hidden_size % number_of_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by number_of_attention_heads")
        if number_of_attention_heads % number_of_key_value_heads != 0:
            raise ValueError(
                "number_of_attention_heads must be divisible by number_of_key_value_heads"
            )

        self.head_size = hidden_size // number_of_attention_heads
        self.number_of_attention_heads = number_of_attention_heads
        self.number_of_key_value_heads = number_of_key_value_heads

        self.query = nn.Linear(
            hidden_size,
            number_of_attention_heads * self.head_size,
            bias=False,
        )
        self.key = nn.Linear(
            hidden_size,
            number_of_key_value_heads * self.head_size,
            bias=False,
        )
        self.value = nn.Linear(
            hidden_size,
            number_of_key_value_heads * self.head_size,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        return queries, keys, values


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        number_of_attention_heads: int,
        number_of_key_value_heads: int,
        rope_base: float = 10_000.0,
    ):
        super().__init__()

        self.number_of_attention_heads = number_of_attention_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.projections = GroupedQueryAttentionProjections(
            hidden_size,
            number_of_attention_heads,
            number_of_key_value_heads,
        )
        self.rotary_embedding = RotaryPositionEmbedding(
            self.projections.head_size,
            rope_base,
        )
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = x.shape
        queries, keys, values = self.projections(x)

        queries = queries.view(
            batch_size,
            sequence_length,
            self.number_of_attention_heads,
            self.projections.head_size,
        ).transpose(1, 2)
        keys = keys.view(
            batch_size,
            sequence_length,
            self.number_of_key_value_heads,
            self.projections.head_size,
        ).transpose(1, 2)
        values = values.view(
            batch_size,
            sequence_length,
            self.number_of_key_value_heads,
            self.projections.head_size,
        ).transpose(1, 2)

        position_ids = torch.arange(sequence_length, device=x.device)
        cosine, sine = self.rotary_embedding(position_ids, x.dtype)
        queries = apply_rotary_position_embedding(queries, cosine, sine)
        keys = apply_rotary_position_embedding(keys, cosine, sine)

        attention_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            is_causal=True,
            enable_gqa=True,
        )

        combined_heads = attention_output.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            hidden_size,
        )
        return self.output_projection(combined_heads)

