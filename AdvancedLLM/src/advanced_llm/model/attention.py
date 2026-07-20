import torch
import torch.nn as nn
import torch.nn.functional as F

from advanced_llm.model.embeddings import (
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
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.query(hidden_states)
        keys = self.key(hidden_states)
        values = self.value(hidden_states)
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        queries, keys, values = self.projections(hidden_states)

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

        position_ids = torch.arange(sequence_length, device=hidden_states.device)
        cosine, sine = self.rotary_embedding(position_ids, hidden_states.dtype)
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
