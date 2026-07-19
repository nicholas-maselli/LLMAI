import torch
import torch.nn as nn

from model.transformer_block.attention import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int, number_of_heads: int):
        super().__init__()

        if embedding_size % number_of_heads != 0:
            raise ValueError("embedding_size must be divisible by number_of_heads")

        head_size = embedding_size // number_of_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(embedding_size, head_size)
                for _ in range(number_of_heads)
            ]
        )
        self.output_projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(input_vectors) for head in self.heads]
        combined_heads = torch.cat(head_outputs, dim=-1)
        return self.output_projection(combined_heads)
