import torch
import torch.nn as nn

from model.transformer_block.feed_forward import FeedForward
from model.transformer_block.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int, number_of_heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.attention = MultiHeadAttention(embedding_size, number_of_heads)

        self.feed_forward_norm = nn.LayerNorm(embedding_size)
        self.feed_forward = FeedForward(embedding_size)

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        vectors = input_vectors + self.attention(
            self.attention_norm(input_vectors)
        )
        vectors = vectors + self.feed_forward(
            self.feed_forward_norm(vectors)
        )
        return vectors
