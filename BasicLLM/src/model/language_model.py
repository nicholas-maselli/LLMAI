import torch
import torch.nn as nn

from model.embeddings import InputEmbedding
from model.transformer_block.block import TransformerBlock


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        context_length: int,
        embedding_size: int,
        number_of_heads: int,
    ):
        super().__init__()
        self.input_embedding = InputEmbedding(
            vocabulary_size,
            context_length,
            embedding_size,
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_size, number_of_heads)
                for _ in range(4)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_size)
        self.output_projection = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        vectors = self.input_embedding(token_ids)
        vectors = self.transformer_blocks(vectors)
        vectors = self.final_norm(vectors)
        logits = self.output_projection(vectors)
        return logits
