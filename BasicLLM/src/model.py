import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionEmbedding(nn.Module):
    def __init__(self, context_length: int, embedding_size: int):
        super().__init__()
        self.embedding = nn.Embedding(context_length, embedding_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        return self.embedding(positions)
