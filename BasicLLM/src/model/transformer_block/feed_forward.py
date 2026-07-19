import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        hidden_size = 4 * embedding_size

        self.network = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        return self.network(input_vectors)
