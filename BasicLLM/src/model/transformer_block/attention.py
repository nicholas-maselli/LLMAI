import torch
import torch.nn as nn


class AttentionProjections(nn.Module):
    def __init__(self, embedding_size: int, head_size: int):
        super().__init__()
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

    def forward(
        self,
        input_vectors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.query(input_vectors)
        keys = self.key(input_vectors)
        values = self.value(input_vectors)
        return queries, keys, values


def calculate_attention_scores(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> torch.Tensor:
    head_size = keys.shape[-1]
    scores = queries @ keys.transpose(-2, -1)
    return scores / head_size**0.5


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    sequence_length = scores.shape[-1]
    mask = torch.tril(
        torch.ones(sequence_length, sequence_length, device=scores.device)
    ).bool()
    return scores.masked_fill(mask == 0, float("-inf"))


def calculate_attention_weights(masked_scores: torch.Tensor) -> torch.Tensor:
    return torch.softmax(masked_scores, dim=-1)


def combine_values(
    attention_weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    return attention_weights @ values


class AttentionHead(nn.Module):
    def __init__(self, embedding_size: int, head_size: int):
        super().__init__()
        self.projections = AttentionProjections(embedding_size, head_size)

    def forward(self, input_vectors: torch.Tensor) -> torch.Tensor:
        queries, keys, values = self.projections(input_vectors)
        scores = calculate_attention_scores(queries, keys)
        masked_scores = apply_causal_mask(scores)
        attention_weights = calculate_attention_weights(masked_scores)
        return combine_values(attention_weights, values)
