import torch
import torch.nn.functional as F


def next_token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets must agree on batch and sequence dimensions")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
