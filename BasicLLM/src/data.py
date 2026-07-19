from pathlib import Path

import torch

from tokenizer import CharacterVocabulary


def load_dataset() -> str:
    dataset_path = Path(__file__).parent.parent / "datasets" / "dataset.txt"
    return dataset_path.read_text(encoding="utf-8")


def tokenize_dataset() -> tuple[CharacterVocabulary, list[int]]:
    text = load_dataset()
    tokenizer = CharacterVocabulary(text)
    token_ids = tokenizer.encode(text)
    return tokenizer, token_ids


def create_tensor(token_ids: list[int]) -> torch.Tensor:
    return torch.tensor(token_ids, dtype=torch.long)


def create_training_example(
    data: torch.Tensor,
    start: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = data[start : start + context_length]
    targets = data[start + 1 : start + context_length + 1]
    return inputs, targets


def create_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0,
        len(data) - context_length,
        (batch_size,),
    )

    examples = [
        create_training_example(data, start.item(), context_length)
        for start in starts
    ]

    inputs = torch.stack([example[0] for example in examples])
    targets = torch.stack([example[1] for example in examples])
    return inputs, targets
