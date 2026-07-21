import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from advanced_agent_llm.config import DataConfig
from advanced_agent_llm.data.tokenizer import Tokenizer


def iterate_documents(config: DataConfig, shuffle: bool = False) -> Iterator[str]:
    if config.local_text_path is not None:
        with Path(config.local_text_path).open("r", encoding="utf-8") as text_file:
            for line in text_file:
                document = line.strip()
                if document:
                    yield document
        return

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "FineWeb streaming requires the 'datasets' package. Run `uv sync`."
        ) from error

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_subset,
        split=config.dataset_split,
        streaming=config.streaming,
    )
    if shuffle:
        dataset = dataset.shuffle(
            seed=config.seed,
            buffer_size=config.shuffle_buffer_size,
        )
    for example in dataset:
        document = example[config.text_column]
        if isinstance(document, str) and document.strip():
            yield document


def _is_validation_document(document: str, validation_fraction: float) -> bool:
    digest = hashlib.blake2b(document.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") / 2**64
    return bucket < validation_fraction


class PackedPretrainingDataset(IterableDataset):
    def __init__(
        self,
        config: DataConfig,
        tokenizer: Tokenizer,
        context_length: int,
        split: str,
    ):
        super().__init__()
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        self.config = config
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.split = split

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        number_of_workers = 1 if worker is None else worker.num_workers
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        shard_id = rank * number_of_workers + worker_id
        number_of_shards = world_size * number_of_workers

        token_buffer: list[int] = []
        chunk_size = self.context_length + 1
        documents = iterate_documents(
            self.config,
            shuffle=self.split == "train",
        )

        for document_index, document in enumerate(documents):
            is_validation = _is_validation_document(
                document,
                self.config.validation_fraction,
            )
            if is_validation != (self.split == "validation"):
                continue
            if document_index % number_of_shards != shard_id:
                continue

            token_buffer.extend(self.tokenizer.encode(document))
            token_buffer.append(self.tokenizer.eos_token_id)

            while len(token_buffer) >= chunk_size:
                chunk = token_buffer[:chunk_size]
                del token_buffer[:chunk_size]
                tokens = torch.tensor(chunk, dtype=torch.long)
                yield tokens[:-1], tokens[1:]
