from collections.abc import Iterable, Iterator
from pathlib import Path


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]


class Tokenizer:
    def __init__(self, tokenizer_path: str | Path):
        try:
            from tokenizers import Tokenizer as HuggingFaceTokenizer
        except ImportError as error:
            raise ImportError(
                "Tokenizer support requires the 'tokenizers' package. Run `uv sync`."
            ) from error

        self.path = Path(tokenizer_path)
        self._tokenizer = HuggingFaceTokenizer.from_file(str(self.path))

        self.pad_token_id = self._required_token_id("<|pad|>")
        self.bos_token_id = self._required_token_id("<|bos|>")
        self.eos_token_id = self._required_token_id("<|eos|>")
        self.unk_token_id = self._required_token_id("<|unk|>")

    def _required_token_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token {token!r}")
        return token_id

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def train_bpe_tokenizer(
    documents: Iterable[str],
    output_path: str | Path,
    vocabulary_size: int,
    minimum_frequency: int = 2,
) -> Path:
    try:
        from tokenizers import Tokenizer as HuggingFaceTokenizer
        from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers
    except ImportError as error:
        raise ImportError(
            "Tokenizer training requires the 'tokenizers' package. Run `uv sync`."
        ) from error

    tokenizer = HuggingFaceTokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=minimum_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    def nonempty_documents() -> Iterator[str]:
        for document in documents:
            if document.strip():
                yield document

    tokenizer.train_from_iterator(nonempty_documents(), trainer=trainer)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))
    return destination
