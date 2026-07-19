from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocabulary_size: int = 50_304
    context_length: int = 2_048
    hidden_size: int = 768
    number_of_layers: int = 12
    number_of_attention_heads: int = 12
    number_of_key_value_heads: int = 4
    feed_forward_multiplier: float = 8 / 3
    rope_base: float = 10_000.0
    rms_norm_epsilon: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.number_of_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by number_of_attention_heads")
        if self.number_of_attention_heads % self.number_of_key_value_heads != 0:
            raise ValueError(
                "number_of_attention_heads must be divisible by number_of_key_value_heads"
            )


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_subset: str = "sample-10BT"
    dataset_split: str = "train"
    text_column: str = "text"
    streaming: bool = True
    shuffle_buffer_size: int = 10_000
    seed: int = 42

