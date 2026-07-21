from dataclasses import dataclass
from pathlib import Path


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
    tokenizer_path: Path = Path("artifacts/tokenizer.json")
    local_text_path: Path | None = None
    validation_fraction: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str = "advanced-agent-llm"
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 10_000
    learning_rate: float = 3e-4
    minimum_learning_rate_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    log_every: int = 10
    validate_every: int = 500
    validation_batches: int = 20
    checkpoint_every: int = 500
    seed: int = 42
    precision: str = "bf16"
    number_of_workers: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch sizes and accumulation steps must be positive")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.log_every < 1 or self.checkpoint_every < 1:
            raise ValueError("logging and checkpoint intervals must be positive")
        if self.validation_batches < 0:
            raise ValueError("validation_batches cannot be negative")
        if self.validation_batches > 0 and self.validate_every < 1:
            raise ValueError("validate_every must be positive when validation is enabled")
        if not 0.0 <= self.minimum_learning_rate_ratio <= 1.0:
            raise ValueError("minimum_learning_rate_ratio must be between 0 and 1")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
