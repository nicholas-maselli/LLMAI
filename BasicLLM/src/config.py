from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "tiny_language_model"
    context_length: int = 32
    embedding_size: int = 64
    number_of_heads: int = 4
    batch_size: int = 16
    training_steps: int = 500
    learning_rate: float = 0.003
    generation_length: int = 200
    logging_interval: int = 200
    checkpoint_interval: int = 200
