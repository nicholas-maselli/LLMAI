import torch
import torch.nn as nn

from advanced_agent_llm.config import ModelConfig
from advanced_agent_llm.model.block import TransformerBlock
from advanced_agent_llm.model.normalization import RMSNorm


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocabulary_size,
            config.hidden_size,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.number_of_layers)
        )
        self.final_norm = RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_epsilon,
        )
        self.language_model_head = nn.Linear(
            config.hidden_size,
            config.vocabulary_size,
            bias=False,
        )

        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.language_model_head.weight = self.token_embeddings.weight

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence length exceeds the configured context length")

        x = self.token_embeddings(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.language_model_head(x)

    def parameter_count(self, count_tied_weights_once: bool = True) -> int:
        if count_tied_weights_once:
            return sum(parameter.numel() for parameter in self.parameters())
        return sum(
            parameter.numel()
            for module in self.modules()
            for parameter in module.parameters(recurse=False)
        )
