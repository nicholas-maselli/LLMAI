import torch

from advanced_agent_llm.config import ModelConfig
from advanced_agent_llm.model.block import TransformerBlock
from advanced_agent_llm.model.embeddings import (
    RotaryPositionEmbedding,
    apply_rotary_position_embedding,
)
from advanced_agent_llm.model.mlp import SwiGLU, calculate_intermediate_size
from advanced_agent_llm.model.normalization import RMSNorm


def test_rms_norm_matches_its_definition() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    normalization = RMSNorm(normalized_shape=4, eps=1e-5)

    actual = normalization(x)
    expected = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)

    torch.testing.assert_close(actual, expected)


def test_rope_preserves_vector_norms() -> None:
    vectors = torch.randn(2, 4, 8, 16)
    rope = RotaryPositionEmbedding(head_size=16)
    positions = torch.arange(8)
    cosine, sine = rope(positions, vectors.dtype)

    rotated = apply_rotary_position_embedding(vectors, cosine, sine)

    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated, dim=-1),
        torch.linalg.vector_norm(vectors, dim=-1),
    )


def test_swiglu_preserves_outer_shape() -> None:
    mlp = SwiGLU(hidden_size=32, intermediate_size=64)
    x = torch.randn(2, 5, 32)

    assert mlp(x).shape == x.shape


def test_transformer_block_preserves_shape_and_backpropagates() -> None:
    config = ModelConfig(
        vocabulary_size=128,
        context_length=16,
        hidden_size=32,
        number_of_layers=2,
        number_of_attention_heads=4,
        number_of_key_value_heads=2,
        feed_forward_multiplier=2.0,
    )
    block = TransformerBlock(config)
    x = torch.randn(2, 8, config.hidden_size, requires_grad=True)

    output = block(x)
    output.sum().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert calculate_intermediate_size(768, 8 / 3) == 2_048

