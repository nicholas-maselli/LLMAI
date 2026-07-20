import torch

from advanced_llm.model.normalization import RMSNorm


def test_rms_norm_matches_its_definition() -> None:
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    normalization = RMSNorm(normalized_shape=4, eps=1e-5)

    actual = normalization(hidden_states)
    expected = hidden_states / torch.sqrt(hidden_states.pow(2).mean(dim=-1, keepdim=True) + 1e-5)

    torch.testing.assert_close(actual, expected)
