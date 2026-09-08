import torch
from torch import nn

from cs336_basics.generation import (
    generate_tokens,
    sample_next_token,
    temperature_scaled_probabilities,
    top_p_probabilities,
)


class ScriptedLanguageModel(nn.Module):
    """Return one predetermined greedy token on each decoding call."""

    def __init__(self, next_tokens: list[int], vocab_size: int, context_length: int):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.next_tokens = next_tokens
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.calls: list[list[int]] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(token_ids[0].tolist())
        next_token = self.next_tokens[len(self.calls) - 1]
        logits = torch.zeros(
            *token_ids.shape,
            self.vocab_size,
            device=token_ids.device,
        )
        logits[:, -1, next_token] = 10.0
        return logits


def test_temperature_scaling_changes_distribution_sharpness():
    logits = torch.tensor([0.0, 1.0])

    cold = temperature_scaled_probabilities(logits, temperature=0.5)
    hot = temperature_scaled_probabilities(logits, temperature=2.0)

    assert cold[1] > hot[1]
    torch.testing.assert_close(cold.sum(), torch.tensor(1.0))
    torch.testing.assert_close(hot.sum(), torch.tensor(1.0))


def test_top_p_keeps_threshold_crossing_token_and_renormalizes():
    probabilities = torch.tensor([0.5, 0.3, 0.2])

    filtered = top_p_probabilities(probabilities, top_p=0.6)

    torch.testing.assert_close(filtered, torch.tensor([0.625, 0.375, 0.0]))


def test_sample_next_token_supports_greedy_decoding_and_top_p():
    logits = torch.tensor([3.0, 2.0, 1.0])

    greedy = sample_next_token(logits, temperature=0.0)
    nucleus_sample = sample_next_token(
        logits,
        top_p=0.5,
        generator=torch.Generator().manual_seed(7),
    )

    assert greedy.item() == 0
    assert nucleus_sample.item() == 0


def test_generate_stops_at_eos_and_restores_training_mode():
    model = ScriptedLanguageModel(
        next_tokens=[2, 4, 3],
        vocab_size=5,
        context_length=3,
    )
    model.train()

    completion = generate_tokens(
        model,
        [0, 1],
        max_new_tokens=10,
        temperature=0.0,
        end_token_id=4,
    )

    assert completion == [2, 4]
    assert model.calls == [[0, 1], [0, 1, 2]]
    assert model.training


def test_generate_uses_sliding_context_window():
    model = ScriptedLanguageModel(
        next_tokens=[3, 4],
        vocab_size=5,
        context_length=3,
    )

    completion = generate_tokens(
        model,
        [0, 1, 2],
        max_new_tokens=2,
        temperature=0.0,
    )

    assert completion == [3, 4]
    assert model.calls == [[0, 1, 2], [1, 2, 3]]
