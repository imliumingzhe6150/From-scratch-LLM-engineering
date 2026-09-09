"""Autoregressive decoding utilities for trained language models."""

from __future__ import annotations

import torch
from torch import Tensor

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax


def _validate_generation_request(
    model: TransformerLM,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    end_token_id: int | None,
) -> None:
    """Validate arguments shared by cached and uncached generation."""

    if not prompt_tokens:
        raise ValueError("prompt_tokens must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if end_token_id is not None and not 0 <= end_token_id < model.vocab_size:
        raise ValueError(f"end_token_id must be in [0, {model.vocab_size}), got {end_token_id}")
    if any(token_id < 0 or token_id >= model.vocab_size for token_id in prompt_tokens):
        raise ValueError("prompt_tokens contains an ID outside the model vocabulary")


def temperature_scaled_probabilities(logits: Tensor, temperature: float) -> Tensor:
    """Convert logits to probabilities after temperature scaling.

    A temperature below one sharpens the distribution, while a value above one
    makes it flatter. Temperature zero is handled by :func:`sample_next_token`
    as deterministic greedy decoding and is therefore not valid here.
    """

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    return softmax(logits / temperature, dim=-1)


def top_p_probabilities(probabilities: Tensor, top_p: float) -> Tensor:
    """Keep the smallest set of most-likely tokens whose mass reaches ``top_p``.

    The retained probabilities are renormalized along the final dimension.
    Every leading dimension is treated as a separate probability distribution.
    """

    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if probabilities.ndim == 0 or probabilities.shape[-1] == 0:
        raise ValueError(
            "probabilities must have a non-empty vocabulary dimension"
        )
    if not torch.all(torch.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if torch.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        dim=-1,
        descending=True,
    )
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    # Include a token exactly when the mass before it is still below top_p.
    # This retains the first token that takes the cumulative mass across the
    # threshold, matching the smallest-set definition of nucleus sampling.
    mass_before_token = cumulative_probabilities - sorted_probabilities
    keep = mass_before_token < top_p
    retained_sorted = torch.where(
        keep,
        sorted_probabilities,
        torch.zeros_like(sorted_probabilities),
    )

    retained = torch.zeros_like(probabilities)
    retained.scatter_(dim=-1, index=sorted_indices, src=retained_sorted)
    normalizer = retained.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise ValueError("each probability distribution must have positive mass")
    return retained / normalizer


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token IDs from next-token logits along the final dimension.

    ``temperature=0`` selects the highest-logit token deterministically. For a
    positive temperature, temperature scaling is applied before top-p filtering
    and multinomial sampling.
    """

    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if logits.ndim == 0 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    probabilities = temperature_scaled_probabilities(logits, temperature)
    probabilities = top_p_probabilities(probabilities, top_p)
    vocabulary_size = probabilities.shape[-1]
    flat_probabilities = probabilities.reshape(-1, vocabulary_size)
    sampled = torch.multinomial(
        flat_probabilities,
        num_samples=1,
        generator=generator,
    )
    return sampled.reshape(probabilities.shape[:-1])


@torch.inference_mode()
def generate_tokens(
    model: TransformerLM,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate completion token IDs for one prompt.

    Only the generated continuation is returned. Once the complete sequence is
    longer than the model context, the most recent ``context_length`` tokens are
    used as the next model input.
    """

    _validate_generation_request(
        model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        end_token_id=end_token_id,
    )

    device = next(model.parameters()).device
    all_tokens = list(prompt_tokens)
    completion: list[int] = []
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            model_input = torch.tensor(
                all_tokens[-model.context_length :],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            next_token_logits = model(model_input)[0, -1]
            next_token = int(
                sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                ).item()
            )
            all_tokens.append(next_token)
            completion.append(next_token)

            if end_token_id is not None and next_token == end_token_id:
                break
    finally:
        model.train(was_training)

    return completion


@torch.inference_mode()
def generate_tokens_with_cache(
    model: TransformerLM,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate one completion using prompt prefill and cached decoding.

    The initial visible prompt is evaluated once. Later calls process only the
    sampled token until the context is full. At that boundary the most recent
    context window is prefetched again, exactly matching :func:`generate_tokens`
    rather than silently changing its sliding-window semantics.
    """

    _validate_generation_request(
        model,
        prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        end_token_id=end_token_id,
    )
    if max_new_tokens == 0:
        return []

    device = next(model.parameters()).device
    all_tokens = list(prompt_tokens)
    completion: list[int] = []
    was_training = model.training
    model.eval()

    try:
        model_input = torch.tensor(
            all_tokens[-model.context_length :],
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        logits, cache = model.forward_with_cache(model_input)

        for decode_step in range(max_new_tokens):
            next_token = int(
                sample_next_token(
                    logits[0, -1],
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                ).item()
            )
            all_tokens.append(next_token)
            completion.append(next_token)

            if end_token_id is not None and next_token == end_token_id:
                break
            if decode_step + 1 == max_new_tokens:
                break

            if cache.sequence_length < model.context_length:
                model_input = torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=device,
                )
                logits, cache = model.forward_with_cache(model_input, cache=cache)
            else:
                # Cached states cannot exactly represent the established
                # shifted-window computation after the oldest token is evicted.
                # Rebuild the full visible window and restart RoPE positions at 0.
                model_input = torch.tensor(
                    all_tokens[-model.context_length :],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                logits, cache = model.forward_with_cache(model_input)
    finally:
        model.train(was_training)

    return completion
