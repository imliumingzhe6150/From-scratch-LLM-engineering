import pytest
import torch

from cs336_basics.generation import generate_tokens, generate_tokens_with_cache
from cs336_basics.model import KVCache, LayerKVCache, TransformerLM


def make_model(
    *,
    context_length: int = 12,
    dtype: torch.dtype | None = None,
) -> TransformerLM:
    torch.manual_seed(7)
    return TransformerLM(
        vocab_size=31,
        context_length=context_length,
        d_model=16,
        num_layers=2,
        num_heads=4,
        d_ff=32,
        rope_theta=10_000.0,
        dtype=dtype,
    )


def test_prefill_cache_has_explicit_per_layer_shapes_and_size():
    model = make_model()
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    logits, cache = model.forward_with_cache(token_ids)

    assert logits.shape == (2, 3, model.vocab_size)
    assert cache.sequence_length == 3
    assert len(cache.layers) == model.num_layers
    for layer_cache in cache.layers:
        assert layer_cache.key.shape == (2, 4, 3, 4)
        assert layer_cache.value.shape == (2, 4, 3, 4)
        assert layer_cache.key.device == token_ids.device
        assert layer_cache.key.dtype == next(model.parameters()).dtype
    expected_elements = model.num_layers * 2 * 2 * 4 * 3 * 4
    assert cache.size_bytes == expected_elements * cache.layers[0].key.element_size()


def test_cached_chunks_match_full_logits_with_leading_batch_dimensions():
    model = make_model(dtype=torch.float64)
    token_ids = torch.randint(0, model.vocab_size, (2, 3, 7))
    full_logits = model(token_ids)

    first_logits, cache = model.forward_with_cache(token_ids[..., :2])
    second_logits, cache = model.forward_with_cache(
        token_ids[..., 2:5],
        cache=cache,
    )
    final_logits, cache = model.forward_with_cache(
        token_ids[..., 5:],
        cache=cache,
    )
    cached_logits = torch.cat((first_logits, second_logits, final_logits), dim=-2)

    torch.testing.assert_close(cached_logits, full_logits, atol=1e-10, rtol=1e-8)
    assert cache.sequence_length == token_ids.shape[-1]
    assert cache.layers[0].key.shape == (2, 3, 4, 7, 4)
    assert cache.layers[0].key.dtype == torch.float64


def test_cached_decode_matches_each_full_prefix_last_logit():
    model = make_model()
    token_ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2]])

    cached_logits, cache = model.forward_with_cache(token_ids[:, :3])
    torch.testing.assert_close(
        cached_logits[:, -1],
        model(token_ids[:, :3])[:, -1],
    )

    for prefix_length in range(4, token_ids.shape[-1] + 1):
        cached_logits, cache = model.forward_with_cache(
            token_ids[:, prefix_length - 1 : prefix_length],
            cache=cache,
        )
        full_prefix_logits = model(token_ids[:, :prefix_length])
        torch.testing.assert_close(
            cached_logits[:, -1],
            full_prefix_logits[:, -1],
            atol=1e-5,
            rtol=1e-5,
        )


def test_model_cache_rejects_context_overflow_and_wrong_layer_count():
    model = make_model(context_length=4)
    _, full_cache = model.forward_with_cache(torch.tensor([[1, 2, 3, 4]]))

    with pytest.raises(ValueError, match="exceeds context length"):
        model.forward_with_cache(torch.tensor([[5]]), cache=full_cache)

    wrong_layer_count = KVCache(
        layers=full_cache.layers[:1],
        sequence_length=full_cache.sequence_length,
    )
    with pytest.raises(ValueError, match="cache contains 1 layers"):
        model.forward_with_cache(torch.tensor([[5]]), cache=wrong_layer_count)


def test_layer_cache_validates_shapes_and_model_validates_batch_shape():
    with pytest.raises(ValueError, match="identical shapes"):
        LayerKVCache(
            key=torch.zeros(2, 4, 3, 4),
            value=torch.zeros(2, 4, 2, 4),
        )

    model = make_model()
    _, cache = model.forward_with_cache(torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="cache batch shape"):
        model.forward_with_cache(torch.tensor([[3], [4]]), cache=cache)


def test_cached_greedy_generation_matches_uncached_across_context_boundary():
    model = make_model(context_length=4)
    prompt = [1, 2, 3]

    uncached = generate_tokens(
        model,
        prompt,
        max_new_tokens=7,
        temperature=0.0,
    )
    cached = generate_tokens_with_cache(
        model,
        prompt,
        max_new_tokens=7,
        temperature=0.0,
    )

    assert cached == uncached


def test_cached_generation_stops_at_eos_and_restores_training_mode():
    model = make_model()
    for parameter in model.parameters():
        parameter.data.zero_()
    model.train()

    completion = generate_tokens_with_cache(
        model,
        [1, 2],
        max_new_tokens=5,
        temperature=0.0,
        end_token_id=0,
    )

    assert completion == [0]
    assert model.training
