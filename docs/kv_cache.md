# KV-Cache Inference

The KV cache removes repeated key/value projection and repeated hidden-state computation for tokens that the model has already processed. The implementation keeps the uncached `forward` path intact and adds explicit cached methods at the attention, block, and language-model levels.

## Interface and shapes

`TransformerLM.forward_with_cache(token_ids, cache=None)` accepts either a complete prompt or one or more new tokens. It returns logits only for those input tokens plus an updated `KVCache`.

For every layer:

```text
key:   (..., num_heads, cached_sequence, d_head)
value: (..., num_heads, cached_sequence, d_head)
```

The leading dimensions are the input batch dimensions. Keys are cached after RoPE; values are cached after the value projection. `KVCache.sequence_length` is shared by all layers, and `KVCache.size_bytes` reports the tensor storage.

## Prefill and decode

During prefill, the model processes all visible prompt tokens and constructs the first cache. If the cache contains `P` tokens, a later cached call assigns new tokens the RoPE positions `P, P + 1, ...`. New queries attend to the cached keys plus the new keys. A causal mask is still required when a cached call adds multiple tokens, because earlier tokens in that new chunk must not see later ones.

For a single-token decode call, all `P + 1` keys are visible to the new query. The returned key and value tensors are extended along their sequence dimension.

## Context-limit policy

The model API rejects `cached_length + input_length > context_length`. This prevents silent cache growth and RoPE lookup beyond the configured positions.

The generation API preserves the project's earlier sliding-window behavior.
When a sampled token would need to extend a full cache, it rebuilds the cache from the latest `context_length` tokens and restarts positions at zero. Keeping the old keys while dropping only their first entry would not be equivalent: retained hidden states would still contain information from the evicted token, and their RoPE coordinates would differ from a freshly evaluated window.

## Compute and memory

For a current sequence length `L`, uncached decoding repeats projections and feed-forward work for all `L` tokens and recomputes an `L x L` attention map.
A cached single-token step projects and transforms only the new token, then computes its attention against `L` stored positions. The cache changes repeated token work from a full-prefix computation to one new-token computation, while attention still grows linearly with the visible history for each decode step.

For self-attention with `num_heads * d_head = d_model`, float storage is:

```text
2 * batch * num_layers * cached_sequence * d_model * bytes_per_element
```

The factor two accounts for keys and values. In the measured float32 base model at batch 1 and 255 cached positions, this is 4,177,920 bytes (3.98 MiB).

The current implementation extends tensors with `torch.cat` because the data flow is easy to inspect. A preallocated cache could reduce copying overhead in a later optimization without changing this public contract.

## Correctness contract

Dedicated tests require:

- cached chunk logits to match full-sequence logits;
- last-token logits to match at every incremental decode step;
- arbitrary leading batch dimensions, dtype, device, and cache shapes to be
  preserved;
- invalid cache shapes, layer counts, and context overflow to fail explicitly;
- cached and uncached greedy generation to match token for token, including
  after the context window shifts;
- EOS stopping and the model's original train/eval mode to be preserved.

## Benchmark

Run the selected TinyStories checkpoint benchmark with:

```sh
uv run scripts/benchmark_kv_cache.py \
  --device cpu \
  --checkpoint-path checkpoints/tinystories_base_lr_3e-3_5000.pt \
  --case 16:32 --case 64:64 --case 128:128 \
  --warmup-repeats 2 --measured-repeats 5
```

Model and checkpoint loading are outside the timed region. The script
synchronizes asynchronous devices around each phase, generates a fixed number
of greedy tokens, checks cached/uncached token equality, records every repeat,
and reports medians. The measured results and limitations are in
`EXPERIMENTS.md`.
