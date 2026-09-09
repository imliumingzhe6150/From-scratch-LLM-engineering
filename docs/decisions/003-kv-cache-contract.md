# 003: Cache rotated keys and values with explicit position offsets

- Status: Accepted
- Date: 2026-09-08

## Context

Autoregressive generation currently recomputes every Transformer layer over the
complete visible prefix before sampling each token. A KV cache can reuse the
keys and values already computed for earlier tokens, but only if cache shapes,
RoPE positions, and context-limit behavior are unambiguous.

## Decision

Represent one layer's cache as a pair of tensors with shape
`(..., heads, cached_sequence, head_dimension)`. Store keys after RoPE has been
applied and values after the value projection. A model-level cache contains one
layer cache per Transformer block plus the shared cached sequence length.

Cached model calls accept one or more new tokens. If the cache contains `P`
tokens, the new tokens use RoPE positions `P, P + 1, ...`, attend to all cached
tokens, and use a causal mask within the new chunk. The model rejects any call
for which `P + new_sequence` exceeds the configured context length.

Cached generation preserves the existing sliding-window semantics. When the
next sampled token would extend a full cache, generation rebuilds the cache
from the most recent `context_length` tokens and resets their RoPE positions to
start at zero. This deliberately favors exact agreement with the existing
uncached implementation over retaining stale hidden states after eviction.

## Rationale

- Rotated keys never need to be rotated again during decoding.
- Explicit sequence length makes position offsets available even for a model
  with zero Transformer layers.
- Supporting multi-token chunks gives the same API to prompt prefill and
  single-token decoding.
- A hard model-level context bound prevents silent cache growth and invalid
  RoPE indices.
- Rebuilding at the boundary exactly matches the established behavior in
  which each shifted context window is evaluated from scratch.

## Consequences

- Before the context boundary, decoding projects and processes only new tokens
  while attention still reads all cached keys and values.
- Cache memory grows linearly with layers, batch size, heads, cached sequence,
  and head dimension.
- Crossing the context boundary incurs a full-window recomputation. A future
  rolling-cache policy would have different semantics and would need separate
  correctness tests.
- Repeated tensor concatenation is simple and inspectable, but a preallocated
  cache may reduce memory-copy overhead in a future optimization.

## Evidence

Dedicated tests compare cached logits with full-prefix logits, cached and
uncached greedy tokens, cache shapes, batch dimensions, dtype/device placement,
and context-boundary behavior.
