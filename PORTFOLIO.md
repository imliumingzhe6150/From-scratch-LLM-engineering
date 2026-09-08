# Portfolio Evidence

This is a working document for future résumé bullets, project descriptions, and
interview preparation. Claims should only be promoted to the public README or a
résumé when they are supported by code, tests, or recorded measurements.

## Project Positioning

Implemented and trained a small decoder-only Transformer from byte-level
tokenization through autoregressive generation, with an emphasis on explicit
tensor operations, controlled experiments, and reproducibility rather than
assembling high-level model components. The next original contributions are
correct, benchmarked KV-cache inference and an evaluated personal-memory
subsystem.

## Evidence-Backed Contributions

### Byte-level BPE tokenizer

- Implemented vocabulary construction, GPT-style regex pre-tokenization,
  frequency-weighted BPE merges, deterministic tie-breaking, and special-token
  boundaries from first principles.
- Replaced full pair recounting with a pair-to-pretoken index and incremental
  updates, reducing a measured 5 MB training run from approximately 5.4 seconds
  to 0.54 seconds.
- Trained 10K and 32K tokenizers on TinyStories and OpenWebText and encoded more
  than 13 GB of source text for later model training.

### Transformer foundations

- Implemented bias-free Linear and Embedding modules without using the matching
  PyTorch high-level layers.
- Implemented RMSNorm with float32 internal statistics for mixed-precision
  stability.
- Implemented the SiLU-gated SwiGLU feed-forward network and verified its
  reference behavior and gradient flow.
- Implemented RoPE with cached trigonometric buffers, explicit adjacent-pair
  rotations, arbitrary token positions, and batch/head broadcasting.
- Implemented numerically stable softmax, scaled dot-product attention, causal
  masking, and parallel multi-head self-attention with optional per-head RoPE.
- Composed pre-norm residual Transformer blocks and a complete language model
  that maps token IDs through embeddings and stacked blocks to vocabulary
  logits.
- Passed all 13 tests in `tests/test_model.py`, including full and truncated
  Transformer-LM snapshots, and separately checked causality, shape handling,
  context limits, mixed precision, and gradient flow.

### Training and experiment infrastructure

- Implemented numerically stable cross-entropy, AdamW, cosine scheduling with
  warmup, global gradient clipping, memory-mapped data sampling, checkpoint
  resume, validation, and throughput/wall-clock logging.
- Built configurable command-line training and generation paths that select
  CUDA, MPS, or CPU and avoid machine-specific artifact paths.
- Completed a controlled five-setting learning-rate screen and retained every
  plotted observation in machine-readable source data.
- Audited the batch-size experiment before publication and identified unequal
  token and validation budgets that prevent a causal batch-size loss claim.
- Built a reproducible three-panel batch-size report that isolates the valid
  fixed-token B1/B8 comparison, preserves the larger runs as a labeled
  fixed-update study, and exports every plotted observation with run metadata.
- Passed the complete current suite with 66 tests passing and two macOS-only
  memory-limit tests skipped.

### TinyStories model result

- Trained a 22,696,448-parameter decoder-only model for 5,000 updates and 40.96
  million tokens on Apple MPS in 4,078.84 seconds.
- Reached a final sampled validation loss of 1.607, corresponding to perplexity
  approximately 4.99, under the recorded low-resource configuration.
- Implemented and tested greedy, temperature-scaled, and exact top-p generation,
  then recorded a three-policy comparison with explicit seeds, stopping
  reasons, token counts, model configuration, and artifact hashes.

## Current Evidence-Backed Résumé Bullets

> Built a decoder-only Transformer architecture and byte-level BPE pipeline
> from first principles in PyTorch, including RoPE, causal multi-head attention,
> pre-norm SwiGLU blocks, and tokenization of 13+ GB of text; optimized BPE pair
> updates from 5.4 seconds to 0.54 seconds on a measured 5 MB debug corpus.

> Implemented a reproducible language-model training stack with custom AdamW,
> cosine warmup, gradient clipping, memory-mapped sampling, validation, and
> checkpoint resume; trained a 22.7M-parameter TinyStories model on 40.96M tokens
> in 68 minutes on Apple MPS and reached 1.607 validation loss.

These bullets may be used now because their numbers are recorded in
`EXPERIMENTS.md`. Add KV-cache and memory-system bullets only after their
roadmap exit criteria are satisfied.

## Interview Topics to Be Able to Explain

- Why byte-level tokenization eliminates out-of-vocabulary inputs.
- Why BPE does not merge across pre-token and special-token boundaries.
- How the pair index changes the complexity of repeated BPE updates.
- Why PyTorch stores Linear weights as `(d_out, d_in)` but computes batched
  row-vector inputs as `x @ weight.T`.
- Why `nn.Parameter` is different from an ordinary Tensor.
- Why RMSNorm computes statistics in float32 during mixed-precision training.
- How SwiGLU uses separate gate and value projections.
- Why token IDs have shape `(batch, sequence)` while embeddings add the
  `d_model` dimension.
- How reshape and transpose split projected features into independent attention
  heads without mixing token and head dimensions.
- How RoPE turns absolute token-position rotations into relative-position
  information in query-key dot products.
- How mask broadcasting, negative-infinity scores, and stable softmax implement
  causal attention.
- Why pre-norm residual blocks normalize before attention and feed-forward
  sublayers, and why the LM head returns logits rather than probabilities.
- Why `sequence_length` is the current input length while `context_length` is
  the configured upper bound.

## Planned Differentiators

- Correct KV-cache inference with token-level equivalence tests and a controlled
  performance benchmark.
- Retrieval-backed personal memory with explicit memory creation, updating,
  conflict resolution, forgetting, and provenance.
- A fixed memory benchmark covering retrieval, conflict resolution, stale-fact
  suppression, provenance, latency, and failure analysis.

## Attribution Boundary

Stanford CS336 supplies the assignment specification, interfaces, fixtures, and
tests. The repository should explicitly credit that source. Portfolio claims
should focus on the implementation, optimizations, experiments, explanations,
and later personal-memory extensions completed by the repository owner.
