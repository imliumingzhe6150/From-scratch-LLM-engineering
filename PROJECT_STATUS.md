# Project Status

Last updated: 2026-09-08

## Objective

Build, measure, and explain a small language model from first principles, then
extend it with reliable cached inference and an evaluated personal-memory
subsystem. The target is a resume-ready engineering project, not completion of
every remaining CS336 assignment experiment.

## Current Milestone

The end-to-end language-model path is complete through autoregressive text
generation. The TinyStories base model completed its low-resource
learning-rate study and reached a validation loss of 1.607 after 40.96M tokens.
The project has now adopted the resume-oriented roadmap in `ROADMAP.md`.
Milestone 1 is complete: the TinyStories case study now includes reproducible
generation evaluation, honest batch-size reporting, a concise public figure,
machine-readable source data, and clean repository-wide lint. The next active
milestone is complete as well: KV-cache inference now has explicit APIs,
correctness tests, a controlled CPU benchmark, source data, and a public figure.
The next active milestone is an evaluated local personal-memory subsystem.

## Scope Decision

- Full-scale RMSNorm, pre-norm/post-norm, NoPE, and SwiGLU ablations are not on
  the critical path.
- OpenWebText language-model training and leaderboard optimization are deferred
  until suitable compute is available and a concrete project question
  justifies the cost.
- KV-cache inference and evaluated personal memory are the two original
  extensions selected for the portfolio project.
- Every resume claim must be traceable to tests or a recorded measurement.

## Completed

### Tokenization

- Byte-level BPE vocabulary initialization, pre-tokenization, merge learning,
  deterministic tie-breaking, and special-token boundaries.
- Multiprocess pre-tokenization and incremental pair-count updates.
- `Tokenizer` encoding, decoding, special-token handling, and iterable encoding.
- TinyStories tokenizer: 10,000 vocabulary entries and 9,743 merges.
- OpenWebText tokenizer: 32,000 vocabulary entries and 31,743 merges.
- TinyStories and OpenWebText training/validation corpora encoded as `uint16`
  NumPy arrays for later language-model training.

### Transformer Components

- `Linear`: bias-free learned projection with assignment-specific initialization.
- `Embedding`: token-ID lookup table with truncated-normal initialization.
- `RMSNorm`: float32 RMS statistics, learned gain, and dtype restoration.
- `SiLU`: explicit `x * sigmoid(x)` implementation.
- `SwiGLU`: gated position-wise feed-forward network using W1, W2, and W3.
- `RotaryPositionalEmbedding`: cached pairwise rotations for arbitrary token
  positions and leading batch/head dimensions.
- `softmax`: numerically stable probability normalization along an arbitrary
  tensor dimension.
- `scaled_dot_product_attention`: masked Q/K similarity, stable probability
  normalization, and weighted value aggregation across arbitrary batch/head
  dimensions.
- `CausalMultiHeadSelfAttention`: parallel Q/K/V projection and head processing,
  causal masking, optional per-head RoPE, head merging, and output projection.
- `TransformerBlock`: pre-norm causal attention and SwiGLU sublayers with two
  sequential residual connections.
- `TransformerLM`: token embeddings, a configurable block stack, final RMSNorm,
  and vocabulary logits projection.

### Training Utilities

- Numerically stable cross-entropy from logits, with arbitrary leading batch
  dimensions and mean reduction over all examples.
- AdamW with per-parameter moment state, bias correction, and decoupled weight
  decay.
- Linear-warmup and cosine-annealing learning-rate schedule with a fixed
  post-annealing minimum.
- In-place global L2 gradient clipping with missing-gradient handling and the
  assignment's numerical-stability epsilon.
- Random next-token batch sampling from CPU-backed NumPy arrays, with `long`
  conversion and transfer of only the sampled batch to the requested device.
- Model and optimizer checkpoint save/load utilities, including training
  iteration restoration and support for paths or binary file-like objects.
- Configurable end-to-end training with memory-mapped `.npy` corpora, scheduled
  learning rates, gradient clipping, periodic step/wall-clock console metrics,
  validation, checkpointing, and checkpoint resume across devices.
- Reusable experiment-log plotting with complete train/validation data parsing,
  step-based and wall-clock loss panels, and SVG/PDF/TIFF/PNG export.
- Learning-rate sweep reporting with controlled-configuration checks, complete
  source-data export, and a combined screening/full-run figure.
- Batch-size runs for 1, 8, 32, 64, and 128, plus memory-limit smoke tests at
  128 and 256. The raw logs are complete; the experiment audit identified
  unequal token budgets for batch sizes 32 through 128.
- Three-panel batch-size report separating final MPS throughput, the valid
  fixed-token B1/B8 comparison, and the fixed-update B8/B32/B64/B128 view with
  its 16x token-budget confound; every plotted value is exported to CSV.
- GitHub-facing two-panel batch-size figure with larger text and only the five
  completed throughput runs plus the valid fixed-token B1/B8 loss
  comparison. The compact PNG, editable SVG, and source CSV are versioned.

### Text Generation

- Temperature-scaled next-token probabilities, including deterministic greedy
  decoding at temperature zero.
- Exact nucleus sampling that keeps the smallest top-probability set reaching
  the requested cumulative mass and then renormalizes it.
- Autoregressive completion with EOS stopping, a maximum generated-token limit,
  context-window truncation, inference mode, and restoration of model state.
- Command-line tokenizer/checkpoint loading and generation from a user prompt.
- Structured generation evaluation with named decoding policies, fixed seed
  semantics, explicit stopping reasons and token counts, artifact SHA-256
  identities, model/environment metadata, and atomic JSON output.
- Three-policy evaluation of the selected TinyStories checkpoint; all greedy,
  conservative, and diverse samples reached EOS within the 256-token limit.

### KV-Cache Inference

- Immutable per-layer key/value cache records with explicit
  `(..., heads, cached_sequence, head_dimension)` shapes and byte accounting.
- Cached prompt prefill and multi-token or single-token decode through the
  existing attention, Transformer block, and language-model hierarchy.
- RoPE offsets derived from cached sequence length, with rotated keys retained
  so old positions are not recomputed.
- Hard model-level context bounds and an exact generation-level sliding-window
  policy that rebuilds the cache when the oldest token is evicted.
- Cached/full-prefix logit equivalence and greedy-token equivalence tests,
  including arbitrary leading batch dimensions, float64, EOS, invalid caches,
  and the context boundary.
- Reproducible cached/uncached benchmark with warmup, synchronization, repeated
  timings, output-equivalence checks, cache memory, raw local measurements,
  median source data, environment metadata, and checkpoint identity.
- On CPU, the selected TinyStories checkpoint measured 2.27x, 3.94x, and 6.28x
  median end-to-end throughput improvements for P16/G32, P64/G64, and
  P128/G128. The longest case used 3.98 MiB of cache storage.

## Verification Status

| Area | Result |
| --- | --- |
| BPE training tests | 3 passed |
| Tokenizer tests | 23 passed, 2 macOS-only memory tests skipped |
| Linear | Passed |
| Embedding | Passed |
| RMSNorm | Passed |
| SiLU and SwiGLU | Passed |
| Rotary Position Embeddings (RoPE) | Passed |
| Numerically stable softmax | Passed |
| Scaled dot-product attention | Passed |
| Causal multi-head self-attention, with and without RoPE | Passed |
| Pre-norm Transformer block | Passed |
| Complete Transformer language model | Passed |
| Complete `tests/test_model.py` suite | 13 passed |
| Numerically stable cross-entropy | Passed |
| AdamW | Passed |
| Cosine learning-rate schedule with warmup | Passed |
| Global L2 gradient clipping | Passed |
| Language-model data sampling | Passed |
| Model and optimizer checkpointing | Passed |
| End-to-end training loop | Passed, including memmap loading and checkpoint resume |
| Experiment loss-curve parsing and panel construction | Passed |
| Complete test suite | 76 passed, 2 macOS-only memory tests skipped |
| Text generation | Passed, including temperature, top-p, EOS, and context-window checks |
| Structured generation evaluation | Passed, including real-checkpoint CLI smoke test |
| KV-cache inference | 7 correctness tests passed, including logit/token equivalence and context boundary |
| KV-cache benchmark | 3 benchmark tests passed; real-checkpoint CPU run completed with 5 repeats per case |
| Learning-rate experiment figure | Passed static preflight: 14 checks, 0 warnings, 0 failures |
| Batch-size experiment figures | 4 plotting tests passed; static preflight: 14 checks, 0 warnings, 0 failures |
| Repository-wide Ruff | Passed |

## Next Steps

1. Define the personal-memory data model, provenance fields, lifecycle rules,
   retrieval boundary, and evaluation cases before implementation.
2. Implement local memory creation, retrieval, revision, supersession, and
   forgetting with fixed tests for stale-fact suppression and provenance.

## Known Issues and Technical Debt

- TinyStories line-by-line iterable encoding is lossless but can produce a
  slightly different token sequence from encoding an entire document at once.
  This does not block the learning project or initial model training.
- `scripts/encode_datasets.py` collects NumPy chunks before concatenation; a
  future version should use a true incremental writer or memory map.
- Tokenizer experiment scripts use the first ten documents rather than a seeded
  random sample, so the reported compression numbers are reproducible but not
  necessarily representative of the full corpus.
- The original 2026-08-30 daily summary predates the completed OpenWebText run
  and should be treated as historical rather than current project status.
- The first KV-cache implementation extends tensors with `torch.cat`; a
  preallocated cache could reduce copy overhead without changing the API.

## Key Paths

| Purpose | Path |
| --- | --- |
| Model components | `cs336_basics/model.py` |
| Data sampling | `cs336_basics/data.py` |
| Checkpoint serialization | `cs336_basics/serialization.py` |
| Training loop | `cs336_basics/training.py` and `scripts/train.py` |
| Experiment plotting | `scripts/plot_training_log.py`, `scripts/plot_learning_rate_experiment.py`, and `scripts/plot_batch_size_experiment.py` |
| Public figures | `docs/assets/` |
| Versioned result tables | `results/` |
| Text generation | `cs336_basics/generation.py` and `scripts/generate.py` |
| KV-cache design and benchmark | `docs/kv_cache.md`, `docs/decisions/003-kv-cache-contract.md`, and `scripts/benchmark_kv_cache.py` |
| Generation evaluation | `cs336_basics/generation_evaluation.py` and `scripts/evaluate_generation.py` |
| Tokenizer implementation | `cs336_basics/tokenizer.py` |
| Test integration | `tests/adapters.py` |
| Tokenizer experiments | `scripts/tokenizer_experiments.py` |
| Dataset encoding | `scripts/encode_datasets.py` |
| Experiment record | `EXPERIMENTS.md` |
| Resume-oriented roadmap | `ROADMAP.md` |
| Development history | `docs/devlog/` |
| Design decisions | `docs/decisions/` |
