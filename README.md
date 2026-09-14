# From-Scratch Language Model

This project builds a small decoder-only language model from byte-level
tokenization through Transformer training and inference.

The current implementation includes a byte-level BPE tokenizer trained on
TinyStories and OpenWebText, dataset encoding, and a complete decoder-only
Transformer architecture built from bias-free projections, embeddings,
RMSNorm, RoPE, causal multi-head attention, and SwiGLU blocks.

The selected TinyStories model has 22,696,448 parameters and reached a
validation loss of 1.607 after training on 40.96 million tokens on Apple MPS.
The project now includes structured generation evaluation and benchmarked
KV-cache inference.

## TinyStories Batch-Size Result

![Batch-size throughput and matched-token validation loss](docs/assets/tinystories_batch_size_overview.png)

Batch 8 delivered the highest final logged throughput on Apple MPS: 10,732
tokens/s, 22.2% above batch 1. In the only matched-token comparison, both runs
processed 3.28M training tokens and batch 8 reached validation loss 2.589 versus
2.717 for batch 1. These are single-seed observations. The larger-batch loss
runs used unequal token budgets and are therefore kept in the full experiment
report rather than presented here as a causal batch-size comparison.

[Source data](./results/tinystories_batch_size_source_data.csv) ·
[Full interpretation](./EXPERIMENTS.md#2026-09-08-batch-size-systems-and-fixed-budget-report)

## KV-Cache Inference Result

![Cached and uncached generation throughput](docs/assets/tinystories_kv_cache_benchmark.png)

On CPU with the selected 22.7M-parameter checkpoint, cached generation reached
636, 582, and 522 tokens/s for the P16/G32, P64/G64, and P128/G128 cases. This
was 2.27x, 3.94x, and 6.28x the corresponding uncached throughput. Each value is
the median of five measured repeats after two warmups; model loading is excluded
and cached/uncached greedy tokens are checked for exact agreement. These are
single-machine measurements, not universal speedups.

[Source data](./results/tinystories_kv_cache_benchmark_summary.csv) ·
[Environment and checkpoint identity](./results/tinystories_kv_cache_benchmark_metadata.json) ·
[Engineering note](./docs/kv_cache.md)

## Repository Guide

| Path                              | Purpose                                                         |
| --------------------------------- | --------------------------------------------------------------- |
| `cs336_basics/tokenizer.py`     | BPE training and tokenizer implementation                       |
| `cs336_basics/model.py`         | Transformer building blocks                                     |
| `cs336_basics/data.py`          | Language-model batch sampling                                   |
| `cs336_basics/serialization.py` | Training checkpoint save/load utilities                         |
| `cs336_basics/training.py`      | Validation and end-to-end training loop                         |
| `cs336_basics/generation.py`    | Temperature-scaled and top-p decoding                           |
| `scripts/benchmark_kv_cache.py` | Cached/uncached inference benchmark                             |
| `scripts/`                      | Tokenizer training, evaluation, and dataset encoding            |
| `tests/adapters.py`             | Connection between local implementations and assignment tests   |
| `docs/devlog/`                  | Chronological development notes                                 |
| `docs/decisions/`               | Architecture decision records                                   |
| `docs/assets/`                  | Version-controlled figures used by the public README            |
| `results/`                      | Small, version-controlled source-data tables for public figures |
| `PORTFOLIO.md`                  | Evidence and interview material                                 |
| `ROADMAP.md`                    | Resume-oriented milestones and exit criteria                    |

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

The assignment tests call local implementations through
[tests/adapters.py](./tests/adapters.py). During development, run a narrow test
such as `uv run pytest -k test_rmsnorm`, then run the full suite at milestone
boundaries. The current suite passes completely except for two macOS-only
memory-limit tests that are skipped on this platform.

### Download data

Download the TinyStories data and a subsample of OpenWebText

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

Raw datasets, encoded arrays, checkpoints, and experiment-service caches are
local artifacts and are excluded from version control.

### Train a model

The training script reads the one-dimensional `.npy` arrays produced by
`scripts/encode_datasets.py` through `np.memmap`, so a multi-gigabyte corpus
does not need to fit in RAM. Model, optimizer, schedule, reporting, and
checkpoint intervals are configurable from the command line.

For example, this starts a TinyStories run and selects CUDA, MPS, or CPU
automatically:

```sh
uv run scripts/train.py \
  --train-data output/encoded/tinystories_train_ids.npy \
  --validation-data output/encoded/tinystories_valid_ids.npy \
  --checkpoint-path checkpoints/tinystories_latest.pt \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --batch-size 32 \
  --max-iterations 10000
```

Use `--help` to see every hyperparameter. To continue an interrupted run, add
`--resume-from checkpoints/tinystories_latest.pt`; `--max-iterations` is the
total target number of updates, not the number of additional updates.

### Generate text

Generation accepts a user prompt, stops at `<|endoftext|>` or the requested
token limit, and supports temperature and nucleus (top-p) sampling. The model
architecture arguments must match those used to create the checkpoint.

```sh
uv run scripts/generate.py \
  --checkpoint-path checkpoints/tinystories_latest.pt \
  --vocab-path output/tokenizer/tinystories_vocab.pkl \
  --merges-path output/tokenizer/tinystories_merges.pkl \
  --prompt "Once upon a time" \
  --max-new-tokens 200 \
  --temperature 0.8 \
  --top-p 0.9
```

Set `--temperature 0` for deterministic greedy decoding. When the prompt grows
beyond the configured context length, generation retains the most recent
context window. Generation uses the KV cache by default; pass `--uncached` to
run the full-prefix baseline.

### Benchmark KV-cache inference

The benchmark excludes model loading, performs device synchronization and
warmup, verifies greedy-token equivalence, saves every timed repeat, and reports
median prefill, decode, and end-to-end throughput.

```sh
uv run scripts/benchmark_kv_cache.py \
  --device cpu \
  --checkpoint-path checkpoints/tinystories_base_lr_3e-3_5000.pt \
  --case 16:32 --case 64:64 --case 128:128 \
  --warmup-repeats 2 --measured-repeats 5
```

### Evaluate decoding policies

The evaluation command resets the same seed for each named policy and writes a
JSON record containing token IDs and counts, stopping reasons, complete text,
model and environment configuration, and SHA-256 identities for every input
artifact. With no `--policy` arguments it compares greedy, conservative, and
higher-diversity defaults.

```sh
uv run scripts/evaluate_generation.py \
  --checkpoint-path checkpoints/tinystories_base_lr_3e-3_5000.pt \
  --vocab-path output/tokenizer/tinystories_vocab.pkl \
  --merges-path output/tokenizer/tinystories_merges.pkl \
  --prompt "Once upon a time" \
  --max-new-tokens 256 \
  --seed 42
```

The recorded comparison produced coherent complete stories under all three
policies. Greedy decoding was the most globally consistent in this sample;
raising temperature and top-p increased variety but also contradictions and
topic drift. See `EXPERIMENTS.md` for the outputs and interpretation.

### Plot the batch-size study

This command separates the valid fixed-token comparison from the confounded
fixed-update runs and exports SVG, PDF, TIFF, PNG, and long-form CSV artifacts.

```sh
uv run scripts/plot_batch_size_experiment.py \
  --full-logs output/experiments/tinystories_batch_{1,8,32,64,128}.log \
  --smoke-log output/experiments/tinystories_batch_256_smoke.log \
  --output-prefix output/experiments/tinystories_batch_size_experiment \
  --readme-prefix docs/assets/tinystories_batch_size_overview \
  --source-data-output results/tinystories_batch_size_source_data.csv
```

The GitHub-facing image contains only completed throughput runs and the valid
B1/B8 matched-token loss comparison. The detailed report retains batches 32,
64, and 128 as a labeled fixed-1,600-update view whose token budgets differ.

## Attribution

This repository is based on Stanford CS336 Assignment 1. The assignment
specification, starter interfaces, tests, fixtures, and handout originate from
the course materials. The implementations, optimization work, experiment
records, explanatory documentation, and planned personal-memory extensions are
developed in this project.

For the original assignment description, see
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf).
