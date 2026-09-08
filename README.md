# From-Scratch Language Model

This project builds a small decoder-only language model from byte-level
tokenization through Transformer training and inference. The primary goal is to
understand and explain the underlying computations, then reuse that foundation
in a personal-memory agent.

The current implementation includes a byte-level BPE tokenizer trained on
TinyStories and OpenWebText, dataset encoding, and a complete decoder-only
Transformer architecture built from bias-free projections, embeddings,
RMSNorm, RoPE, causal multi-head attention, and SwiGLU blocks.

The selected TinyStories model has 22,696,448 parameters and reached a
validation loss of 1.607 after training on 40.96 million tokens on Apple MPS.
The active portfolio roadmap now focuses on rigorous generation evaluation,
KV-cache inference, and an evaluated local personal-memory subsystem rather
than completing every remaining assignment experiment.

## Project Status

- Tokenizer milestone: complete for model development.
- Transformer architecture: complete and verified by the model test suite.
- Numerically stable cross-entropy: complete and verified.
- AdamW: complete and verified.
- Gradient clipping: complete and verified.
- Data sampling: complete and verified.
- Checkpointing: complete and verified.
- Configurable training loop: complete and verified.
- Temperature-scaled and top-p text generation: complete and verified.
- Structured three-policy generation evaluation: complete and recorded.
- Batch-size systems/fixed-budget figure: complete with source-data export.
- Low-resource TinyStories training target: reached with validation loss 1.607.
- KV-cache inference: next extension.
- Personal-memory subsystem: planned after cached inference.

See [PROJECT_STATUS.md](./PROJECT_STATUS.md) for the live project snapshot and
[EXPERIMENTS.md](./EXPERIMENTS.md) for measured results. The milestone plan,
scope boundary, and resume-ready exit criteria are in
[ROADMAP.md](./ROADMAP.md).

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

## Repository Guide

| Path | Purpose |
| --- | --- |
| `cs336_basics/tokenizer.py` | BPE training and tokenizer implementation |
| `cs336_basics/model.py` | Transformer building blocks |
| `cs336_basics/data.py` | Language-model batch sampling |
| `cs336_basics/serialization.py` | Training checkpoint save/load utilities |
| `cs336_basics/training.py` | Validation and end-to-end training loop |
| `cs336_basics/generation.py` | Temperature-scaled and top-p decoding |
| `scripts/` | Tokenizer training, evaluation, and dataset encoding |
| `tests/adapters.py` | Connection between local implementations and assignment tests |
| `docs/devlog/` | Chronological development notes |
| `docs/decisions/` | Architecture decision records |
| `docs/assets/` | Version-controlled figures used by the public README |
| `results/` | Small, version-controlled source-data tables for public figures |
| `PORTFOLIO.md` | Evidence and interview material |
| `ROADMAP.md` | Resume-oriented milestones and exit criteria |

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

``` sh
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
context window.

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
