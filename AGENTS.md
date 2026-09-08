# AI Collaboration Guide

This repository is a learning-first implementation of a small language model,
starting from byte-level BPE and progressing through Transformer training and
inference. Code should remain understandable enough for the owner to explain in
an interview and reuse in later personal-memory and agent projects.

## Read First

At the beginning of a development session, inspect these files in order:

1. `PROJECT_STATUS.md` for the current milestone and next steps.
2. `README.md` for the public project story and supported commands.
3. `EXPERIMENTS.md` for measured results and experiment assumptions.
4. The latest file in `docs/devlog/` for recent implementation context.
5. `git status --short` before editing, because the working tree may contain
   user-owned changes.

## Development Principles

- Optimize for understanding first: keep tensor shapes and mathematical intent
  explicit in names, docstrings, and comments.
- Implement assignment-restricted components from first principles. Do not
  replace them with PyTorch equivalents that the handout prohibits.
- Reuse small modules such as `Linear` instead of duplicating parameter setup,
  initialization, and matrix multiplication.
- Preserve arbitrary leading batch dimensions unless an interface explicitly
  requires a fixed rank.
- Do not modify or remove unrelated user changes.
- Do not commit raw datasets, encoded arrays, checkpoints, or experiment-service
  caches. These are local/reproducible artifacts.

## Verification

Prefer the narrowest relevant test while iterating:

```bash
.venv/bin/pytest -q -k test_linear
.venv/bin/pytest -q -k test_embedding
.venv/bin/pytest -q -k test_rmsnorm
.venv/bin/pytest -q -k 'test_silu or test_swiglu'
```

Before declaring a larger milestone complete, run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check cs336_basics tests/adapters.py
```

## Documentation Workflow

After a meaningful implementation session:

1. Update the current snapshot in `PROJECT_STATUS.md`.
2. Add a dated entry under `docs/devlog/` describing changes and verification.
3. Record measured runs in `EXPERIMENTS.md`; do not record guessed metrics.
4. Update `PORTFOLIO.md` only when there is concrete evidence for a new claim.
5. Keep `README.md` stable and public-facing rather than turning it into a diary.

## Attribution

The assignment specification, fixtures, and test harness originate from
Stanford CS336 Assignment 1. Clearly separate those materials from the
implementation, optimizations, experiments, and later extensions authored in
this repository.
