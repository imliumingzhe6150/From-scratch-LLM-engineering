# Resume-Ready Project Roadmap

Decision date: 2026-09-08

## North Star

Turn the current CS336-based implementation into a portfolio project that can
support precise resume claims and a substantive technical interview. The final
story is not "completed every assignment question." It is:

> Built and verified a decoder-only language model from first principles,
> measured its training behavior on TinyStories, added correct and benchmarked
> stateful inference, and used the resulting engineering foundation to build an
> evaluated personal-memory subsystem.

The project should favor correctness, reproducibility, and explainability over
large compute expenditure or a long list of loosely supported features.

## Scope Boundary

### In scope

- Finish the TinyStories baseline as a self-contained, evidence-backed case
  study.
- Implement KV-cache inference with correctness tests and a controlled
  benchmark.
- Build a local, retrieval-backed personal-memory subsystem with explicit
  lifecycle rules and quantitative evaluation.
- Make the repository portable, testable, and easy to understand from its
  README, figures, and demo commands.
- Maintain a clear attribution boundary between Stanford-provided assignment
  materials and original implementation, experiments, and extensions.

### Not on the critical path

- Completing every remaining CS336 Assignment 1 question.
- Full-scale RMSNorm, pre-norm/post-norm, NoPE, or SwiGLU ablation training.
- Training a language model to fluency on OpenWebText.
- Leaderboard optimization or dependence on rented GPU time.
- Claims based on screenshots, isolated samples, or unrecorded runs.

Small CPU-friendly ablations may be added later if they answer a concrete
question, but they must not delay the four milestones below.

## Operating Constraints

- The existing TinyStories checkpoint is the training anchor; no new large
  training run is required for the main roadmap.
- Every critical-path task must run on CPU or Apple Silicon. Access to CUDA may
  improve benchmark coverage but must not be required for correctness.
- Raw datasets, encoded arrays, checkpoints, caches, and service metadata stay
  out of version control.
- Measured claims must record configuration, hardware, seed policy, sample
  size, raw data location, and limitations.
- New abstractions should keep tensor shapes and mathematical intent explicit
  enough to explain in an interview.

## Milestone 1: Close the TinyStories Baseline

Status: completed on 2026-09-08. Structured generation evaluation, the
three-policy checkpoint run, honest batch-size reporting, the concise public
figure, versioned source data, repository cleanup, full tests, and full Ruff
checks are complete.

### Goal

Turn the already-trained model and existing experiment logs into a complete,
honest case study before adding new features.

### Work

1. Add a repeatable generation-evaluation command that loads the selected
   5,000-step checkpoint and records the prompt, seed, generated token count,
   stopping reason, temperature, top-p, and generated text.
2. Evaluate at least three decoding policies: greedy, a conservative sampling
   setting, and a higher-diversity setting. Keep the prompt and seed policy
   explicit.
3. Preserve the existing batch-size runs as a systems and fixed-update study.
   Do not rerun the expensive configurations merely to satisfy the assignment.
4. Create a batch-size figure and source-data table that expose training-token
   budgets, wall-clock time, and throughput. The figure and prose must not
   attribute lower loss to batch size when token budgets differ.
5. Resolve the known repository-wide Ruff findings and rerun the full tests.
6. Update the public README and portfolio evidence with the finalized training
   and generation results.

### Deliverables

- Reproducible generation samples and a decoding comparison in
  `EXPERIMENTS.md`.
- Batch-size comparison figure plus machine-readable source data.
- Updated README project summary and quick-start commands.
- Clean full-suite test and Ruff results.

### Exit criteria

- At least one recorded completion contains 256 generated tokens or documents
  that generation stopped at EOS.
- Decoding commentary identifies at least two quality factors without treating
  one sample as a general evaluation.
- Batch-size conclusions distinguish optimizer updates, processed tokens,
  throughput, and memory pressure.
- Every public numerical claim points to a recorded artifact.

### Compute profile

Low. This milestone uses existing logs and checkpoints; only inference,
plotting, tests, and documentation are required.

## Milestone 2: Correct and Measurable KV-Cache Inference

### Goal

Add a resume-visible inference optimization whose correctness and speedup are
both demonstrated rather than assumed.

### Work

1. Define an explicit per-layer KV-cache interface. Document shapes as
   `(..., heads, cached_sequence, head_dimension)` and make position handling
   explicit.
2. Separate prompt prefill from single-token decode while reusing the existing
   attention, RoPE, block, and language-model modules where practical.
3. Define and test the context-limit policy. A simple correct policy is better
   than silent cache growth or invalid RoPE positions.
4. Test cached logits against full-prefix logits at every decode step, including
   multi-token prompts, batch dimensions, EOS, dtype/device behavior, and the
   context boundary.
5. Require greedy cached and uncached generation to produce identical tokens.
6. Build a benchmark that excludes model/tokenizer loading, performs warmup,
   synchronizes asynchronous devices, counts actual generated tokens, repeats
   measurements, and reports a robust summary such as the median.
7. Measure prefill latency, decode latency or tokens per second, and cache
   memory over multiple prompt and generation lengths.

### Deliverables

- Reusable cache-aware model and generation APIs.
- Dedicated KV-cache correctness tests.
- A benchmark script, raw benchmark table, and comparison figure.
- An engineering note explaining complexity, memory cost, RoPE offsets, and
  the chosen context-limit behavior.

### Exit criteria

- Cached and uncached logits agree within a documented numerical tolerance.
- Greedy outputs match token for token on the fixed test cases.
- The benchmark can be rerun with one command and records device and software
  configuration.
- Any reported speedup is measured on this implementation; the roadmap does
  not require a predetermined speedup value.

### Compute profile

Low. No model training is required. CPU measurements are valid; MPS or CUDA
measurements are optional additions.

## Milestone 3: Evaluated Personal Memory

### Goal

Build the original differentiator of the repository: a small, inspectable
memory layer that can create, retrieve, revise, supersede, and forget personal
facts with provenance.

### Architecture

- Store memories locally in SQLite so persistence and queries require no
  external service.
- Represent both readable text and structured identity fields such as user,
  subject, predicate, value, timestamps, source, confidence, status, and the
  memory superseded by an update.
- Begin with deterministic lexical retrieval, such as BM25 plus optional
  recency and confidence features. Embedding retrieval is a later comparison,
  not a prerequisite.
- Separate retrieval from response generation behind a small interface. The
  TinyStories model is not an instruction-tuned assistant, so do not claim that
  it produces high-quality personalized answers without direct evidence.

### Work

1. Specify memory lifecycle semantics for create, update, conflict,
   supersession, and forgetting.
2. Implement the persistent store and retrieval ranking with explainable score
   components.
3. Return provenance with every retrieved memory and exclude forgotten or
   superseded facts by default.
4. Treat retrieved text as untrusted context when constructing prompts; keep
   memory content distinct from system or application instructions.
5. Create a fixed synthetic benchmark containing relevant, irrelevant,
   conflicting, updated, and forgotten memories.
6. Report retrieval Recall@k, mean reciprocal rank, conflict-resolution
   accuracy, stale-memory suppression, latency, and representative failures.
7. Add a CLI demo that shows memory creation, an update, retrieval with
   provenance, and forgetting.

### Deliverables

- Tested memory schema, store, lifecycle operations, and retriever.
- Versioned evaluation fixture and one-command evaluation script.
- Metrics table and failure analysis in `EXPERIMENTS.md`.
- A short end-to-end CLI demonstration.

### Exit criteria

- Lifecycle rules are deterministic and covered by tests.
- Evaluation data and metric definitions are fixed and reviewable rather than
  reconstructed from successful examples.
- Retrieval results expose why a memory was selected and where it came from.
- Documentation clearly separates measured retrieval quality from any
  unmeasured downstream answer quality.

### Compute profile

Low. SQLite, lexical retrieval, unit tests, and synthetic evaluation are all
CPU-friendly and work offline.

## Milestone 4: Portfolio and Repository Hardening

### Goal

Make the work understandable and reproducible to a reviewer who spends one
minute on the README and ten minutes running the project.

### Work

1. Rewrite the README around the project story, not assignment chronology.
2. Add a compact architecture/data-flow diagram covering tokenization,
   training, cached inference, and memory retrieval.
3. Provide quick-start commands for tests, generation, the KV benchmark, the
   memory evaluation, and the demo.
4. Remove machine-specific paths and check that a fresh configuration fails
   with actionable messages when local artifacts are absent.
5. Add continuous integration for unit tests and Ruff once the repository
   structure is stable.
6. Keep small source-data tables and representative outputs in version control;
   keep large reproducible artifacts ignored.
7. Finalize resume bullets and interview notes only from verified evidence.

### Exit criteria

- A new reader can identify the problem, architecture, results, limitations,
  and original contributions from the README.
- All documented commands are exercised in the supported local environment.
- Tests and Ruff pass in CI and locally.
- At least four evidence-backed results are available: tokenizer optimization,
  TinyStories training quality, KV-cache performance, and memory evaluation.
- The repository contains no secrets, personal service metadata, raw corpora,
  or machine-specific absolute paths.

### Compute profile

Low. This milestone is packaging, validation, and documentation work.

## Resume-Ready Definition of Done

The project is ready to place on a resume when all of the following are true:

- The end-to-end baseline and every original extension have automated tests.
- The README contains reproducible commands and measured results rather than
  unsupported adjectives.
- Training, inference, and memory metrics each include configuration and
  limitations.
- A two-minute demo works without retraining the model.
- The owner can explain BPE, causal attention, RoPE, pre-norm residuals,
  optimization, KV caching, retrieval ranking, conflict handling, and the main
  experimental confounds.
- Resume bullets in `PORTFOLIO.md` cite only outcomes already present in tests
  or `EXPERIMENTS.md`.

## Deferred Backlog

These tasks are valid future work but do not block the resume-ready release:

- Small-model, multi-seed normalization or feed-forward ablations on CPU.
- Embedding or hybrid retrieval compared against the lexical baseline.
- Weight-only quantization and CPU inference benchmarking.
- OpenWebText language-model training when appropriate compute becomes
  available.
- A richer chat-model backend for testing whether retrieved memories improve
  final answer quality.

## Immediate Next Task

Begin Milestone 2 by defining the per-layer KV-cache contract: cache shapes,
prefill versus decode behavior, RoPE position offsets, batch semantics, and the
context-limit policy. Implement correctness tests before benchmarking speed.
