# Experiment Ledger

This file records reproducible measurements and conclusions. Raw datasets and
large generated artifacts stay local and are intentionally not committed.

## BPE Training

| Date | Corpus | Vocabulary | Merges | Result |
| --- | --- | ---: | ---: | --- |
| 2026-08-29 | TinyStories train | 10,000 | 9,743 | Completed in 105.93 seconds; peak memory approximately 3.26 GiB |
| 2026-08-30 | OpenWebText train | 32,000 | 31,743 | Completed and serialized locally |

### TinyStories optimization result

On the 5 MB debugging corpus with a 1,000-token vocabulary, incremental pair
updates reduced observed training time from approximately 5.4 seconds to 0.54
seconds.

### Longest learned tokens

| Tokenizer | Token | UTF-8 byte length | Interpretation |
| --- | --- | ---: | --- |
| TinyStories | `b' accomplishment'` | 15 | A frequent story-domain word including its leading space |
| OpenWebText | repeated mojibake sequence decoding as `ÃÂ...` | 64 | Reflects repeated encoding artifacts present in web text rather than a semantic word |

## Tokenizer Experiments

Method: the first ten documents from each corpus were used by the current
script. Throughput was measured on approximately 10 MB of OpenWebText using
single-process CPU time; results can vary by machine load and hardware.

| Experiment | Bytes | Tokens | Result |
| --- | ---: | ---: | ---: |
| TinyStories sample with TinyStories 10K tokenizer | 7,435 | 1,808 | 4.112 bytes/token |
| OpenWebText sample with OpenWebText 32K tokenizer | 31,487 | 6,712 | 4.691 bytes/token |
| OpenWebText sample with TinyStories 10K tokenizer | 31,487 | 9,873 | 3.189 bytes/token |

The domain-matched OpenWebText tokenizer compresses its sample better than the
TinyStories tokenizer. Applying the smaller, story-domain vocabulary to web
text produces more tokens for the same byte sequence.

### Throughput

| Tokenizer | Measured throughput |
| --- | ---: |
| TinyStories 10K | 1,523,646 bytes/second |
| OpenWebText 32K | 1,436,930 bytes/second |

At the measured OpenWebText rate, encoding 825 GB would take approximately
159.5 hours, or 6.6 days, on the tested machine.

## Encoded Datasets

All four arrays use `uint16`. This type supports IDs through 65,535, covering
both vocabularies while using two bytes per token.

| Dataset | Token count | Payload size | Validation |
| --- | ---: | ---: | --- |
| TinyStories train | 540,796,778 | 1.082 GB | ID range and reconstructed byte count verified |
| TinyStories validation | 5,461,210 | 10.9 MB | ID range and reconstructed byte count verified |
| OpenWebText train | 2,727,120,452 | 5.454 GB | ID range and reconstructed byte count verified |
| OpenWebText validation | 66,401,098 | 132.8 MB | ID range and reconstructed byte count verified |

## Learning-Rate Tuning

On 2026-09-06, the assignment's decaying-SGD toy example was run for 10
iterations from the same seeded initial weights for every learning rate. A
learning rate of 10 produced a steady loss decrease, 100 drove the loss rapidly
to approximately zero, and 1,000 caused the loss to grow rapidly and diverge.

## Model Component Verification

| Date | Component | Test result | Additional check |
| --- | --- | --- | --- |
| 2026-09-03 | Linear | Passed | Arbitrary leading dimensions and parameter count checked |
| 2026-09-03 | Embedding | Passed | Repeated-ID lookup and gradients checked |
| 2026-09-03 | RMSNorm | Passed | float16 dtype restoration, zero-vector stability, and gradients checked |
| 2026-09-03 | SiLU and SwiGLU | Passed | PyTorch parity, output shape, hidden-size alignment, and all projection gradients checked |
| 2026-09-04 | Rotary Position Embeddings | Passed | 4D head broadcasting, position-zero identity, norm preservation, float16 dtype, validation, and gradients checked |
| 2026-09-04 | Numerically stable softmax | Passed | Multiple dimensions, normalization, large-logit stability, float16 dtype, and gradients checked |
| 2026-09-04 | Scaled dot-product attention | Passed | 3D/4D snapshots, causal-mask broadcasting, unmasked behavior, fully masked rows, and gradients checked |
| 2026-09-04 | Causal multi-head self-attention | Passed | Reference snapshots with/without RoPE, causality, implicit/explicit positions, arbitrary leading dimensions, and all projection gradients checked |
| 2026-09-04 | Pre-norm Transformer block | Passed | Reference snapshot, zero-sublayer identity residuals, causality, positions, arbitrary leading dimensions, and all parameter gradients checked |
| 2026-09-04 | Complete Transformer language model | Passed | Full/truncated snapshots, output shapes, causal-prefix equality, arbitrary leading dimensions, context limit, and all parameter gradients checked; complete model suite 13 passed |
| 2026-09-06 | Numerically stable cross-entropy | Passed | PyTorch parity at ordinary and large logits, arbitrary leading batch dimensions, and finite backward gradients checked |
| 2026-09-06 | AdamW | Passed | Assignment's 1,000-step optimizer comparison passed; implementation and adapter passed Ruff |
| 2026-09-06 | Cosine learning-rate schedule with warmup | Passed | All warmup, cosine-annealing, boundary, and post-annealing reference values passed |
| 2026-09-06 | Global L2 gradient clipping | Passed | PyTorch parity, below-threshold no-op behavior, missing-gradient handling, and adapter lint checks passed |
| 2026-09-06 | Language-model data sampling | Passed | Random-start distribution, input/target offset, requested-device handling, uint16-to-long conversion, and lint checks passed |
| 2026-09-06 | Autoregressive text generation | Passed | Temperature sharpness, exact top-p filtering, greedy decoding, EOS stopping, context-window truncation, and mode restoration checked |

## TinyStories Training Experiments

### 2026-09-07: Base-model learning-rate sweep

- Question: How does the peak learning rate affect convergence, and where does
  optimization become unstable for the prescribed TinyStories base model?
- Screening configuration: 22,696,448 total parameters, batch size 32, context
  length 256, 100 steps, 10 warmup steps, cosine decay ending at step 100,
  minimum learning rate 3e-5, seed 42, and one run per setting. All controlled
  hyperparameters were identical across the five runs.

| Peak learning rate | Final train loss | Final validation loss | Observation |
| ---: | ---: | ---: | --- |
| 3e-4 | 4.286 | 4.243 | Stable but slow |
| 1e-3 | 3.689 | 3.661 | Stable and substantially faster |
| 3e-3 | 3.635 | 3.606 | Lowest final validation loss in the sweep |
| 1e-2 | 4.186 | 4.160 | Stable numerically but less effective |
| 3e-2 | 5.006 | 4.972 | Loss spiked to 9.862 near the peak learning rate, then recovered as the schedule decayed |

- Search strategy: Increase the peak learning rate approximately
  logarithmically until crossing from improved convergence into degraded or
  unstable behavior. The 3e-3 setting was selected because 1e-2 performed
  worse, while 3e-2 showed a transient divergence during its high-rate phase.
- Limitation: These are short screening runs with one seed. They identify a
  useful region, not a universal optimum, and the decaying schedule allowed the
  3e-2 run to recover after its loss spike.
- Artifacts: `output/experiments/tinystories_base_lr_*.log`.

### 2026-09-07: 40.96M-token TinyStories base-model run

- Configuration: The prescribed base architecture with a 3e-3 peak learning
  rate, 3e-5 minimum learning rate, 500 warmup steps, cosine decay ending at
  step 5,000, batch size 32, context length 256, and seed 42.
- Tokens processed: 40,960,000, matching the assignment's Apple Silicon
  low-resource budget.
- Wall-clock time: 4,078.84 seconds (67.98 minutes) on MPS.
- Final train loss: 1.634.
- Final validation loss: 1.607, corresponding to a perplexity of approximately
  4.99. This satisfies the low-resource target of at most 2.00.
- Limitation: Validation measurements average 10 randomly sampled batches and
  the experiment contains one seed, so the last decimal places should not be
  interpreted as a precise full-corpus estimate.
- Artifacts: `output/experiments/tinystories_base_lr_3e-3_5000.log`,
  `checkpoints/tinystories_base_lr_3e-3_5000.pt`, and
  `output/experiments/tinystories_learning_rate_experiment*`.

### 2026-09-08: Structured TinyStories generation evaluation

- Question: How do greedy, conservative nucleus sampling, and a
  higher-diversity policy affect one fixed prompt under the selected checkpoint?
- Checkpoint: The 22,696,448-parameter, 5,000-step TinyStories base model with
  validation loss 1.607.
- Prompt: `Once upon a time` (4 tokenizer tokens).
- Seed policy: Reset seed 42 before every policy.
- Runtime environment: CPU with PyTorch 2.11.0.
- Maximum generation: 256 sampled tokens. Counts below include the sampled
  `<|endoftext|>` token, which is excluded from visible text.

| Policy | Temperature | Top-p | Sampled tokens | Stop | Qualitative observation |
| --- | ---: | ---: | ---: | --- | --- |
| Greedy | 0.0 | 1.0 | 159 | EOS | Most coherent story arc and moral, but repetitive wording and ideas |
| Conservative | 0.8 | 0.9 | 216 | EOS | More varied story, with a mild contradiction about whether Tim may play with the ball |
| Diverse | 1.1 | 0.95 | 172 | EOS | More novel events, but weaker grammar and object continuity (`bigDoggy`, cereal/muffin/pie/yogurt drift) |

Representative greedy output:

```text
Once upon a time, there was a little girl named Lily. She loved to play with
her toys and eat yummy food. One day, she found a big, red apple in the
kitchen. She was very happy and wanted to eat it all by herself.

Lily's mom saw her looking at the apple and said, "Lily, you can have a big
apple if you promise to share your food with your friends." Lily thought about
it and decided to share her apple with her friends. They all took turns eating
the apple and having fun.

As they played, Lily's friends came to join them. They all played together and
had a great time. Lily learned that sharing is good and makes everyone happy.
And that was the moral of the story: sharing with others can make everyone
happy.
```

- Interpretation: Lower-entropy decoding produced the strongest global
  coherence in this example. Raising temperature and top-p increased lexical
  and event diversity, but also increased contradictions, malformed words, and
  topic drift. Output quality is also constrained by the 40.96M-token training
  budget and narrow TinyStories domain, not only by the decoding policy.
- Limitation: This is a controlled qualitative comparison using one prompt,
  one seed, and one checkpoint. It illustrates failure modes but does not
  estimate their frequency across the data distribution.
- Reproducibility: The JSON record contains prompt and generated token IDs,
  policy parameters, stopping reasons, model configuration, device and PyTorch
  version, plus SHA-256 identities for the checkpoint, vocabulary, and merges.
- Artifact: `output/experiments/tinystories_generation_evaluation.json`.

### 2026-09-07: Batch-size experiment configuration audit

- Question: How does batch size affect MPS throughput and optimization?
- Shared configuration: Prescribed TinyStories base architecture, context
  length 256, peak learning rate 3e-3, minimum learning rate 3e-5, AdamW,
  seed 42, and five validation measurements per run.

| Batch size | Steps | Training tokens | Final train loss | Final validation loss | Final logged throughput | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12,800 | 3,276,800 | 2.793 | 2.717 | 8,780 tokens/s | 512.44 s |
| 8 | 1,600 | 3,276,800 | 2.658 | 2.589 | 10,732 tokens/s | 327.16 s |
| 32 | 1,600 | 13,107,200 | 1.940 | 1.950 | 10,566 tokens/s | 1,278.14 s |
| 64 | 1,600 | 26,214,400 | 1.728 | 1.722 | 10,505 tokens/s | 2,563.68 s |
| 128 | 1,600 | 52,428,800 | 1.592 | 1.592 | 10,475 tokens/s | 5,120.21 s |

- Systems observation: Batch size 1 had lower throughput than all settings
  from 8 through 128. Batch size 8 produced the highest final logged
  throughput, although the 8--128 differences were small. A separate two-step
  batch-256 smoke test fell to 1,542 tokens/s, indicating severe memory
  pressure; batch 128 is therefore the practical maximum tested on this MPS
  machine.
- Configuration issue: The intended design held training tokens fixed at
  3,276,800, but the batch-32, batch-64, and batch-128 runs retained the
  batch-8 value of 1,600 steps. Their token budgets were therefore 4x, 8x, and
  16x larger than the batch-8 budget. They also retained 16 validation batches,
  so the number of validation tokens per measurement increased with batch
  size.
- Interpretation boundary: The progressively lower final losses for batch
  sizes 32, 64, and 128 cannot be attributed to batch size alone because those
  runs processed progressively more tokens. The logs support a throughput and
  memory-pressure comparison, plus a fixed-update comparison, but not the
  originally intended fixed-token optimization comparison.
- Next decision: Either present the existing results explicitly as a
  fixed-update experiment and discuss the token-budget confound, or rerun batch
  sizes 32, 64, and 128 with 400, 200, and 100 steps respectively and schedules
  scaled to those run lengths before making a fixed-token loss claim.
- Artifacts: `output/experiments/tinystories_batch_{1,8,32,64,128}.log`,
  `output/experiments/tinystories_batch_128_smoke.log`, and
  `output/experiments/tinystories_batch_256_smoke.log`.

### 2026-09-08: Batch-size systems and fixed-budget report

- Figure design: Panel a compares the final logged interval throughput for the
  five complete runs and labels batch 256 as a two-step smoke test. Panel b is
  the only fixed-token loss comparison (batch 1 versus 8): both runs processed
  3,276,800 training tokens and each validation point averages 32,768 tokens.
  Panel c compares batches 8, 32, 64, and 128 after the same 1,600 updates while
  displaying their 3.28M, 13.11M, 26.21M, and 52.43M token budgets.
- Result: Batch 8 improved final logged throughput from 8,780 to 10,732
  tokens/s relative to batch 1 (22.2%) and finished the matched-token run with
  validation loss 2.589 rather than 2.717. Throughput from batch 8 through 128
  stayed within 2.5% of the batch-8 value. These are single-seed observations,
  not estimates with uncertainty intervals.
- Interpretation: The fixed-token B1/B8 comparison supports the practical
  choice of batch 8 on this MPS machine. The lower losses for batches 32, 64,
  and 128 are shown only as outcomes after equal update counts and increasing
  data exposure; they do not establish a pure batch-size benefit.
- Metric definition: Throughput is the `tokens_per_second` value from the final
  logged training interval. Validation cross-entropy is the mean over the
  configured sampled validation batches. One run was available per setting;
  seed 42; no error bars or smoothing.
- Reproducibility: The loader checks shared architecture, optimizer, dataset,
  device, and seed fields; exact B1/B8 training and validation-token equality;
  and equal update counts for B8/B32/B64/B128 before plotting.
- Public-figure subset: The README figure includes all five completed
  throughput runs and all ten validation observations from the matched-token
  B1/B8 runs. It excludes the B256 two-step smoke result and the twenty
  fixed-update validation observations because neither belongs in the public
  fixed-token claim; both remain in the full report and source table.
- Artifacts: `scripts/plot_batch_size_experiment.py`,
  `output/experiments/tinystories_batch_size_experiment.{svg,pdf,tiff,png}`,
  `docs/assets/tinystories_batch_size_overview.{svg,png}`, and
  `results/tinystories_batch_size_source_data.csv`.

### 2026-09-07: 200-step TinyStories baseline

- Question: After 200 training steps, what level of language structure can the
  model learn?
- Prediction: The loss would begin to stabilize; training loss would be
  slightly lower than validation loss without a large overfitting gap; generated
  text would show local sentence structure but not a complete story.
- Configuration: 2,937,472 parameters, batch size 8, context length 64, 200
  steps, maximum learning rate 3e-4, minimum learning rate 3e-5, 20 warmup
  steps, and cosine decay ending at step 200.
- Tokens processed: 102,400.
- Wall-clock time: 3.41 seconds on MPS.
- Initial validation loss: 9.067 at step 20.
- Final validation loss: 5.885 at step 200, corresponding to a perplexity of
  approximately 360.
- Generation observation: The completion contained recognizable words, basic
  grammatical fragments, names, and punctuation, but it did not maintain a
  coherent story line across the complete sample.
- Conclusion: Two hundred steps were sufficient for the model to begin learning
  local language structure, but not sufficient to train a model capable of
  producing a complete, coherent story.
- Limitations: A single generated sample is not enough to characterize overall
  generation quality. In addition, the learning rate fell from approximately
  2.70e-4 at step 60 to 3.00e-5 at step 200. Therefore, the slower loss decrease
  late in training cannot yet be attributed solely to model saturation; the
  cosine learning-rate decay is a confounding factor.
- Artifact: `output/experiments/tinystories_baseline_200.log`. Its intermediate
  checkpoint was pruned after the 5,000-step model became the project anchor.

## Template for Future Runs

```markdown
### YYYY-MM-DD: Experiment name

- Question:
- Code/commit:
- Data:
- Configuration:
- Hardware:
- Command:
- Metrics:
- Conclusion:
- Limitations:
- Artifact location:
```
