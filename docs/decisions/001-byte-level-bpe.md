# 001: Use a byte-level BPE tokenizer

- Status: Accepted
- Date: 2026-08-30

## Context

The language model needs a finite vocabulary that can represent arbitrary
Unicode input while producing shorter sequences than raw byte tokenization.

## Decision

Use UTF-8 bytes as the base 256-token vocabulary and learn additional subword
tokens with BPE. Apply GPT-style regex pre-tokenization and treat
`<|endoftext|>` as a hard document boundary and a single special token.

## Rationale

- Every input string can be represented without an unknown token.
- Learned byte sequences improve compression over one-token-per-byte encoding.
- Pre-tokenization prevents merges from freely crossing unrelated word and
  punctuation boundaries.
- A dedicated end-of-text token makes document boundaries available to the
  language model.

## Consequences

- Unicode characters may span multiple base tokens before suitable merges are
  learned.
- Tokenizer behavior depends on both the learned merges and the precise
  pre-tokenization boundaries.
- Domain-specific vocabularies compress their training domains differently.

## Evidence

The TinyStories and OpenWebText tokenizers passed the provided training and
encoding tests. Recorded compression results are maintained in
`EXPERIMENTS.md`.
