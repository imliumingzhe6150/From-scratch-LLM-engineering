# 002: Use SwiGLU for the position-wise feed-forward network

- Status: Accepted
- Date: 2026-09-03

## Context

Each Transformer block needs a nonlinear transformation that processes every
token independently while preserving the model dimension at its interface.

## Decision

Use three bias-free projections with the computation:

```text
gate = SiLU(W1 x)
value = W3 x
output = W2(gate * value)
```

W1 and W3 expand from `d_model` to `d_ff`; W2 projects back to `d_model`. When
not explicitly configured, choose `d_ff` near `8/3 * d_model` and align it to a
multiple of 64.

## Rationale

- The gate gives the network a learned, input-dependent way to control hidden
  features.
- SiLU is smooth and retains small negative activations.
- A multiple-of-64 hidden dimension is suitable for efficient accelerator
  matrix operations.
- Reusing the custom `Linear` module centralizes initialization and shape logic.

## Consequences

- SwiGLU uses three projection matrices rather than the two in a conventional
  activation-only feed-forward network.
- It does not exchange information between tokens; attention must provide that
  interaction.

## Evidence

The implementation matches the provided reference snapshot and all three
projection parameters receive gradients in an additional backward-pass check.
