"""Neural-network building blocks used by the Transformer language model.

The assignment implements these layers from first principles so that their
parameters, tensor shapes, and computations remain explicit.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cs336_basics.nn_utils import softmax


@dataclass(frozen=True)
class LayerKVCache:
    """Projected keys and values retained by one attention layer.

    Both tensors have shape
    ``(..., num_heads, cached_sequence, d_head)``. Keys are stored after RoPE
    has been applied, so only newly projected keys need rotation during decode.
    """

    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if self.key.ndim < 3:
            raise ValueError("cached key and value must have at least head, sequence, and feature dimensions")
        if self.key.shape != self.value.shape:
            raise ValueError(
                "cached key and value must have identical shapes, got "
                f"{tuple(self.key.shape)} and {tuple(self.value.shape)}"
            )
        if self.key.device != self.value.device:
            raise ValueError("cached key and value must be on the same device")
        if self.key.dtype != self.value.dtype:
            raise ValueError("cached key and value must have the same dtype")

    @property
    def sequence_length(self) -> int:
        """Number of token positions stored in this layer."""

        return self.key.shape[-2]

    @property
    def size_bytes(self) -> int:
        """Storage occupied by this layer's key and value tensors."""

        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


@dataclass(frozen=True)
class KVCache:
    """KV-cache state shared across all layers of a Transformer language model."""

    layers: tuple[LayerKVCache, ...]
    sequence_length: int

    def __post_init__(self) -> None:
        if self.sequence_length < 0:
            raise ValueError(f"cache sequence_length must be non-negative, got {self.sequence_length}")
        for layer_index, layer_cache in enumerate(self.layers):
            if layer_cache.sequence_length != self.sequence_length:
                raise ValueError(
                    f"layer {layer_index} caches {layer_cache.sequence_length} "
                    f"positions, expected {self.sequence_length}"
                )

    @property
    def size_bytes(self) -> int:
        """Storage occupied by every cached key and value tensor."""

        return sum(layer.size_bytes for layer in self.layers)


class Linear(nn.Module):
    """Apply a learned linear transformation without a bias term.

    The weight is stored as ``(out_features, in_features)``. For an input whose
    final dimension is ``in_features``, the output is mathematically ``y = Wx``
    and has final dimension ``out_features``. In PyTorch, batched inputs are
    represented as row vectors, so ``forward`` computes ``x @ W.T``.

    Args:
        in_features: Size of the final dimension of the input.
        out_features: Size of the final dimension of the output.
        device: Device on which to allocate the weight parameter.
        dtype: Data type of the weight parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Each row contains the weights used to produce one output feature.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Xavier-style initialization from the assignment. The normal
        # distribution has variance 2 / (d_in + d_out) and is truncated at
        # three standard deviations to avoid unusually large initial weights.
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform ``(..., in_features)`` into ``(..., out_features)``."""

        # weight.T has shape (in_features, out_features), allowing matmul to
        # preserve every leading batch dimension of x automatically.
        return x @ self.weight.T


class Embedding(nn.Module):
    """Map integer token IDs to learned embedding vectors.

    The embedding table has shape ``(num_embeddings, embedding_dim)``. Each row
    stores the vector for one token ID, so looking up an ID is simply indexing
    the corresponding row of the table.

    Args:
        num_embeddings: Number of entries in the vocabulary.
        embedding_dim: Number of features in each embedding vector.
        device: Device on which to allocate the embedding table.
        dtype: Data type of the embedding table.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Row i is the learnable representation associated with token ID i.
        # Registering the table as nn.Parameter makes it visible to optimizers,
        # state_dict(), device transfers, and automatic differentiation.
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # The assignment initializes embedding values from a standard normal
        # distribution, truncated to exclude values outside [-3, 3].
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Look up vectors for token IDs with any number of batch dimensions.

        If ``token_ids`` has shape ``(...)``, the returned tensor has shape
        ``(..., embedding_dim)``.
        """

        # Advanced indexing gathers one complete row for every token ID while
        # preserving the input tensor's shape as the output's leading shape.
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Normalize each token vector by its root-mean-square magnitude.

    Unlike LayerNorm, RMSNorm does not subtract the feature mean. For a token
    vector ``x`` with ``d_model`` features, it computes::

        rms = sqrt(mean(x**2) + eps)
        output = (x / rms) * weight

    ``weight`` is a learned gain vector with one value per model feature. All
    leading dimensions are treated as batch-like dimensions; normalization is
    performed independently along the final dimension only.

    Args:
        d_model: Number of features in each token representation.
        eps: Small positive value that prevents division by zero.
        device: Device on which to allocate the gain parameter.
        dtype: Data type of the gain parameter.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Starting every gain at 1 makes the initial RMSNorm perform pure
        # normalization. Training can then learn a separate scale per feature.
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``(..., d_model)`` and return a tensor of the same shape."""

        input_dtype = x.dtype

        # Squaring low-precision float16/bfloat16 values can overflow or lose
        # accuracy. RMS statistics are therefore always computed in float32.
        x_float = x.to(torch.float32)
        mean_square = x_float.pow(2).mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + self.eps)

        # Broadcasting applies the d_model-length gain vector to the final
        # dimension of every token, regardless of the leading batch shape.
        normalized = x_float * inverse_rms
        output = normalized * self.weight.to(torch.float32)

        # Preserve the module interface expected by mixed-precision training.
        return output.to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotate query/key feature pairs according to their token positions.

    RoPE groups the final feature dimension into adjacent pairs. For pair
    ``i`` at token position ``t``, the rotation angle is::

        angle(t, i) = t * theta ** (-2 * i / d_k)

    Applying the same position-dependent rotations to queries and keys makes
    their dot product depend on relative position. The cosine and sine values
    are fixed (not learned), so they are stored as module buffers rather than
    parameters.

    Args:
        theta: Base used to space the rotation frequencies.
        d_k: Size of the query/key feature dimension. It must be even because
            every two adjacent features form one 2D rotation pair.
        max_seq_len: Number of token positions to precompute.
        device: Device on which to allocate the cached cosine and sine values.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"d_k must be a positive even integer, got {d_k}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive integer, got {max_seq_len}"
            )

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Pair i contains features (2i, 2i + 1). Thus arange(0, d_k, 2)
        # supplies the 2i term in theta^(-2i / d_k).
        feature_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-feature_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]

        # Buffers follow the module across devices but are omitted from the
        # state dict because they can always be reconstructed from the config.
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """Apply RoPE to ``(..., sequence_length, d_k)`` input features.

        ``token_positions`` may omit input dimensions such as an attention-head
        dimension. Missing dimensions are inserted immediately before the
        sequence dimension so the selected rotations broadcast over them.
        """

        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"expected input's final dimension to be {self.d_k}, "
                f"got {x.shape[-1]}"
            )

        positions = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        if positions.numel() > 0 and (
            positions.min().item() < 0
            or positions.max().item() >= self.max_seq_len
        ):
            raise IndexError(
                "token positions must be between 0 and "
                f"{self.max_seq_len - 1}, inclusive"
            )

        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]

        # A (batch, sequence) position tensor must also broadcast over an
        # input shaped (batch, heads, sequence, d_k). Insert any missing axes
        # immediately before the sequence and paired-feature axes.
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        even_features = x_float[..., 0::2]
        odd_features = x_float[..., 1::2]

        rotated_even = even_features * cos - odd_features * sin
        rotated_odd = even_features * sin + odd_features * cos

        # Stack each pair as (..., d_k / 2, 2), then flatten only the final
        # two dimensions to restore the original adjacent feature ordering.
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
        return rotated.to(input_dtype)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary leading dimensions.

    Shapes:
        - ``query``: ``(..., queries, d_k)``
        - ``key``: ``(..., keys, d_k)``
        - ``value``: ``(..., keys, d_v)``
        - ``mask``: broadcastable to ``(..., queries, keys)``; ``True`` entries
          are visible and ``False`` entries are excluded.
        - output: ``(..., queries, d_v)``
    """

    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key must have the same feature dimension, got "
            f"{query.shape[-1]} and {key.shape[-1]}"
        )
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            "key and value must contain the same number of positions, got "
            f"{key.shape[-2]} and {value.shape[-2]}"
        )

    # Each entry is the similarity between one query and one key. Scaling by
    # sqrt(d_k) prevents the dot products from growing with the head dimension.
    scores = query @ key.transpose(-2, -1)
    scores = scores / math.sqrt(query.shape[-1])

    if mask is not None:
        visible = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~visible, -torch.inf)

    # Normalize across keys: every query gets one distribution over the key
    # positions, which is then used to take a weighted sum of their values.
    attention_weights = softmax(scores, dim=-1)
    if mask is not None:
        # This also turns an entirely masked row from NaNs into zero weights,
        # making its weighted value output a well-defined zero vector.
        attention_weights = attention_weights.masked_fill(~visible, 0.0)
    return attention_weights @ value


class CausalMultiHeadSelfAttention(nn.Module):
    """Apply causal self-attention in parallel across multiple heads.

    Each Q/K/V projection produces all ``d_model`` features at once. The final
    dimension is then viewed as ``(num_heads, d_head)`` so each head can run
    scaled dot-product attention independently. RoPE is optional here so the
    same implementation can satisfy both attention tests in the assignment.

    Args:
        d_model: Input and output feature dimension.
        num_heads: Number of attention heads. Must divide ``d_model`` exactly.
        theta: RoPE frequency base. Supply this together with ``max_seq_len``
            to enable RoPE; leave both as ``None`` to disable it.
        max_seq_len: Maximum token position cached by RoPE.
        device: Device on which parameters and optional RoPE buffers are stored.
        dtype: Data type of the learned projection parameters.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(
                "num_heads must be positive and divide d_model exactly, got "
                f"d_model={d_model}, num_heads={num_heads}"
            )
        if (theta is None) != (max_seq_len is None):
            raise ValueError(
                "theta and max_seq_len must either both be provided or both be None"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = (
            RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        """Change ``(..., sequence, d_model)`` to ``(..., heads, sequence, d_head)``."""

        sequence_length = x.shape[-2]
        split = x.reshape(*x.shape[:-2], sequence_length, self.num_heads, self.d_head)
        return split.transpose(-3, -2)

    def _validate_cache(self, cache: LayerKVCache, key: Tensor) -> None:
        """Check that a layer cache can be extended by ``key``."""

        expected_batch_shape = key.shape[:-3]
        if cache.key.shape[:-3] != expected_batch_shape:
            raise ValueError(
                "cache batch shape must match the input batch shape, got "
                f"{tuple(cache.key.shape[:-3])} and {tuple(expected_batch_shape)}"
            )
        if cache.key.shape[-3] != self.num_heads:
            raise ValueError(f"cache has {cache.key.shape[-3]} heads, expected {self.num_heads}")
        if cache.key.shape[-1] != self.d_head:
            raise ValueError(f"cache head dimension is {cache.key.shape[-1]}, expected {self.d_head}")
        if cache.key.device != key.device:
            raise ValueError(f"cache is on {cache.key.device}, but input is on {key.device}")
        if cache.key.dtype != key.dtype:
            raise ValueError(f"cache dtype is {cache.key.dtype}, but input dtype is {key.dtype}")

    def _forward_with_optional_cache(
        self,
        x: Tensor,
        token_positions: Tensor | None,
        cache: LayerKVCache | None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Run attention and return its output together with the extended cache."""

        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected input's final dimension to be {self.d_model}, got {x.shape[-1]}")

        sequence_length = x.shape[-2]
        if sequence_length == 0:
            raise ValueError("attention input must contain at least one token")

        # Project all heads together, then expose a separate head dimension.
        query = self._split_heads(self.q_proj(x))
        new_key = self._split_heads(self.k_proj(x))
        new_value = self._split_heads(self.v_proj(x))

        past_length = 0 if cache is None else cache.sequence_length
        if cache is not None:
            self._validate_cache(cache, new_key)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    past_length,
                    past_length + sequence_length,
                    device=x.device,
                )
            query = self.rope(query, token_positions)
            new_key = self.rope(new_key, token_positions)

        if cache is None:
            key = new_key
            value = new_value
        else:
            # Sequence is the penultimate dimension in the split-head layout.
            key = torch.cat((cache.key, new_key), dim=-2)
            value = torch.cat((cache.value, new_value), dim=-2)

        total_key_length = past_length + sequence_length
        query_indices = past_length + torch.arange(
            sequence_length,
            device=x.device,
        )
        key_indices = torch.arange(total_key_length, device=x.device)
        causal_mask = key_indices.unsqueeze(0) <= query_indices.unsqueeze(1)
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            mask=causal_mask,
        )

        # Move sequence before heads again and flatten the two feature axes:
        # (..., heads, sequence, d_head) -> (..., sequence, d_model).
        merged = attended.transpose(-3, -2).contiguous()
        merged = merged.reshape(*x.shape[:-2], sequence_length, self.d_model)
        return self.output_proj(merged), LayerKVCache(key=key, value=value)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """Return causal self-attention output with the same shape as ``x``."""

        output, _ = self._forward_with_optional_cache(
            x,
            token_positions=token_positions,
            cache=None,
        )
        return output

    def forward_with_cache(
        self,
        x: Tensor,
        *,
        cache: LayerKVCache | None = None,
        token_positions: Tensor | None = None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Attend over cached history and return output plus the extended cache."""

        return self._forward_with_optional_cache(
            x,
            token_positions=token_positions,
            cache=cache,
        )


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU (also called Swish) activation element by element.

    SiLU is defined as ``x * sigmoid(x)``. Unlike ReLU, it is smooth around
    zero and allows small negative outputs instead of discarding every negative
    input completely.
    """

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network using a SiLU-gated linear unit.

    For every token vector ``x`` this module computes::

        gate = SiLU(W1 x)
        value = W3 x
        output = W2 (gate * value)


    Args:
        d_model: Input and output feature dimension.
        d_ff: Hidden feature dimension. If omitted, use approximately
            ``8/3 * d_model``, rounded to the nearest multiple of 64.
        device: Device on which to allocate all three projections.
        dtype: Data type of all three projections.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_ff is None:
            # Multiples of 64 tend to use accelerator matrix hardware more
            # efficiently. max(64, ...) also handles very small toy models.
            approximate_d_ff = (8.0 / 3.0) * d_model
            d_ff = max(64, round(approximate_d_ff / 64) * 64)

        self.d_model = d_model
        self.d_ff = d_ff

        # W1 produces the values passed through SiLU to form the gate.
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # W2 is the down-projection that restores the original model dimension.
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

        # W3 produces the content controlled by the gate from W1.
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """Process ``(..., d_model)`` and return the same overall shape."""

        gate = silu(self.w1(x))
        value = self.w3(x)

        # This is element-wise multiplication in the d_ff dimension, not a
        # matrix multiplication. The gate scales each hidden feature.
        gated_hidden = gate * value
        return self.w2(gated_hidden)


class TransformerBlock(nn.Module):
    """Combine causal self-attention and SwiGLU using pre-norm residual paths.

    The block performs these two sequential updates::

        x = x + attention(rmsnorm_1(x))
        x = x + ffn(rmsnorm_2(x))

    Normalizing before each sublayer leaves an unnormalized identity path for
    gradients through both residual connections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        """Process token representations without changing their shape."""

        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_with_cache(
        self,
        x: Tensor,
        *,
        cache: LayerKVCache | None = None,
        token_positions: Tensor | None = None,
    ) -> tuple[Tensor, LayerKVCache]:
        """Process new tokens while extending this block's attention cache."""

        attention_output, updated_cache = self.attn.forward_with_cache(
            self.ln1(x),
            cache=cache,
            token_positions=token_positions,
        )
        x = x + attention_output
        x = x + self.ffn(self.ln2(x))
        return x, updated_cache


class TransformerLM(nn.Module):
    """Decoder-only Transformer that maps token IDs to vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {context_length}"
            )
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return ``(..., sequence_length, vocab_size)`` unnormalized logits."""

        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        token_positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                token_positions=token_positions,
            )

        normalized = self.ln_final(hidden_states)
        return self.lm_head(normalized)

    def forward_with_cache(
        self,
        token_ids: Tensor,
        cache: KVCache | None = None,
    ) -> tuple[Tensor, KVCache]:
        """Return logits for new tokens and an extended per-layer KV cache.

        ``token_ids`` may contain a complete prompt (prefill) or one or more
        later tokens (decode). If ``cache`` already contains ``P`` positions,
        the new tokens receive RoPE positions beginning at ``P``.
        """

        sequence_length = token_ids.shape[-1]
        if sequence_length == 0:
            raise ValueError("token_ids must contain at least one token")

        if cache is None:
            past_length = 0
            layer_caches: tuple[LayerKVCache | None, ...] = (None,) * self.num_layers
        else:
            if len(cache.layers) != self.num_layers:
                raise ValueError(f"cache contains {len(cache.layers)} layers, but model has {self.num_layers}")
            past_length = cache.sequence_length
            layer_caches = cache.layers

        total_length = past_length + sequence_length
        if total_length > self.context_length:
            raise ValueError(
                f"cached sequence length {past_length} plus input length "
                f"{sequence_length} exceeds context length {self.context_length}"
            )

        token_positions = torch.arange(
            past_length,
            total_length,
            device=token_ids.device,
        )
        hidden_states = self.token_embeddings(token_ids)
        updated_layer_caches: list[LayerKVCache] = []
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            hidden_states, updated_layer_cache = layer.forward_with_cache(
                hidden_states,
                cache=layer_cache,
                token_positions=token_positions,
            )
            updated_layer_caches.append(updated_layer_cache)

        normalized = self.ln_final(hidden_states)
        logits = self.lm_head(normalized)
        updated_cache = KVCache(
            layers=tuple(updated_layer_caches),
            sequence_length=total_length,
        )
        return logits, updated_cache
