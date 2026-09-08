"""Neural-network utility functions implemented from first principles."""

from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Normalize ``x`` into probabilities along ``dim``.

    Softmax is unchanged if the same constant is subtracted from every value
    along the normalized dimension. Subtracting that dimension's maximum makes
    the largest shifted value zero, so its exponential is exactly one instead
    of risking overflow for large logits.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which probabilities should sum to one.

    Returns:
        A tensor with the same shape and dtype as ``x``.
    """

    maximum = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - maximum
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials, dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy for logits with arbitrary batch dimensions.

    The final dimension of ``logits`` contains vocabulary scores, while
    ``targets`` contains the correct vocabulary index for every preceding
    position. Subtracting the maximum score before exponentiation prevents
    overflow without changing the resulting probabilities.

    Args:
        logits: Unnormalized scores of shape ``(..., vocab_size)``.
        targets: Correct token indices of shape ``(...)``.

    Returns:
        The scalar cross-entropy averaged over all batch-like dimensions.
    """

    maximum = torch.max(logits, dim=-1, keepdim=True).values
    shifted_logits = logits - maximum

    log_normalizer = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    return torch.mean(log_normalizer - target_logits)


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    """Clip the combined L2 norm of all available gradients in place."""

    if max_l2_norm < 0:
        raise ValueError(f"max_l2_norm must be non-negative, got {max_l2_norm}")

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return

    squared_l2_norm = sum(torch.sum(gradient**2) for gradient in gradients)
    l2_norm = torch.sqrt(squared_l2_norm)

    if l2_norm > max_l2_norm:
        scale = max_l2_norm / (l2_norm + eps)
        for gradient in gradients:
            gradient.mul_(scale)
