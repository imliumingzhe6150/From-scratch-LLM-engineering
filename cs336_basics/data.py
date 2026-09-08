"""Data-sampling utilities for language-model training."""

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample input sequences and their one-token-shifted targets.

    The dataset remains a NumPy array on CPU. Only the sampled batch is
    converted to ``torch.long`` and transferred to the requested device.
    """

    if dataset.ndim != 1:
        raise ValueError(f"dataset must be one-dimensional, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    num_starting_positions = len(dataset) - context_length
    if num_starting_positions <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens, "
            f"got {len(dataset)} tokens and context_length={context_length}"
        )

    starting_positions = np.random.randint(
        low=0,
        high=num_starting_positions,
        size=batch_size,
    )
    offsets = np.arange(context_length)

    input_batch = dataset[starting_positions[:, None] + offsets[None, :]]
    target_batch = dataset[starting_positions[:, None] + offsets[None, :] + 1]

    inputs = torch.from_numpy(np.asarray(input_batch)).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(np.asarray(target_batch)).to(device=device, dtype=torch.long)
    return inputs, targets
