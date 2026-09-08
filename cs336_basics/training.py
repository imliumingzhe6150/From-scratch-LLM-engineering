"""End-to-end utilities for training a Transformer language model."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradients, cross_entropy
from cs336_basics.optimizer import AdamW, cosine_learning_rate_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    """All model, optimizer, and loop settings needed for one training run."""

    train_data_path: Path
    validation_data_path: Path
    checkpoint_path: Path
    vocab_size: int
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000.0
    batch_size: int = 32
    max_iterations: int = 10_000
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iterations: int = 100
    cosine_cycle_iterations: int = 10_000
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    log_interval: int = 10
    validation_interval: int = 100
    validation_batches: int = 10
    checkpoint_interval: int = 1_000
    seed: int = 42
    device: str = "cpu"
    resume_from: Path | None = None

    def validate(self) -> None:
        """Reject configurations that cannot form a valid training run."""

        positive_integers = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "batch_size": self.batch_size,
            "max_iterations": self.max_iterations,
            "cosine_cycle_iterations": self.cosine_cycle_iterations,
            "log_interval": self.log_interval,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.cosine_cycle_iterations <= self.warmup_iterations:
            raise ValueError(
                "cosine_cycle_iterations must be greater than warmup_iterations"
            )
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.max_learning_rate < 0 or self.min_learning_rate < 0:
            raise ValueError("learning rates must be non-negative")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError(
                "min_learning_rate cannot be greater than max_learning_rate"
            )
        if self.max_gradient_norm < 0:
            raise ValueError("max_gradient_norm must be non-negative")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("the per-head dimension must be even for RoPE")


def load_token_array(path: str | Path, context_length: int) -> np.memmap:
    """Open a one-dimensional integer ``.npy`` token array without loading it."""

    token_ids = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(token_ids, np.memmap):
        raise ValueError(f"expected {path} to be a memory-mappable .npy array")
    if token_ids.ndim != 1:
        raise ValueError(f"token array must be one-dimensional, got {token_ids.shape}")
    if not np.issubdtype(token_ids.dtype, np.integer):
        raise ValueError(f"token array must have an integer dtype, got {token_ids.dtype}")
    if len(token_ids) <= context_length:
        raise ValueError(
            "token array must contain at least context_length + 1 tokens, "
            f"got {len(token_ids)} tokens and context_length={context_length}"
        )
    return token_ids


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    *,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    """Estimate mean loss on independently sampled validation batches."""

    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        losses.append(cross_entropy(model(inputs), targets).item())
    model.train(was_training)
    return float(np.mean(losses))


def train(
    config: TrainingConfig,
    *,
    log: Callable[[str], None] = print,
) -> tuple[TransformerLM, AdamW]:
    """Train from scratch or resume until ``config.max_iterations`` updates."""

    config.validate()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_data = load_token_array(config.train_data_path, config.context_length)
    validation_data = load_token_array(
        config.validation_data_path,
        config.context_length,
    )

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    start_iteration = 0
    if config.resume_from is not None:
        start_iteration = load_checkpoint(
            config.resume_from,
            model,
            optimizer,
            map_location=config.device,
        )
        if start_iteration > config.max_iterations:
            raise ValueError(
                f"checkpoint is at iteration {start_iteration}, beyond requested "
                f"max_iterations={config.max_iterations}"
            )
        log(f"resumed_from={config.resume_from} iteration={start_iteration}")

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log(
        f"device={config.device} parameters={parameter_count:,} "
        f"train_tokens={len(train_data):,} validation_tokens={len(validation_data):,}"
    )

    model.train()
    interval_loss = 0.0
    interval_tokens = 0
    interval_iterations = 0
    interval_train_seconds = 0.0
    run_start = time.perf_counter()

    for iteration in range(start_iteration, config.max_iterations):
        learning_rate = cosine_learning_rate_schedule(
            iteration=iteration,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_iters=config.warmup_iterations,
            cosine_cycle_iters=config.cosine_cycle_iterations,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        iteration_start = time.perf_counter()
        inputs, targets = get_batch(
            train_data,
            config.batch_size,
            config.context_length,
            config.device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        clip_gradients(model.parameters(), config.max_gradient_norm)
        optimizer.step()
        interval_train_seconds += time.perf_counter() - iteration_start

        completed_iterations = iteration + 1
        interval_loss += loss.item()
        interval_tokens += targets.numel()
        interval_iterations += 1

        if (
            completed_iterations % config.log_interval == 0
            or completed_iterations == config.max_iterations
        ):
            elapsed_seconds = time.perf_counter() - run_start
            log(
                f"iteration={completed_iterations} "
                f"train_loss={interval_loss / interval_iterations:.6f} "
                f"lr={learning_rate:.3e} "
                f"tokens_per_second={interval_tokens / interval_train_seconds:.0f} "
                f"elapsed_seconds={elapsed_seconds:.2f}"
            )
            interval_loss = 0.0
            interval_tokens = 0
            interval_iterations = 0
            interval_train_seconds = 0.0

        if (
            completed_iterations % config.validation_interval == 0
            or completed_iterations == config.max_iterations
        ):
            validation_loss = evaluate_loss(
                model,
                validation_data,
                batch_size=config.batch_size,
                context_length=config.context_length,
                device=config.device,
                num_batches=config.validation_batches,
            )
            elapsed_seconds = time.perf_counter() - run_start
            log(
                f"iteration={completed_iterations} "
                f"validation_loss={validation_loss:.6f} "
                f"elapsed_seconds={elapsed_seconds:.2f}"
            )

        if (
            config.checkpoint_interval > 0
            and completed_iterations % config.checkpoint_interval == 0
            and completed_iterations != config.max_iterations
        ):
            save_checkpoint(
                model,
                optimizer,
                completed_iterations,
                config.checkpoint_path,
            )
            log(
                f"iteration={completed_iterations} "
                f"checkpoint={config.checkpoint_path}"
            )

    save_checkpoint(
        model,
        optimizer,
        config.max_iterations,
        config.checkpoint_path,
    )
    log(f"final_checkpoint={config.checkpoint_path} iteration={config.max_iterations}")
    return model, optimizer


def config_as_dict(config: TrainingConfig) -> dict[str, object]:
    """Return a JSON-friendly view of a training configuration."""

    values = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }
