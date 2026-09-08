"""Train the decoder-only Transformer on memory-mapped token arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.training import TrainingConfig, config_as_dict, train


def default_device() -> str:
    """Choose the best accelerator available to this PyTorch installation."""

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TransformerLM from encoded one-dimensional .npy token arrays."
        )
    )
    parser.add_argument(
        "--train-data",
        dest="train_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-data",
        dest="validation_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, required=True)
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--batch-size", type=int, default=32)
    optimization.add_argument("--max-iterations", type=int, default=10_000)
    optimization.add_argument("--max-learning-rate", type=float, default=3e-4)
    optimization.add_argument("--min-learning-rate", type=float, default=3e-5)
    optimization.add_argument("--warmup-iterations", type=int, default=100)
    optimization.add_argument("--cosine-cycle-iterations", type=int, default=10_000)
    optimization.add_argument("--beta1", type=float, default=0.9)
    optimization.add_argument("--beta2", type=float, default=0.95)
    optimization.add_argument("--adam-epsilon", type=float, default=1e-8)
    optimization.add_argument("--weight-decay", type=float, default=0.1)
    optimization.add_argument("--max-gradient-norm", type=float, default=1.0)

    runtime = parser.add_argument_group("runtime and reporting")
    runtime.add_argument("--log-interval", type=int, default=10)
    runtime.add_argument("--validation-interval", type=int, default=100)
    runtime.add_argument("--validation-batches", type=int, default=10)
    runtime.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1_000,
        help="save every N updates; use 0 to save only the final checkpoint",
    )
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--device", default=default_device())
    return parser.parse_args(arguments)


def main() -> None:
    config = TrainingConfig(**vars(parse_args()))
    print(json.dumps(config_as_dict(config), indent=2, sort_keys=True))
    train(config)


if __name__ == "__main__":
    main()
