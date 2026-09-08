"""Compare decoding policies and save a reproducible generation record."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.generation_evaluation import (
    ArtifactIdentity,
    DecodingPolicy,
    evaluate_generation,
    identify_artifact,
    write_generation_evaluation,
)
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from scripts.generate import END_OF_TEXT, default_device

DEFAULT_POLICIES = (
    DecodingPolicy(name="greedy", temperature=0.0, top_p=1.0),
    DecodingPolicy(name="conservative", temperature=0.8, top_p=0.9),
    DecodingPolicy(name="diverse", temperature=1.1, top_p=0.95),
)


def parse_policy(value: str) -> DecodingPolicy:
    """Parse ``NAME:TEMPERATURE:TOP_P`` from the command line."""

    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("policy must have the form NAME:TEMPERATURE:TOP_P")
    name, temperature_text, top_p_text = parts
    try:
        return DecodingPolicy(
            name=name,
            temperature=float(temperature_text),
            top_p=float(top_p_text),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate one completion per decoding policy and save all reproducibility metadata as JSON.")
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--policy",
        action="append",
        type=parse_policy,
        help=(
            "repeatable NAME:TEMPERATURE:TOP_P policy; when omitted, evaluate "
            "greedy, conservative, and diverse defaults"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/experiments/tinystories_generation_evaluation.json"),
    )

    model = parser.add_argument_group("model architecture")
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)
    return parser.parse_args(arguments)


def _artifact_identities(args: argparse.Namespace) -> dict[str, ArtifactIdentity]:
    return {
        "checkpoint": identify_artifact(args.checkpoint_path),
        "vocabulary": identify_artifact(args.vocab_path),
        "merges": identify_artifact(args.merges_path),
    }


def main() -> None:
    args = parse_args()
    policies = args.policy or list(DEFAULT_POLICIES)
    tokenizer = Tokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=[END_OF_TEXT],
    )
    end_token_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]

    model_config: dict[str, int | float] = {
        "vocab_size": len(tokenizer.vocab),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = TransformerLM(
        vocab_size=len(tokenizer.vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    checkpoint = torch.load(
        args.checkpoint_path,
        weights_only=True,
        map_location=args.device,
    )
    model.load_state_dict(checkpoint["model"])

    evaluation = evaluate_generation(
        model,
        tokenizer,
        prompt=args.prompt,
        policies=policies,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        end_token_id=end_token_id,
        model_config=model_config,
        artifacts=_artifact_identities(args),
    )
    output_path = write_generation_evaluation(evaluation, args.output_path)

    print(f"saved: {output_path}")
    for run in evaluation.runs:
        print(
            f"\n[{run.policy.name}] "
            f"temperature={run.policy.temperature:g} "
            f"top_p={run.policy.top_p:g} "
            f"tokens={run.generated_token_count} "
            f"stop={run.stopping_reason}"
        )
        print(run.full_text)


if __name__ == "__main__":
    main()
