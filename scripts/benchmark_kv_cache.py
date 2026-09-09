"""Benchmark cached and uncached autoregressive decoding."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / "output" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / "output" / ".cache"))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from cs336_basics.model import KVCache, TransformerLM  # noqa: E402
from scripts.generate import default_device  # noqa: E402


@dataclass(frozen=True)
class BenchmarkCase:
    """One prompt-length and generation-length combination."""

    prompt_tokens: int
    generated_tokens: int

    @classmethod
    def parse(cls, value: str) -> BenchmarkCase:
        """Parse ``PROMPT_TOKENS:GENERATED_TOKENS``."""

        try:
            prompt_text, generated_text = value.split(":")
            case = cls(int(prompt_text), int(generated_text))
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError("case must have the form PROMPT_TOKENS:GENERATED_TOKENS") from error
        if case.prompt_tokens <= 0 or case.generated_tokens <= 0:
            raise argparse.ArgumentTypeError("case lengths must both be positive")
        return case


@dataclass(frozen=True)
class Measurement:
    """One timed generation repeat."""

    mode: str
    prompt_tokens: int
    generated_tokens: int
    repeat: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    tokens_per_second: float
    peak_cache_bytes: int


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before reading the clock."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run_generation_once(
    model: TransformerLM,
    prompt: torch.Tensor,
    generated_tokens: int,
    *,
    use_cache: bool,
) -> tuple[list[int], float, float, int]:
    """Greedily generate tokens and return output, phase timings, and cache size."""

    device = prompt.device
    all_tokens = prompt[0].tolist()
    completion: list[int] = []

    synchronize(device)
    prefill_start = time.perf_counter()
    cache: KVCache | None = None
    if use_cache:
        logits, cache = model.forward_with_cache(prompt)
    else:
        logits = model(prompt)
    synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start

    peak_cache_bytes = 0 if cache is None else cache.size_bytes
    synchronize(device)
    decode_start = time.perf_counter()
    for decode_step in range(generated_tokens):
        next_token = int(torch.argmax(logits[0, -1]).item())
        completion.append(next_token)
        all_tokens.append(next_token)
        if decode_step + 1 == generated_tokens:
            break

        if use_cache:
            if cache is None:
                raise RuntimeError("cached benchmark lost its cache state")
            if cache.sequence_length < model.context_length:
                model_input = torch.tensor(
                    [[next_token]],
                    dtype=torch.long,
                    device=device,
                )
                logits, cache = model.forward_with_cache(model_input, cache=cache)
            else:
                model_input = torch.tensor(
                    all_tokens[-model.context_length :],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                logits, cache = model.forward_with_cache(model_input)
            peak_cache_bytes = max(peak_cache_bytes, cache.size_bytes)
        else:
            model_input = torch.tensor(
                all_tokens[-model.context_length :],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            logits = model(model_input)

    synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    return completion, prefill_seconds, decode_seconds, peak_cache_bytes


def benchmark_cases(
    model: TransformerLM,
    cases: list[BenchmarkCase],
    *,
    warmup_repeats: int,
    measured_repeats: int,
    seed: int,
) -> list[Measurement]:
    """Benchmark both modes and verify token-for-token greedy equivalence."""

    if warmup_repeats < 0:
        raise ValueError("warmup_repeats must be non-negative")
    if measured_repeats <= 0:
        raise ValueError("measured_repeats must be positive")

    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    prompts: dict[BenchmarkCase, torch.Tensor] = {}
    for case in cases:
        if case.prompt_tokens > model.context_length:
            raise ValueError(f"prompt length {case.prompt_tokens} exceeds context length {model.context_length}")
        prompts[case] = torch.randint(
            0,
            model.vocab_size,
            (1, case.prompt_tokens),
            generator=generator,
        ).to(device)

    model.eval()
    for _ in range(warmup_repeats):
        for case in cases:
            for use_cache in (False, True):
                run_generation_once(
                    model,
                    prompts[case],
                    case.generated_tokens,
                    use_cache=use_cache,
                )

    measurements: list[Measurement] = []
    for repeat in range(measured_repeats):
        for case in cases:
            outputs: dict[str, list[int]] = {}
            for use_cache in (False, True):
                mode = "cached" if use_cache else "uncached"
                output, prefill, decode, peak_cache = run_generation_once(
                    model,
                    prompts[case],
                    case.generated_tokens,
                    use_cache=use_cache,
                )
                outputs[mode] = output
                total = prefill + decode
                measurements.append(
                    Measurement(
                        mode=mode,
                        prompt_tokens=case.prompt_tokens,
                        generated_tokens=case.generated_tokens,
                        repeat=repeat,
                        prefill_seconds=prefill,
                        decode_seconds=decode,
                        total_seconds=total,
                        tokens_per_second=case.generated_tokens / total,
                        peak_cache_bytes=peak_cache,
                    )
                )
            if outputs["cached"] != outputs["uncached"]:
                raise RuntimeError(
                    "cached and uncached greedy outputs differ for "
                    f"prompt={case.prompt_tokens}, generated={case.generated_tokens}"
                )
    return measurements


def summarize_measurements(
    measurements: list[Measurement],
) -> list[dict[str, int | float | str]]:
    """Return median timings and throughput for each case and mode."""

    grouped: dict[tuple[str, int, int], list[Measurement]] = {}
    for measurement in measurements:
        key = (
            measurement.mode,
            measurement.prompt_tokens,
            measurement.generated_tokens,
        )
        grouped.setdefault(key, []).append(measurement)

    summaries: list[dict[str, int | float | str]] = []
    for (mode, prompt_tokens, generated_tokens), group in sorted(grouped.items()):
        summaries.append(
            {
                "mode": mode,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "repeats": len(group),
                "median_prefill_seconds": statistics.median(item.prefill_seconds for item in group),
                "median_decode_seconds": statistics.median(item.decode_seconds for item in group),
                "median_total_seconds": statistics.median(item.total_seconds for item in group),
                "median_tokens_per_second": statistics.median(item.tokens_per_second for item in group),
                "peak_cache_bytes": max(item.peak_cache_bytes for item in group),
            }
        )
    return summaries


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write a non-empty list of dictionaries atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_summary(
    summaries: list[dict[str, object]],
    output_path: Path,
    *,
    title: str = "Autoregressive generation throughput",
) -> None:
    """Plot median end-to-end generation throughput by benchmark case."""

    cases = sorted({(int(row["prompt_tokens"]), int(row["generated_tokens"])) for row in summaries})
    by_key = {(str(row["mode"]), int(row["prompt_tokens"]), int(row["generated_tokens"])): row for row in summaries}
    uncached = [
        float(by_key[("uncached", prompt, generated)]["median_tokens_per_second"]) for prompt, generated in cases
    ]
    cached = [float(by_key[("cached", prompt, generated)]["median_tokens_per_second"]) for prompt, generated in cases]

    positions = list(range(len(cases)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(6.5, len(cases) * 1.6), 4.2))
    axis.bar(
        [position - width / 2 for position in positions],
        uncached,
        width,
        label="Uncached",
        color="#9aa0a6",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        cached,
        width,
        label="KV cache",
        color="#2a6fbb",
    )
    for position, uncached_rate, cached_rate in zip(
        positions,
        uncached,
        cached,
        strict=True,
    ):
        speedup = cached_rate / uncached_rate
        axis.text(
            position + width / 2,
            cached_rate,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_xticks(
        positions,
        [f"P{prompt}\nG{generated}" for prompt, generated in cases],
    )
    axis.set_xlabel("Prompt tokens (P) and generated tokens (G)")
    axis.set_ylabel("Generated tokens / second (median)")
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 identity for a benchmark input artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=BenchmarkCase.parse,
        help="repeatable PROMPT_TOKENS:GENERATED_TOKENS; defaults to three cases",
    )
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--measured-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("output/experiments/kv_cache_benchmark"),
    )
    parser.add_argument(
        "--readme-prefix",
        type=Path,
        help="optional prefix for an additional public PNG and SVG figure",
    )
    parser.add_argument(
        "--source-data-output",
        type=Path,
        help="optional path for a copy of the median summary CSV",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="optional path for a copy of the reproducibility metadata JSON",
    )

    model = parser.add_argument_group("model architecture")
    model.add_argument("--vocab-size", type=int, default=10_000)
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = args.case or [
        BenchmarkCase(16, 32),
        BenchmarkCase(64, 64),
        BenchmarkCase(128, 128),
    ]
    device = torch.device(args.device)
    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = TransformerLM(**model_config, device=device)
    if args.checkpoint_path is not None:
        checkpoint = torch.load(
            args.checkpoint_path,
            weights_only=True,
            map_location=device,
        )
        model.load_state_dict(checkpoint["model"])

    measurements = benchmark_cases(
        model,
        cases,
        warmup_repeats=args.warmup_repeats,
        measured_repeats=args.measured_repeats,
        seed=args.seed,
    )
    summaries = summarize_measurements(measurements)

    raw_path = args.output_prefix.with_name(f"{args.output_prefix.name}_raw.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    figure_path = args.output_prefix.with_suffix(".png")
    svg_path = args.output_prefix.with_suffix(".svg")
    metadata_path = args.output_prefix.with_suffix(".json")
    write_csv([asdict(item) for item in measurements], raw_path)
    write_csv(summaries, summary_path)
    figure_title = (
        "KV-cache generation throughput "
        f"({device.type.upper()}, median of {args.measured_repeats})"
    )
    plot_summary(summaries, figure_path, title=figure_title)
    plot_summary(summaries, svg_path, title=figure_title)
    if args.readme_prefix is not None:
        plot_summary(
            summaries,
            args.readme_prefix.with_suffix(".png"),
            title=figure_title,
        )
        plot_summary(
            summaries,
            args.readme_prefix.with_suffix(".svg"),
            title=figure_title,
        )
    if args.source_data_output is not None:
        write_csv(summaries, args.source_data_output)

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "warmup_repeats": args.warmup_repeats,
        "measured_repeats": args.measured_repeats,
        "model_config": model_config,
        "checkpoint_path": (str(args.checkpoint_path) if args.checkpoint_path is not None else None),
        "checkpoint_sha256": (
            sha256_file(args.checkpoint_path)
            if args.checkpoint_path is not None
            else None
        ),
        "cases": [asdict(case) for case in cases],
    }
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    for destination in (metadata_path, args.metadata_output):
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(metadata_text, encoding="utf-8")

    print(f"raw measurements: {raw_path}")
    print(f"median summary: {summary_path}")
    print(f"figure: {figure_path}")
    print(f"editable figure: {svg_path}")
    print(f"metadata: {metadata_path}")
    for row in summaries:
        print(
            f"{row['mode']:>8} P{row['prompt_tokens']} G{row['generated_tokens']}: "
            f"{float(row['median_tokens_per_second']):.2f} tokens/s"
        )


if __name__ == "__main__":
    main()
