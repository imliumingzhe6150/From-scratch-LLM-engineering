"""Plot training and validation loss against steps and wall-clock time."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Keep matplotlib's font cache inside the ignored output directory. This avoids
# relying on a writable user-level configuration directory on shared machines.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MATPLOTLIB_CONFIG = _PROJECT_ROOT / "output" / ".matplotlib"
_MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / "output" / ".cache"))

import matplotlib  # noqa: E402 - cache directories must be set before import

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be set before pyplot

# Preserve editable text in SVG/PDF output and use broadly available fonts.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

TRAIN_COLOR = "#0F4D92"
VALIDATION_COLOR = "#B64342"
FIELD_PATTERN = re.compile(r"([a-z_]+)=([^\s]+)")


@dataclass(frozen=True)
class LossPoint:
    """One logged loss observation."""

    iteration: int
    loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class ExperimentLog:
    """Configuration and all loss observations parsed from one run."""

    config: dict[str, object]
    training: list[LossPoint]
    validation: list[LossPoint]


def _parse_config(text: str) -> dict[str, object]:
    """Read the leading JSON configuration printed by the training script."""

    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return {}
    try:
        config, _ = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return {}
    if not isinstance(config, dict):
        return {}
    return config


def _check_monotonic(points: list[LossPoint], split: str) -> None:
    """Ensure the log order is suitable for an unmodified trend line."""

    for previous, current in zip(points, points[1:]):
        if current.iteration <= previous.iteration:
            raise ValueError(f"{split} iterations must be strictly increasing")
        if current.elapsed_seconds <= previous.elapsed_seconds:
            raise ValueError(f"{split} elapsed times must be strictly increasing")


def parse_experiment_log(log_path: str | Path) -> ExperimentLog:
    """Parse every train and validation loss point from a training log."""

    text = Path(log_path).read_text(encoding="utf-8")
    training: list[LossPoint] = []
    validation: list[LossPoint] = []

    for line in text.splitlines():
        fields = dict(FIELD_PATTERN.findall(line))
        if not {"iteration", "elapsed_seconds"} <= fields.keys():
            continue

        if "train_loss" in fields:
            training.append(
                LossPoint(
                    iteration=int(fields["iteration"]),
                    loss=float(fields["train_loss"]),
                    elapsed_seconds=float(fields["elapsed_seconds"]),
                )
            )
        elif "validation_loss" in fields:
            validation.append(
                LossPoint(
                    iteration=int(fields["iteration"]),
                    loss=float(fields["validation_loss"]),
                    elapsed_seconds=float(fields["elapsed_seconds"]),
                )
            )

    if not training:
        raise ValueError(f"no training-loss points found in {log_path}")
    if not validation:
        raise ValueError(f"no validation-loss points found in {log_path}")

    _check_monotonic(training, "training")
    _check_monotonic(validation, "validation")
    return ExperimentLog(
        config=_parse_config(text),
        training=training,
        validation=validation,
    )


def _plot_split(
    axis: plt.Axes,
    x_values: list[float],
    points: list[LossPoint],
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    """Draw one unsmoothed loss series."""

    axis.plot(
        x_values,
        [point.loss for point in points],
        color=color,
        linewidth=1.6,
        marker=marker,
        markersize=3.5,
        markeredgewidth=0,
        label=label,
    )


def create_figure(experiment: ExperimentLog) -> plt.Figure:
    """Create the two-panel Section 7.1 loss figure without saving it."""

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    by_step, by_time = axes

    _plot_split(
        by_step,
        [point.iteration for point in experiment.training],
        experiment.training,
        label="Train",
        color=TRAIN_COLOR,
        marker="o",
    )
    _plot_split(
        by_step,
        [point.iteration for point in experiment.validation],
        experiment.validation,
        label="Validation",
        color=VALIDATION_COLOR,
        marker="s",
    )
    by_step.set_xlabel("Gradient step")
    by_step.set_ylabel("Cross-entropy loss")
    by_step.set_title("Optimization progress")
    by_step.legend(loc="upper right")

    _plot_split(
        by_time,
        [point.elapsed_seconds for point in experiment.training],
        experiment.training,
        label="Train",
        color=TRAIN_COLOR,
        marker="o",
    )
    _plot_split(
        by_time,
        [point.elapsed_seconds for point in experiment.validation],
        experiment.validation,
        label="Validation",
        color=VALIDATION_COLOR,
        marker="s",
    )
    by_time.set_xlabel("Wall-clock time (seconds)")
    by_time.set_title("End-to-end runtime")

    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.13,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        axis.tick_params(width=0.8, length=3)
        axis.margins(x=0.03)

    final_step = max(point.iteration for point in experiment.training)
    log_interval = experiment.config.get("log_interval", "unknown")
    validation_batches = experiment.config.get("validation_batches", "unknown")
    seed = experiment.config.get("seed", "unknown")
    figure.suptitle(
        f"Training and validation loss decline together over {final_step} steps",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.035,
        (
            f"Train: mean over each logging interval (up to {log_interval} updates); "
            f"validation: mean over {validation_batches} sampled batches; "
            f"seed={seed}; one run; no smoothing."
        ),
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#4D4D4D",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.23,
        top=0.80,
        wspace=0.14,
    )
    return figure


def save_figure(figure: plt.Figure, output_prefix: Path) -> list[Path]:
    """Export editable vectors, a print raster, and a PNG preview."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".tiff"),
        output_prefix.with_suffix(".png"),
    ]
    figure.savefig(output_paths[0], bbox_inches="tight")
    figure.savefig(output_paths[1], bbox_inches="tight")
    figure.savefig(output_paths[2], dpi=600, bbox_inches="tight")
    figure.savefig(output_paths[3], dpi=300, bbox_inches="tight")
    return output_paths


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/validation loss against steps and wall-clock time."
    )
    parser.add_argument("log_path", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="output path without an extension; defaults beside the input log",
    )
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    experiment = parse_experiment_log(args.log_path)
    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.log_path.with_name(f"{args.log_path.stem}_loss_curves")

    figure = create_figure(experiment)
    output_paths = save_figure(figure, output_prefix)
    plt.close(figure)

    print(f"training points: {len(experiment.training)}")
    print(f"validation points: {len(experiment.validation)}")
    for output_path in output_paths:
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
