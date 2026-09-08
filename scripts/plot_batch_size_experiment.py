"""Plot the valid systems and optimization views of the batch-size runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.plot_training_log import (
    FIELD_PATTERN,
    ExperimentLog,
    parse_experiment_log,
    plt,
)


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
    }
)

FULL_BATCH_SIZES = (1, 8, 32, 64, 128)
FIXED_TOKEN_BATCH_SIZES = (1, 8)
FIXED_UPDATE_BATCH_SIZES = (8, 32, 64, 128)
CONTROLLED_FIELDS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "rope_theta",
    "max_learning_rate",
    "min_learning_rate",
    "beta1",
    "beta2",
    "adam_epsilon",
    "weight_decay",
    "max_gradient_norm",
    "seed",
    "train_data_path",
    "validation_data_path",
    "device",
)
COLORS = {
    1: "#777777",
    8: "#32689B",
    32: "#4A9085",
    64: "#D08A34",
    128: "#A64B4B",
}
README_SIGNAL_COLOR = "#2F6B9F"
README_NEUTRAL_COLOR = "#858585"


@dataclass(frozen=True)
class BatchRun:
    """One parsed run plus the accounting needed for honest comparison."""

    source_path: Path
    experiment: ExperimentLog
    final_tokens_per_second: float
    run_kind: str

    @property
    def batch_size(self) -> int:
        return _required_int(self.experiment, "batch_size")

    @property
    def context_length(self) -> int:
        return _required_int(self.experiment, "context_length")

    @property
    def max_iterations(self) -> int:
        return _required_int(self.experiment, "max_iterations")

    @property
    def total_training_tokens(self) -> int:
        return self.batch_size * self.context_length * self.max_iterations

    @property
    def validation_tokens_per_measurement(self) -> int:
        validation_batches = _required_int(self.experiment, "validation_batches")
        return self.batch_size * self.context_length * validation_batches


def _required_int(experiment: ExperimentLog, field: str) -> int:
    try:
        return int(experiment.config[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"every log must define integer {field}") from error


def parse_batch_run(log_path: Path, *, run_kind: str = "full") -> BatchRun:
    """Parse a log and retain its final reported interval throughput."""

    if run_kind not in {"full", "smoke"}:
        raise ValueError("run_kind must be 'full' or 'smoke'")

    experiment = parse_experiment_log(log_path)
    throughput_values: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        fields = dict(FIELD_PATTERN.findall(line))
        if "train_loss" in fields and "tokens_per_second" in fields:
            throughput_values.append(float(fields["tokens_per_second"]))
    if not throughput_values:
        raise ValueError(f"no throughput observations found in {log_path}")

    return BatchRun(
        source_path=log_path,
        experiment=experiment,
        final_tokens_per_second=throughput_values[-1],
        run_kind=run_kind,
    )


def _validate_controlled_fields(runs: list[BatchRun]) -> None:
    reference = runs[0].experiment.config
    for run in runs[1:]:
        mismatches = [field for field in CONTROLLED_FIELDS if run.experiment.config.get(field) != reference.get(field)]
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(f"batch logs differ in controlled fields: {joined}")


def load_batch_runs(
    full_log_paths: list[Path],
    smoke_log_path: Path,
) -> tuple[list[BatchRun], BatchRun]:
    """Load logs and verify the two comparison designs used by the figure."""

    full_runs = [parse_batch_run(path) for path in full_log_paths]
    if sorted(run.batch_size for run in full_runs) != list(FULL_BATCH_SIZES):
        raise ValueError(f"full logs must contain batch sizes {FULL_BATCH_SIZES}")
    full_runs.sort(key=lambda run: run.batch_size)

    smoke_run = parse_batch_run(smoke_log_path, run_kind="smoke")
    if smoke_run.batch_size != 256:
        raise ValueError("smoke log must use batch size 256")
    if smoke_run.max_iterations != 2:
        raise ValueError("batch-256 result must be the expected two-step smoke test")

    _validate_controlled_fields([*full_runs, smoke_run])
    by_batch = {run.batch_size: run for run in full_runs}

    fixed_token_runs = [by_batch[batch] for batch in FIXED_TOKEN_BATCH_SIZES]
    training_budgets = {run.total_training_tokens for run in fixed_token_runs}
    validation_budgets = {run.validation_tokens_per_measurement for run in fixed_token_runs}
    if len(training_budgets) != 1 or len(validation_budgets) != 1:
        raise ValueError("batch 1 and 8 must match in training tokens and validation tokens")

    fixed_update_runs = [by_batch[batch] for batch in FIXED_UPDATE_BATCH_SIZES]
    update_counts = {run.max_iterations for run in fixed_update_runs}
    if len(update_counts) != 1:
        raise ValueError("batch 8, 32, 64, and 128 must use the same update count")
    token_budgets = [run.total_training_tokens for run in fixed_update_runs]
    if len(set(token_budgets)) != len(token_budgets):
        raise ValueError("fixed-update runs must expose their unequal token budgets")

    return full_runs, smoke_run


def _tokens_label(tokens: int) -> str:
    return f"{tokens / 1_000_000:.2f}M"


def create_figure(full_runs: list[BatchRun], smoke_run: BatchRun) -> plt.Figure:
    """Create the three-panel batch-size systems and loss figure."""

    by_batch = {run.batch_size: run for run in full_runs}
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 3.5),
        gridspec_kw={"width_ratios": (0.9, 1.05, 1.45)},
    )
    throughput_axis, token_axis, update_axis = axes

    throughput_runs = [*full_runs, smoke_run]
    batch_labels = [str(run.batch_size) for run in throughput_runs]
    throughputs = [run.final_tokens_per_second / 1_000 for run in throughput_runs]
    bar_colors = [COLORS[run.batch_size] for run in full_runs] + ["#FFFFFF"]
    bars = throughput_axis.bar(
        batch_labels,
        throughputs,
        color=bar_colors,
        edgecolor=["none"] * len(full_runs) + ["#555555"],
        linewidth=[0.0] * len(full_runs) + [1.0],
        hatch=[None] * len(full_runs) + ["////"],
        width=0.72,
    )
    for bar, throughput in zip(bars, throughputs):
        throughput_axis.text(
            bar.get_x() + bar.get_width() / 2,
            throughput + 0.22,
            f"{throughput:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    throughput_axis.text(
        len(batch_labels) - 1,
        throughputs[-1] + 1.15,
        "2-step\nsmoke",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    throughput_axis.set_ylim(0, 12.6)
    throughput_axis.set_xlabel("Batch size")
    throughput_axis.set_ylabel("Final throughput (k tokens/s)")
    throughput_axis.set_title("Throughput plateaus by B=8")

    for batch_size in FIXED_TOKEN_BATCH_SIZES:
        run = by_batch[batch_size]
        validation = run.experiment.validation
        token_axis.plot(
            [point.iteration * run.batch_size * run.context_length / 1_000_000 for point in validation],
            [point.loss for point in validation],
            color=COLORS[batch_size],
            linewidth=1.6,
            marker="o" if batch_size == 1 else "s",
            markersize=3.5,
            markeredgewidth=0,
            label=f"B={batch_size}",
        )
    token_axis.set_xlabel("Processed training tokens (M)")
    token_axis.set_ylabel("Validation cross-entropy")
    token_axis.set_title("Controlled token budget")
    token_axis.legend(loc="upper right", fontsize=6.5)
    token_axis.text(
        0.04,
        0.08,
        "Both: 3.28M train tokens\n32,768 validation tokens/point",
        transform=token_axis.transAxes,
        fontsize=6.2,
        color="#4D4D4D",
        va="bottom",
    )

    for batch_size in FIXED_UPDATE_BATCH_SIZES:
        run = by_batch[batch_size]
        validation = run.experiment.validation
        update_axis.plot(
            [point.iteration for point in validation],
            [point.loss for point in validation],
            color=COLORS[batch_size],
            linewidth=1.5,
            marker="s",
            markersize=3.2,
            markeredgewidth=0,
            label=f"B={batch_size} · {_tokens_label(run.total_training_tokens)} tok",
        )
    update_axis.set_xlabel("Gradient updates")
    update_axis.set_ylabel("Validation cross-entropy")
    update_axis.set_title("Fixed updates, unequal data")
    update_axis.legend(loc="upper right", fontsize=6.1, handlelength=1.7)
    update_axis.text(
        0.04,
        0.08,
        "All: 1,600 updates\nTraining tokens differ 16×",
        transform=update_axis.transAxes,
        fontsize=6.2,
        color="#4D4D4D",
        va="bottom",
    )

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.17,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        axis.tick_params(width=0.8, length=3)
        axis.margins(x=0.04)

    seed = full_runs[0].experiment.config["seed"]
    figure.suptitle(
        "Batch 8 reaches the observed throughput ceiling; loss needs matched budgets",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.02,
        (
            f"TinyStories, MPS, seed={seed}; one run per setting, no error bars. "
            "Panel a uses the final logged interval (B=256 is a two-step smoke test). "
            "Panel c is not a fixed-token or fixed-compute comparison."
        ),
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.22,
        top=0.79,
        wspace=0.36,
    )
    return figure


def create_readme_figure(full_runs: list[BatchRun]) -> plt.Figure:
    """Create a concise two-panel figure for the public project README.

    The systems panel includes only completed runs. The optimization panel uses
    only batches 1 and 8 because they are the two runs with matched training and
    validation token budgets. The complete fixed-update view remains available
    through :func:`create_figure`.
    """

    by_batch = {run.batch_size: run for run in full_runs}
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    throughput_axis, loss_axis = axes

    batch_sizes = [run.batch_size for run in full_runs]
    throughputs = [run.final_tokens_per_second / 1_000 for run in full_runs]
    bar_colors = [README_SIGNAL_COLOR if batch_size == 8 else README_NEUTRAL_COLOR for batch_size in batch_sizes]
    bars = throughput_axis.bar(
        [str(batch_size) for batch_size in batch_sizes],
        throughputs,
        color=bar_colors,
        width=0.68,
    )
    for bar, throughput in zip(bars, throughputs):
        throughput_axis.text(
            bar.get_x() + bar.get_width() / 2,
            throughput + 0.18,
            f"{throughput:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if throughput == max(throughputs) else "normal",
            color="#333333",
        )
    throughput_axis.set_ylim(0, 12.0)
    throughput_axis.set_xlabel("Batch size", fontsize=11)
    throughput_axis.set_ylabel("Throughput (k tokens/s)", fontsize=11)
    throughput_axis.set_title("Training throughput on Apple MPS", fontsize=13)

    marker_by_batch = {1: "o", 8: "s"}
    for batch_size in FIXED_TOKEN_BATCH_SIZES:
        run = by_batch[batch_size]
        validation = run.experiment.validation
        color = README_SIGNAL_COLOR if batch_size == 8 else README_NEUTRAL_COLOR
        x_values = [point.iteration * run.batch_size * run.context_length / 1_000_000 for point in validation]
        losses = [point.loss for point in validation]
        loss_axis.plot(
            x_values,
            losses,
            color=color,
            linewidth=2.2,
            marker=marker_by_batch[batch_size],
            markersize=5.0,
            markeredgewidth=0,
        )
        loss_axis.text(
            x_values[-1] + 0.06,
            losses[-1],
            f"B={batch_size}  {losses[-1]:.2f}",
            color=color,
            fontsize=10,
            fontweight="bold" if batch_size == 8 else "normal",
            va="center",
        )
    loss_axis.set_xlim(0.55, 3.75)
    loss_axis.set_ylim(2.52, 3.83)
    loss_axis.set_xlabel("Training tokens processed (millions)", fontsize=11)
    loss_axis.set_ylabel("Validation cross-entropy", fontsize=11)
    loss_axis.set_title("Validation loss at equal token budget", fontsize=13)
    loss_axis.text(
        0.04,
        0.08,
        "Both runs: 3.28M tokens",
        transform=loss_axis.transAxes,
        fontsize=9,
        color="#555555",
    )

    for label, axis in zip(("a", "b"), axes):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.tick_params(width=0.8, length=3.5, labelsize=10)
        axis.margins(x=0.04)

    figure.subplots_adjust(
        left=0.075,
        right=0.965,
        bottom=0.17,
        top=0.84,
        wspace=0.30,
    )
    return figure


def write_source_data(
    output_path: Path,
    full_runs: list[BatchRun],
    smoke_run: BatchRun,
) -> None:
    """Write every observation plotted across the three panels."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "panel",
        "batch_size",
        "run_kind",
        "x_metric",
        "x_value",
        "y_metric",
        "y_value",
        "iteration",
        "total_training_tokens",
        "validation_tokens_per_measurement",
        "run_elapsed_seconds",
        "seed",
        "source_log",
    )
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for run in [*full_runs, smoke_run]:
            writer.writerow(
                _source_row(
                    run,
                    panel="a_throughput",
                    x_metric="batch_size",
                    x_value=run.batch_size,
                    y_metric="final_tokens_per_second",
                    y_value=run.final_tokens_per_second,
                    iteration=run.experiment.training[-1].iteration,
                )
            )

        by_batch = {run.batch_size: run for run in full_runs}
        for batch_size in FIXED_TOKEN_BATCH_SIZES:
            run = by_batch[batch_size]
            for point in run.experiment.validation:
                writer.writerow(
                    _source_row(
                        run,
                        panel="b_fixed_training_tokens",
                        x_metric="processed_training_tokens",
                        x_value=point.iteration * run.batch_size * run.context_length,
                        y_metric="validation_cross_entropy",
                        y_value=point.loss,
                        iteration=point.iteration,
                    )
                )

        for batch_size in FIXED_UPDATE_BATCH_SIZES:
            run = by_batch[batch_size]
            for point in run.experiment.validation:
                writer.writerow(
                    _source_row(
                        run,
                        panel="c_fixed_updates",
                        x_metric="gradient_updates",
                        x_value=point.iteration,
                        y_metric="validation_cross_entropy",
                        y_value=point.loss,
                        iteration=point.iteration,
                    )
                )


def _source_row(
    run: BatchRun,
    *,
    panel: str,
    x_metric: str,
    x_value: int | float,
    y_metric: str,
    y_value: float,
    iteration: int,
) -> dict[str, object]:
    return {
        "panel": panel,
        "batch_size": run.batch_size,
        "run_kind": run.run_kind,
        "x_metric": x_metric,
        "x_value": x_value,
        "y_metric": y_metric,
        "y_value": y_value,
        "iteration": iteration,
        "total_training_tokens": run.total_training_tokens,
        "validation_tokens_per_measurement": (run.validation_tokens_per_measurement),
        "run_elapsed_seconds": run.experiment.validation[-1].elapsed_seconds,
        "seed": run.experiment.config["seed"],
        "source_log": run.source_path.name,
    }


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


def save_readme_figure(figure: plt.Figure, output_prefix: Path) -> list[Path]:
    """Export the GitHub-facing figure as an editable SVG and compact PNG."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".png"),
    ]
    figure.savefig(output_paths[0], bbox_inches="tight")
    figure.savefig(output_paths[1], dpi=300, bbox_inches="tight")
    return output_paths


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot valid systems and optimization views of batch-size logs.")
    parser.add_argument("--full-logs", nargs="+", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--readme-prefix",
        type=Path,
        help="optional output path for the concise GitHub-facing SVG and PNG",
    )
    parser.add_argument(
        "--source-data-output",
        type=Path,
        help="optional public source-data path; defaults beside output-prefix",
    )
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    full_runs, smoke_run = load_batch_runs(args.full_logs, args.smoke_log)
    figure = create_figure(full_runs, smoke_run)
    output_paths = save_figure(figure, args.output_prefix)
    source_data_path = args.source_data_output or args.output_prefix.with_name(
        f"{args.output_prefix.name}_source_data.csv"
    )
    write_source_data(source_data_path, full_runs, smoke_run)
    plt.close(figure)

    readme_output_paths: list[Path] = []
    if args.readme_prefix is not None:
        readme_figure = create_readme_figure(full_runs)
        readme_output_paths = save_readme_figure(readme_figure, args.readme_prefix)
        plt.close(readme_figure)

    print(f"full runs: {len(full_runs)}")
    print("fixed-token comparison: batch 1 vs 8")
    print("fixed-update comparison: batch 8, 32, 64, 128")
    for output_path in [*output_paths, *readme_output_paths, source_data_path]:
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
