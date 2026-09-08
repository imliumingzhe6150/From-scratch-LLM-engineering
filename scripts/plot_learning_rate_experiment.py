"""Summarize a TinyStories learning-rate sweep and selected full run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.plot_training_log import ExperimentLog, parse_experiment_log, plt


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

SWEEP_COLORS = {
    3e-4: "#4477AA",
    1e-3: "#66A61E",
    3e-3: "#228833",
    1e-2: "#EE7733",
    3e-2: "#AA3377",
}
COMPARABILITY_FIELDS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "batch_size",
    "max_iterations",
    "min_learning_rate",
    "warmup_iterations",
    "cosine_cycle_iterations",
    "beta1",
    "beta2",
    "adam_epsilon",
    "weight_decay",
    "max_gradient_norm",
    "seed",
)


def _learning_rate(experiment: ExperimentLog) -> float:
    """Return the configured peak learning rate for one experiment."""

    try:
        return float(experiment.config["max_learning_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("every log must define max_learning_rate") from error


def load_sweep(log_paths: list[Path]) -> list[ExperimentLog]:
    """Parse and sort sweep logs, rejecting incomparable configurations."""

    if len(log_paths) < 2:
        raise ValueError("at least two sweep logs are required")

    experiments = [parse_experiment_log(path) for path in log_paths]
    reference = experiments[0].config
    for experiment in experiments[1:]:
        mismatches = [
            field
            for field in COMPARABILITY_FIELDS
            if experiment.config.get(field) != reference.get(field)
        ]
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(f"sweep logs differ in controlled fields: {joined}")

    learning_rates = [_learning_rate(experiment) for experiment in experiments]
    if len(set(learning_rates)) != len(learning_rates):
        raise ValueError("sweep learning rates must be unique")
    return sorted(experiments, key=_learning_rate)


def _lr_label(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("e-0", "e-")


def _color_for(learning_rate: float, index: int) -> str:
    return SWEEP_COLORS.get(learning_rate, plt.get_cmap("viridis")(index / 6))


def _plot_sweep_panel(
    axis: plt.Axes,
    experiments: list[ExperimentLog],
    *,
    split: str,
) -> None:
    """Plot unsmoothed losses for one split across all learning rates."""

    marker = "o" if split == "training" else "s"
    for index, experiment in enumerate(experiments):
        learning_rate = _learning_rate(experiment)
        points = getattr(experiment, split)
        axis.plot(
            [point.iteration for point in points],
            [point.loss for point in points],
            color=_color_for(learning_rate, index),
            linewidth=1.5,
            marker=marker,
            markersize=3.0,
            markeredgewidth=0,
            label=_lr_label(learning_rate),
        )


def create_figure(
    sweep: list[ExperimentLog],
    full_run: ExperimentLog,
) -> plt.Figure:
    """Create the three-panel learning-rate experiment figure."""

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.25))
    train_axis, validation_axis, full_axis = axes

    _plot_sweep_panel(train_axis, sweep, split="training")
    train_axis.set_title("Short-run optimization")
    train_axis.set_xlabel("Gradient step")
    train_axis.set_ylabel("Training cross-entropy")
    highest_lr_run = max(sweep, key=_learning_rate)
    spike = max(highest_lr_run.training, key=lambda point: point.loss)
    train_axis.annotate(
        "Transient loss spike",
        xy=(spike.iteration, spike.loss),
        xytext=(12, -14),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 0.7},
        fontsize=6.5,
        color="#4D4D4D",
    )

    _plot_sweep_panel(validation_axis, sweep, split="validation")
    validation_axis.set_title("Held-out comparison")
    validation_axis.set_xlabel("Gradient step")
    validation_axis.set_ylabel("Validation cross-entropy")

    selected_lr = _learning_rate(full_run)
    full_axis.plot(
        [point.iteration for point in full_run.training],
        [point.loss for point in full_run.training],
        color="#0F4D92",
        linewidth=1.5,
        label="Train",
    )
    full_axis.plot(
        [point.iteration for point in full_run.validation],
        [point.loss for point in full_run.validation],
        color="#B64342",
        linewidth=1.6,
        marker="s",
        markersize=3.2,
        markeredgewidth=0,
        label="Validation",
    )
    full_axis.axhline(
        2.0,
        color="#666666",
        linewidth=1.0,
        linestyle="--",
        label="MPS target (2.00)",
    )
    final_validation = full_run.validation[-1]
    full_axis.annotate(
        f"{final_validation.loss:.3f}",
        xy=(final_validation.iteration, final_validation.loss),
        xytext=(-7, 9),
        textcoords="offset points",
        ha="right",
        color="#B64342",
        fontsize=7,
        fontweight="bold",
    )
    full_axis.set_title(f"Selected LR {_lr_label(selected_lr)}")
    full_axis.set_xlabel("Gradient step")
    full_axis.set_ylabel("Cross-entropy loss")
    full_axis.legend(loc="upper right", fontsize=6.5)

    handles, labels = train_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="Peak learning rate",
        loc="upper center",
        bbox_to_anchor=(0.36, 0.84),
        ncol=len(labels),
        fontsize=6.5,
        title_fontsize=6.5,
        handlelength=2.0,
        columnspacing=1.1,
    )

    for label, axis in zip(("a", "b", "c"), axes):
        axis.text(
            -0.16,
            1.06,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
        axis.tick_params(width=0.8, length=3)
        axis.margins(x=0.03)

    batch_size = int(full_run.config["batch_size"])
    context_length = int(full_run.config["context_length"])
    final_step = full_run.training[-1].iteration
    tokens_millions = batch_size * context_length * final_step / 1_000_000
    validation_batches = full_run.config.get("validation_batches", "unknown")
    seed = full_run.config.get("seed", "unknown")
    figure.suptitle(
        "Short-run screening favors a peak learning rate of 3e-3 for TinyStories",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.025,
        (
            "Short runs: 100 steps per learning rate; full run: "
            f"{tokens_millions:.2f}M tokens. Train curves are logging-interval means; "
            f"validation uses {validation_batches} sampled batches; seed={seed}; "
            "one run per setting; no smoothing."
        ),
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="#4D4D4D",
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.22,
        top=0.66,
        wspace=0.30,
    )
    return figure


def write_source_data(
    output_path: Path,
    sweep: list[ExperimentLog],
    full_run: ExperimentLog,
) -> None:
    """Write every plotted observation to a long-form CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "phase",
                "peak_learning_rate",
                "split",
                "iteration",
                "loss",
                "elapsed_seconds",
            ),
        )
        writer.writeheader()
        for phase, experiments in (("sweep", sweep), ("full", [full_run])):
            for experiment in experiments:
                learning_rate = _learning_rate(experiment)
                for split in ("training", "validation"):
                    for point in getattr(experiment, split):
                        writer.writerow(
                            {
                                "phase": phase,
                                "peak_learning_rate": learning_rate,
                                "split": split,
                                "iteration": point.iteration,
                                "loss": point.loss,
                                "elapsed_seconds": point.elapsed_seconds,
                            }
                        )


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
        description="Plot a learning-rate sweep and its selected full run."
    )
    parser.add_argument("--sweep-logs", nargs="+", type=Path, required=True)
    parser.add_argument("--full-log", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    sweep = load_sweep(args.sweep_logs)
    full_run = parse_experiment_log(args.full_log)
    figure = create_figure(sweep, full_run)
    output_paths = save_figure(figure, args.output_prefix)
    source_data_path = args.output_prefix.with_name(
        f"{args.output_prefix.name}_source_data.csv"
    )
    write_source_data(source_data_path, sweep, full_run)
    plt.close(figure)

    print(f"sweep runs: {len(sweep)}")
    print(f"selected peak learning rate: {_learning_rate(full_run):g}")
    print(f"final validation loss: {full_run.validation[-1].loss:.6f}")
    for output_path in [*output_paths, source_data_path]:
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
