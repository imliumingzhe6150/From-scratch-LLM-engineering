import csv

import pytest

from scripts import plot_batch_size_experiment


def _write_log(
    path,
    batch_size,
    max_iterations,
    validation_batches,
    *,
    context_length=4,
):
    observation_count = min(5, max_iterations)
    lines = [
        "{",
        '  "vocab_size": 100,',
        f'  "batch_size": {batch_size},',
        f'  "context_length": {context_length},',
        f'  "max_iterations": {max_iterations},',
        f'  "validation_batches": {validation_batches},',
        '  "d_model": 16,',
        '  "num_layers": 2,',
        '  "num_heads": 2,',
        '  "d_ff": 32,',
        '  "rope_theta": 10000.0,',
        '  "max_learning_rate": 0.003,',
        '  "min_learning_rate": 0.00003,',
        '  "beta1": 0.9,',
        '  "beta2": 0.95,',
        '  "adam_epsilon": 1e-8,',
        '  "weight_decay": 0.1,',
        '  "max_gradient_norm": 1.0,',
        '  "seed": 42,',
        '  "train_data_path": "train.npy",',
        '  "validation_data_path": "valid.npy",',
        '  "device": "mps"',
        "}",
    ]
    for index in range(1, observation_count + 1):
        iteration = max_iterations * index // observation_count
        elapsed = index * batch_size
        lines.append(
            f"iteration={iteration} train_loss={5 - index / 2:.3f} "
            f"tokens_per_second={9000 + batch_size} elapsed_seconds={elapsed:.1f}"
        )
        lines.append(f"iteration={iteration} validation_loss={5 - index / 2:.3f} elapsed_seconds={elapsed + 0.1:.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_complete_set(tmp_path):
    paths = []
    for batch_size, max_iterations, validation_batches in (
        (1, 1280, 128),
        (8, 160, 16),
        (32, 160, 16),
        (64, 160, 16),
        (128, 160, 16),
    ):
        path = tmp_path / f"batch_{batch_size}.log"
        _write_log(path, batch_size, max_iterations, validation_batches)
        paths.append(path)
    smoke_path = tmp_path / "batch_256_smoke.log"
    _write_log(smoke_path, 256, 2, 1)
    return paths, smoke_path


def test_load_batch_runs_checks_comparison_designs(tmp_path):
    paths, smoke_path = _write_complete_set(tmp_path)

    full_runs, smoke_run = plot_batch_size_experiment.load_batch_runs(list(reversed(paths)), smoke_path)

    assert [run.batch_size for run in full_runs] == [1, 8, 32, 64, 128]
    assert full_runs[0].total_training_tokens == full_runs[1].total_training_tokens
    assert full_runs[0].validation_tokens_per_measurement == 512
    assert full_runs[1].validation_tokens_per_measurement == 512
    assert {run.max_iterations for run in full_runs[1:]} == {160}
    assert smoke_run.run_kind == "smoke"
    assert smoke_run.final_tokens_per_second == 9256


def test_load_batch_runs_rejects_invalid_fixed_token_comparison(tmp_path):
    paths, smoke_path = _write_complete_set(tmp_path)
    _write_log(paths[0], 1, 1280, 127)

    with pytest.raises(ValueError, match="training tokens and validation tokens"):
        plot_batch_size_experiment.load_batch_runs(paths, smoke_path)


def test_figure_and_source_data_include_every_plotted_point(tmp_path):
    paths, smoke_path = _write_complete_set(tmp_path)
    full_runs, smoke_run = plot_batch_size_experiment.load_batch_runs(paths, smoke_path)

    figure = plot_batch_size_experiment.create_figure(full_runs, smoke_run)
    source_path = tmp_path / "batch_source_data.csv"
    plot_batch_size_experiment.write_source_data(source_path, full_runs, smoke_run)

    assert len(figure.axes) == 3
    assert len(figure.axes[0].patches) == 6
    assert len(figure.axes[1].lines) == 2
    assert len(figure.axes[2].lines) == 4
    with source_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 36
    assert {row["panel"] for row in rows} == {
        "a_throughput",
        "b_fixed_training_tokens",
        "c_fixed_updates",
    }
    batch_128_rows = [row for row in rows if row["panel"] == "c_fixed_updates" and row["batch_size"] == "128"]
    assert {row["total_training_tokens"] for row in batch_128_rows} == {str(128 * 4 * 160)}
    assert {row["run_elapsed_seconds"] for row in batch_128_rows} == {"640.1"}
    plot_batch_size_experiment.plt.close(figure)


def test_readme_figure_keeps_only_completed_throughput_and_matched_token_loss(tmp_path):
    paths, smoke_path = _write_complete_set(tmp_path)
    full_runs, _ = plot_batch_size_experiment.load_batch_runs(paths, smoke_path)

    figure = plot_batch_size_experiment.create_readme_figure(full_runs)

    assert len(figure.axes) == 2
    assert len(figure.axes[0].patches) == 5
    assert len(figure.axes[1].lines) == 2
    assert figure.axes[0].get_title() == "Training throughput on Apple MPS"
    assert figure.axes[1].get_title() == "Validation loss at equal token budget"
    plot_batch_size_experiment.plt.close(figure)
