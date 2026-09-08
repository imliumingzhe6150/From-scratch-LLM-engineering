import csv

import pytest

from scripts import plot_learning_rate_experiment


def _write_log(path, learning_rate, *, batch_size=32):
    path.write_text(
        f'''{{
  "batch_size": {batch_size},
  "context_length": 256,
  "max_iterations": 100,
  "max_learning_rate": {learning_rate},
  "min_learning_rate": 3e-5,
  "warmup_iterations": 10,
  "cosine_cycle_iterations": 100,
  "seed": 42
}}
iteration=10 train_loss=8.0 elapsed_seconds=1.0
iteration=20 train_loss=7.0 elapsed_seconds=2.0
iteration=20 validation_loss=7.2 elapsed_seconds=2.2
iteration=40 validation_loss=6.5 elapsed_seconds=4.2
''',
        encoding="utf-8",
    )


def test_load_sweep_sorts_learning_rates_and_checks_controls(tmp_path):
    high_path = tmp_path / "high.log"
    low_path = tmp_path / "low.log"
    _write_log(high_path, 1e-2)
    _write_log(low_path, 1e-3)

    sweep = plot_learning_rate_experiment.load_sweep([high_path, low_path])

    assert [run.config["max_learning_rate"] for run in sweep] == [1e-3, 1e-2]

    mismatch_path = tmp_path / "mismatch.log"
    _write_log(mismatch_path, 3e-3, batch_size=64)
    with pytest.raises(ValueError, match="batch_size"):
        plot_learning_rate_experiment.load_sweep([low_path, mismatch_path])


def test_figure_and_source_data_include_all_runs(tmp_path):
    first_path = tmp_path / "first.log"
    second_path = tmp_path / "second.log"
    full_path = tmp_path / "full.log"
    _write_log(first_path, 1e-3)
    _write_log(second_path, 3e-3)
    _write_log(full_path, 3e-3)
    sweep = plot_learning_rate_experiment.load_sweep([first_path, second_path])
    full_run = plot_learning_rate_experiment.parse_experiment_log(full_path)

    figure = plot_learning_rate_experiment.create_figure(sweep, full_run)
    source_path = tmp_path / "source.csv"
    plot_learning_rate_experiment.write_source_data(source_path, sweep, full_run)

    assert len(figure.axes) == 3
    assert len(figure.axes[0].lines) == 2
    assert len(figure.axes[1].lines) == 2
    assert len(figure.axes[2].lines) == 3
    with source_path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 12
    assert {row["phase"] for row in rows} == {"sweep", "full"}
    plot_learning_rate_experiment.plt.close(figure)
