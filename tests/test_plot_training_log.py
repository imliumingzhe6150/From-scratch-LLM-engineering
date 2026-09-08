from scripts import plot_training_log


def test_parse_experiment_log_preserves_all_loss_points(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        """{
  "log_interval": 10,
  "validation_batches": 5,
  "seed": 42
}
iteration=10 train_loss=8.0 lr=1e-4 tokens_per_second=100 elapsed_seconds=1.0
iteration=20 train_loss=7.0 lr=1e-4 tokens_per_second=100 elapsed_seconds=2.0
iteration=20 validation_loss=7.2 elapsed_seconds=2.5
iteration=40 validation_loss=6.5 elapsed_seconds=4.5
""",
        encoding="utf-8",
    )

    experiment = plot_training_log.parse_experiment_log(log_path)

    assert experiment.config["seed"] == 42
    assert [point.iteration for point in experiment.training] == [10, 20]
    assert [point.loss for point in experiment.training] == [8.0, 7.0]
    assert [point.elapsed_seconds for point in experiment.validation] == [2.5, 4.5]


def test_create_figure_contains_step_and_wall_clock_panels(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        """iteration=10 train_loss=8.0 elapsed_seconds=1.0
iteration=20 train_loss=7.0 elapsed_seconds=2.0
iteration=10 validation_loss=8.2 elapsed_seconds=1.2
iteration=20 validation_loss=7.2 elapsed_seconds=2.2
""",
        encoding="utf-8",
    )
    experiment = plot_training_log.parse_experiment_log(log_path)

    figure = plot_training_log.create_figure(experiment)

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xlabel() == "Gradient step"
    assert figure.axes[1].get_xlabel() == "Wall-clock time (seconds)"
    assert all(len(axis.lines) == 2 for axis in figure.axes)
    plot_training_log.plt.close(figure)
