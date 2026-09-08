from dataclasses import replace

import numpy as np
import torch

from cs336_basics.training import TrainingConfig, load_token_array, train
from scripts.train import parse_args


def test_load_token_array_uses_memory_mapping(tmp_path):
    data_path = tmp_path / "tokens.npy"
    np.save(data_path, np.arange(20, dtype=np.uint16))

    token_ids = load_token_array(data_path, context_length=4)

    assert isinstance(token_ids, np.memmap)
    assert token_ids.shape == (20,)
    assert token_ids.dtype == np.uint16


def test_command_line_data_paths_match_training_config_fields(tmp_path):
    arguments = [
        "--train-data",
        str(tmp_path / "train.npy"),
        "--validation-data",
        str(tmp_path / "validation.npy"),
        "--checkpoint-path",
        str(tmp_path / "checkpoint.pt"),
        "--vocab-size",
        "10000",
    ]

    config = TrainingConfig(**vars(parse_args(arguments)))

    assert config.train_data_path == tmp_path / "train.npy"
    assert config.validation_data_path == tmp_path / "validation.npy"


def test_training_loop_saves_and_resumes_checkpoint(tmp_path):
    train_path = tmp_path / "train.npy"
    validation_path = tmp_path / "validation.npy"
    checkpoint_path = tmp_path / "nested" / "checkpoint.pt"
    token_ids = np.arange(64, dtype=np.uint16) % 16
    np.save(train_path, token_ids)
    np.save(validation_path, token_ids[::-1].copy())

    config = TrainingConfig(
        train_data_path=train_path,
        validation_data_path=validation_path,
        checkpoint_path=checkpoint_path,
        vocab_size=16,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        d_ff=16,
        batch_size=2,
        max_iterations=2,
        max_learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_iterations=0,
        cosine_cycle_iterations=4,
        log_interval=10,
        validation_interval=2,
        validation_batches=1,
        checkpoint_interval=1,
    )
    messages = []

    train(config, log=messages.append)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["iteration"] == 2
    assert any("train_loss=" in message for message in messages)
    assert any("validation_loss=" in message for message in messages)
    assert all(
        "elapsed_seconds=" in message
        for message in messages
        if "_loss=" in message
    )

    resumed_messages = []
    train(
        replace(config, max_iterations=3, resume_from=checkpoint_path),
        log=resumed_messages.append,
    )

    resumed_checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert resumed_checkpoint["iteration"] == 3
    assert any("resumed_from=" in message for message in resumed_messages)
