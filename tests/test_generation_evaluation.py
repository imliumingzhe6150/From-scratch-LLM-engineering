import argparse
import json

import pytest
import torch
from torch import nn

from cs336_basics.generation_evaluation import (
    ArtifactIdentity,
    DecodingPolicy,
    evaluate_generation,
    identify_artifact,
    write_generation_evaluation,
)
from scripts.evaluate_generation import DEFAULT_POLICIES, parse_policy


class IntegerTokenizer:
    """Encode space-separated integers and decode them without hidden state."""

    def encode(self, text: str) -> list[int]:
        return [int(piece) for piece in text.split()] if text else []

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in ids)


class TransitionLanguageModel(nn.Module):
    """Choose token 2 after token 1, then choose the EOS token 4."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = 5
        self.context_length = 4

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*token_ids.shape, self.vocab_size)
        next_tokens = torch.where(token_ids[:, -1] == 1, 2, 4)
        logits[torch.arange(token_ids.shape[0]), -1, next_tokens] = 10.0
        return logits


def test_generation_evaluation_records_eos_and_reproducibility_metadata():
    evaluation = evaluate_generation(
        TransitionLanguageModel(),
        IntegerTokenizer(),
        prompt="0 1",
        policies=[DecodingPolicy("greedy", temperature=0.0, top_p=1.0)],
        max_new_tokens=8,
        seed=17,
        end_token_id=4,
        model_config={"vocab_size": 5, "context_length": 4},
        artifacts={
            "checkpoint": ArtifactIdentity("model.pt", 123, "abc"),
        },
        created_at_utc="2026-09-08T00:00:00+00:00",
    )

    run = evaluation.runs[0]
    assert evaluation.schema_version == 1
    assert evaluation.prompt_token_ids == [0, 1]
    assert evaluation.prompt_token_count == 2
    assert evaluation.model_prompt_token_ids == [0, 1]
    assert evaluation.model_prompt_token_count == 2
    assert evaluation.max_new_tokens == 8
    assert evaluation.seed_policy == "reset_same_seed_before_each_policy"
    assert evaluation.device == "cpu"
    assert run.seed == 17
    assert run.generated_token_ids == [2, 4]
    assert run.generated_token_count == 2
    assert run.visible_generated_token_count == 1
    assert run.stopping_reason == "end_of_text"
    assert run.completion_text == "2"
    assert run.full_text == "0 1 2"


def test_generation_evaluation_records_length_stop_and_empty_prompt_context():
    evaluation = evaluate_generation(
        TransitionLanguageModel(),
        IntegerTokenizer(),
        prompt="",
        policies=[DecodingPolicy("greedy", temperature=0.0, top_p=1.0)],
        max_new_tokens=0,
        seed=7,
        end_token_id=4,
        model_config={},
        artifacts={},
    )

    run = evaluation.runs[0]
    assert evaluation.prompt_token_ids == []
    assert evaluation.prompt_token_count == 0
    assert evaluation.model_prompt_token_ids == [4]
    assert evaluation.model_prompt_token_count == 1
    assert run.generated_token_ids == []
    assert run.stopping_reason == "max_new_tokens"
    assert run.full_text == ""


def test_evaluation_writes_json_and_hashes_artifacts(tmp_path):
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"abc")
    identity = identify_artifact(artifact_path)
    assert identity.size_bytes == 3
    assert identity.sha256 == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    evaluation = evaluate_generation(
        TransitionLanguageModel(),
        IntegerTokenizer(),
        prompt="0 1",
        policies=[DecodingPolicy("greedy", temperature=0.0, top_p=1.0)],
        max_new_tokens=2,
        seed=17,
        end_token_id=4,
        model_config={"vocab_size": 5},
        artifacts={"checkpoint": identity},
    )
    output_path = write_generation_evaluation(
        evaluation,
        tmp_path / "evaluation.json",
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["artifacts"]["checkpoint"]["sha256"] == identity.sha256
    assert payload["runs"][0]["policy"]["name"] == "greedy"
    assert not output_path.with_suffix(".json.tmp").exists()


def test_policy_parsing_defaults_and_validation():
    assert parse_policy("sample:0.8:0.9") == DecodingPolicy("sample", 0.8, 0.9)
    assert [policy.name for policy in DEFAULT_POLICIES] == [
        "greedy",
        "conservative",
        "diverse",
    ]

    with pytest.raises(ValueError, match="unique"):
        evaluate_generation(
            TransitionLanguageModel(),
            IntegerTokenizer(),
            prompt="0 1",
            policies=[
                DecodingPolicy("same", 0.0, 1.0),
                DecodingPolicy("same", 0.8, 0.9),
            ],
            max_new_tokens=1,
            seed=1,
            end_token_id=4,
            model_config={},
            artifacts={},
        )

    with pytest.raises(ValueError, match="top_p"):
        DecodingPolicy("bad", temperature=1.0, top_p=0.0)

    with pytest.raises(argparse.ArgumentTypeError, match="NAME:TEMPERATURE:TOP_P"):
        parse_policy("missing-fields")
