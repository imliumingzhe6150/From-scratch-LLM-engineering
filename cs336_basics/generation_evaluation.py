"""Structured, reproducible records for autoregressive generation quality."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import torch

from cs336_basics.generation import generate_tokens
from cs336_basics.model import TransformerLM

SCHEMA_VERSION = 1


class TokenizerLike(Protocol):
    """Minimal tokenizer interface required by generation evaluation."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class DecodingPolicy:
    """One named temperature and nucleus-sampling configuration."""

    name: str
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("policy name must not be empty")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity for one local input artifact."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GenerationRun:
    """The complete result of applying one decoding policy to one prompt."""

    policy: DecodingPolicy
    seed: int
    generated_token_ids: list[int]
    generated_token_count: int
    visible_generated_token_count: int
    stopping_reason: str
    completion_text: str
    full_text: str


@dataclass(frozen=True)
class GenerationEvaluation:
    """Serializable metadata and outputs for a generation comparison."""

    schema_version: int
    created_at_utc: str
    prompt: str
    prompt_token_ids: list[int]
    prompt_token_count: int
    model_prompt_token_ids: list[int]
    model_prompt_token_count: int
    max_new_tokens: int
    end_token_id: int
    seed_policy: str
    device: str
    torch_version: str
    model_config: dict[str, int | float]
    artifacts: dict[str, ArtifactIdentity]
    runs: list[GenerationRun]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of the full evaluation."""

        return asdict(self)


def identify_artifact(path: str | Path) -> ArtifactIdentity:
    """Return the path, byte size, and SHA-256 digest of a local artifact."""

    artifact_path = Path(path)
    digest = hashlib.sha256()
    with artifact_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return ArtifactIdentity(
        path=str(artifact_path),
        size_bytes=artifact_path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def evaluate_generation(
    model: TransformerLM,
    tokenizer: TokenizerLike,
    *,
    prompt: str,
    policies: list[DecodingPolicy],
    max_new_tokens: int,
    seed: int,
    end_token_id: int,
    model_config: dict[str, int | float],
    artifacts: dict[str, ArtifactIdentity],
    created_at_utc: str | None = None,
) -> GenerationEvaluation:
    """Generate once per policy and retain the inputs needed to reproduce it.

    ``generated_token_count`` includes a sampled end token, while
    ``visible_generated_token_count`` and both text fields exclude that marker.
    Every policy starts from the same random seed so the seed policy is explicit
    and does not depend on the order in which policies are evaluated.
    """

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if not policies:
        raise ValueError("at least one decoding policy is required")
    policy_names = [policy.name for policy in policies]
    if len(policy_names) != len(set(policy_names)):
        raise ValueError("decoding policy names must be unique")

    prompt_token_ids = tokenizer.encode(prompt)
    model_prompt_token_ids = prompt_token_ids or [end_token_id]
    runs: list[GenerationRun] = []

    for policy in policies:
        torch.manual_seed(seed)
        generated_token_ids = generate_tokens(
            model,
            model_prompt_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=policy.temperature,
            top_p=policy.top_p,
            end_token_id=end_token_id,
        )
        stopped_at_end = bool(generated_token_ids and generated_token_ids[-1] == end_token_id)
        visible_token_ids = generated_token_ids[:-1] if stopped_at_end else generated_token_ids
        runs.append(
            GenerationRun(
                policy=policy,
                seed=seed,
                generated_token_ids=generated_token_ids,
                generated_token_count=len(generated_token_ids),
                visible_generated_token_count=len(visible_token_ids),
                stopping_reason=("end_of_text" if stopped_at_end else "max_new_tokens"),
                completion_text=tokenizer.decode(visible_token_ids),
                full_text=tokenizer.decode(prompt_token_ids + visible_token_ids),
            )
        )

    if created_at_utc is None:
        created_at_utc = datetime.now(UTC).isoformat()

    device = str(next(model.parameters()).device)
    return GenerationEvaluation(
        schema_version=SCHEMA_VERSION,
        created_at_utc=created_at_utc,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        prompt_token_count=len(prompt_token_ids),
        model_prompt_token_ids=model_prompt_token_ids,
        model_prompt_token_count=len(model_prompt_token_ids),
        max_new_tokens=max_new_tokens,
        end_token_id=end_token_id,
        seed_policy="reset_same_seed_before_each_policy",
        device=device,
        torch_version=torch.__version__,
        model_config=dict(model_config),
        artifacts=dict(artifacts),
        runs=runs,
    )


def write_generation_evaluation(
    evaluation: GenerationEvaluation,
    output_path: str | Path,
) -> Path:
    """Write an evaluation as stable, human-readable JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(evaluation.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
