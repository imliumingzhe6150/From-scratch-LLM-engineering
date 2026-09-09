import argparse

import pytest

from scripts.benchmark_kv_cache import (
    BenchmarkCase,
    benchmark_cases,
    summarize_measurements,
)
from tests.test_kv_cache import make_model


def test_benchmark_case_parsing_and_validation():
    assert BenchmarkCase.parse("16:32") == BenchmarkCase(16, 32)

    with pytest.raises(argparse.ArgumentTypeError, match="PROMPT_TOKENS"):
        BenchmarkCase.parse("16")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        BenchmarkCase.parse("0:4")


def test_benchmark_records_both_modes_and_cache_memory():
    model = make_model(context_length=8)
    measurements = benchmark_cases(
        model,
        [BenchmarkCase(prompt_tokens=3, generated_tokens=4)],
        warmup_repeats=0,
        measured_repeats=2,
        seed=11,
    )
    summaries = summarize_measurements(measurements)

    assert len(measurements) == 4
    assert {measurement.mode for measurement in measurements} == {
        "cached",
        "uncached",
    }
    assert all(measurement.total_seconds > 0 for measurement in measurements)
    assert all(measurement.tokens_per_second > 0 for measurement in measurements)
    assert len(summaries) == 2
    by_mode = {str(row["mode"]): row for row in summaries}
    assert by_mode["cached"]["peak_cache_bytes"] > 0
    assert by_mode["uncached"]["peak_cache_bytes"] == 0


def test_benchmark_rejects_prompt_beyond_context():
    model = make_model(context_length=4)

    with pytest.raises(ValueError, match="prompt length 5"):
        benchmark_cases(
            model,
            [BenchmarkCase(prompt_tokens=5, generated_tokens=1)],
            warmup_repeats=0,
            measured_repeats=1,
            seed=1,
        )
