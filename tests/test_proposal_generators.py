"""Unit tests for proposal_generators package."""

from __future__ import annotations

import pytest

from proposal_generators import get_generator
from proposal_generators.heuristic_generator import HeuristicProposalGenerator
from proposal_generators.random_generator import RandomProposalGenerator


@pytest.fixture
def knobs() -> dict[str, dict[str, int | float | str]]:
    return {
        "shared_buffers": {
            "type": "integer",
            "min": 16,
            "max": 2359296,
            "default": 16384,
            "step": 8,
        },
        "work_mem": {
            "type": "integer",
            "min": 64,
            "max": 2359296,
            "default": 4096,
            "step": 64,
        },
        "effective_cache_size": {
            "type": "integer",
            "min": 1,
            "max": 2359296,
            "default": 524288,
            "step": 1,
        },
        "maintenance_work_mem": {
            "type": "integer",
            "min": 64,
            "max": 2359296,
            "default": 65536,
            "step": 1024,
        },
        "checkpoint_completion_target": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "step": 0.01,
        },
    }


def _assert_config_within_bounds(
    proposals: list[dict[str, int | float]],
    knobs: dict[str, dict[str, int | float | str]],
) -> None:
    for proposal in proposals:
        for param, value in proposal.items():
            assert knobs[param]["min"] <= value <= knobs[param]["max"]


def test_random_uniform_count(knobs):
    result = RandomProposalGenerator(strategy="uniform", seed=42).generate({}, [], knobs, n=5)
    assert len(result) == 5


def test_random_uniform_all_keys_present(knobs):
    result = RandomProposalGenerator(strategy="uniform", seed=42).generate({}, [], knobs, n=3)
    expected_keys = set(knobs.keys())
    for proposal in result:
        assert set(proposal.keys()) == expected_keys


def test_random_uniform_within_bounds(knobs):
    result = RandomProposalGenerator(strategy="uniform", seed=42).generate({}, [], knobs, n=20)
    _assert_config_within_bounds(result, knobs)


def test_random_uniform_integer_type(knobs):
    result = RandomProposalGenerator(strategy="uniform", seed=42).generate({}, [], knobs, n=10)
    integer_keys = {
        name for name, detail in knobs.items() if detail["type"] == "integer"
    }
    for proposal in result:
        for key in integer_keys:
            assert isinstance(proposal[key], int)


def test_random_lhs_count(knobs):
    result = RandomProposalGenerator(strategy="lhs", seed=42).generate({}, [], knobs, n=10)
    assert len(result) == 10


def test_random_lhs_within_bounds(knobs):
    result = RandomProposalGenerator(strategy="lhs", seed=42).generate({}, [], knobs, n=10)
    _assert_config_within_bounds(result, knobs)


def test_random_invalid_strategy_raises():
    with pytest.raises(ValueError):
        RandomProposalGenerator(strategy="bad")


def test_random_reproducible(knobs):
    first = RandomProposalGenerator(strategy="uniform", seed=0).generate({}, [], knobs, n=1)
    second = RandomProposalGenerator(strategy="uniform", seed=0).generate({}, [], knobs, n=1)
    assert first[0] == second[0]


def test_random_zero_n(knobs):
    result = RandomProposalGenerator(strategy="uniform", seed=0).generate({}, [], knobs, n=0)
    assert result == []


def test_heuristic_read_heavy(knobs):
    features = {"read_ratio": 95.0, "join_count": 3, "aggregation_count": 2}
    result = HeuristicProposalGenerator(seed=42).generate(features, [], knobs, n=1)
    assert result[0]["shared_buffers"] > knobs["shared_buffers"]["default"]


def test_heuristic_join_heavy(knobs):
    features = {"read_ratio": 50.0, "join_count": 30, "aggregation_count": 2}
    result = HeuristicProposalGenerator(seed=42).generate(features, [], knobs, n=1)
    assert result[0]["work_mem"] > knobs["work_mem"]["default"]


def test_heuristic_agg_heavy(knobs):
    features = {"read_ratio": 50.0, "join_count": 5, "aggregation_count": 20}
    result = HeuristicProposalGenerator(seed=42).generate(features, [], knobs, n=1)
    assert result[0]["maintenance_work_mem"] > knobs["maintenance_work_mem"]["default"]


def test_heuristic_within_bounds(knobs):
    generator = HeuristicProposalGenerator(seed=42)
    feature_sets = [
        {"read_ratio": 95.0, "join_count": 3, "aggregation_count": 2},
        {"read_ratio": 50.0, "join_count": 30, "aggregation_count": 2},
        {"read_ratio": 50.0, "join_count": 5, "aggregation_count": 20},
        {"read_ratio": 10.0, "join_count": 1, "aggregation_count": 0},
    ]

    proposals: list[dict[str, int | float]] = []
    for features in feature_sets:
        proposals.extend(generator.generate(features, [], knobs, n=10))

    _assert_config_within_bounds(proposals, knobs)


def test_heuristic_zero_n(knobs):
    result = HeuristicProposalGenerator(seed=0).generate({}, [], knobs, n=0)
    assert result == []


def test_validate_proposal_clamps(knobs):
    generator = RandomProposalGenerator(seed=0)
    result = generator.validate_proposal({"shared_buffers": 9_999_999}, knobs)
    assert result["shared_buffers"] == 2359296


def test_factory_random(knobs):
    gen = get_generator("random", knobs_detail=knobs, seed=0)
    assert gen.name == "random"


def test_factory_lhs(knobs):
    gen = get_generator("lhs", knobs_detail=knobs, seed=0)
    assert gen.name == "lhs"


def test_factory_heuristic(knobs):
    gen = get_generator("heuristic", knobs_detail=knobs, seed=0)
    assert gen.name == "heuristic"
