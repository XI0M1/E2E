"""Unit tests for SMACProposalGenerator."""

from __future__ import annotations

import pytest

pytest.importorskip("smac")

from proposal_generators.smac_generator import SMACProposalGenerator


@pytest.fixture
def knobs():
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
        "checkpoint_completion_target": {
            "type": "float",
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "step": 0.01,
        },
    }


@pytest.fixture
def valid_config(knobs):
    return {key: value["default"] for key, value in knobs.items()}


def _assert_within_bounds(proposals, knobs):
    for proposal in proposals:
        for param, value in proposal.items():
            assert knobs[param]["min"] <= value <= knobs[param]["max"]


def test_name(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    assert gen.name == "smac"


def test_generate_count(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    proposals = gen.generate({}, [], knobs, n=1)
    assert len(proposals) == 1


def test_generate_all_knobs_present(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    proposal = gen.generate({}, [], knobs, n=1)[0]
    assert set(proposal.keys()) == set(knobs.keys())


def test_generate_within_bounds(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    proposals = gen.generate({}, [], knobs, n=3)
    _assert_within_bounds(proposals, knobs)


def test_generate_integer_types(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    proposal = gen.generate({}, [], knobs, n=1)[0]
    for key, detail in knobs.items():
        if detail["type"] == "integer":
            assert isinstance(proposal[key], int)


def test_generate_zero_n(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    assert gen.generate({}, [], knobs, n=0) == []


def test_tell_adds_to_runhistory(tmp_path, knobs, valid_config):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    initial = len(gen.runhistory)
    gen.tell(valid_config, tps=100.0)
    assert len(gen.runhistory) == initial + 1


def test_tell_deduplicates(tmp_path, knobs, valid_config):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    gen.tell(valid_config, 100.0)
    gen.tell(valid_config, 200.0)
    assert len(gen.runhistory) == 1


def test_tell_different_configs_adds_both(tmp_path, knobs, valid_config):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    config2 = {**valid_config, "shared_buffers": valid_config["shared_buffers"] + 8}
    gen.tell(valid_config, 100.0)
    gen.tell(config2, 200.0)
    assert len(gen.runhistory) == 2


def test_generate_uses_history(tmp_path, knobs, valid_config):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    gen.generate({}, [{"config": valid_config, "tps": 300.0}], knobs, n=1)
    assert len(gen.runhistory) >= 1


def test_save_load_roundtrip(tmp_path, knobs, valid_config):
    state_path = tmp_path / "state.pkl"
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    gen.tell(valid_config, 100.0)
    gen.save_state(str(state_path))

    gen2 = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s2"), seed=42)
    loaded = gen2.load_state(str(state_path))

    assert loaded is True
    assert len(gen2.runhistory) >= 1


def test_load_state_returns_false_missing(tmp_path, knobs):
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    assert gen.load_state(str(tmp_path / "nonexistent.pkl")) is False


def test_load_state_handles_corruption(tmp_path, knobs):
    corrupt_path = tmp_path / "corrupt.pkl"
    corrupt_path.write_bytes(b"CORRUPTED_DATA_XYZ")

    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    assert gen.load_state(str(corrupt_path)) is False
