"""Unit tests for SMACProposalGenerator."""

from __future__ import annotations

import json

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


def test_true_resume_same_output_dir(tmp_path, knobs, valid_config):
    """Validate real resume behavior when a new generator starts in the same output directory."""
    output_dir = tmp_path / "smac_run"
    gen1 = SMACProposalGenerator(knobs, output_dir=str(output_dir), seed=42)
    gen1.tell(valid_config, 100.0)
    expected_key = gen1._history_key(valid_config)
    gen1.save_state(gen1.state_path)

    del gen1

    gen2 = SMACProposalGenerator(knobs, output_dir=str(output_dir), seed=42)
    assert len(gen2.runhistory) >= 1
    assert expected_key in gen2._history_keys


def test_generate_after_direct_tell_without_ask(tmp_path, knobs, valid_config):
    """Validate SMAC can still generate proposals after direct external feedback without a prior ask."""
    gen = SMACProposalGenerator(
        knobs,
        output_dir=str(tmp_path / "smac_run"),
        runcount_limit=10,
        seed=42,
    )
    gen.tell(valid_config, 100.0)

    proposals = gen.generate({}, [], knobs, n=1)

    assert proposals
    assert set(proposals[0].keys()) == set(knobs.keys())


def test_ask_then_tell_feedback_loop(tmp_path, knobs):
    """Validate the primary production path where ask() is followed by tell() using a pending trial."""
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "smac_run"), seed=42)

    proposals = gen.generate({}, [], knobs, n=1)
    config = proposals[0]
    gen.tell(config, tps=150.0)

    assert len(gen.runhistory) == 1
    assert len(gen._pending_trials) == 0
    assert len(gen._trial_records) == 1


def test_replay_skips_malformed_record(tmp_path, knobs, valid_config):
    """Validate malformed trial records are skipped while valid replay history is preserved."""
    output_dir = tmp_path / "smac_run"
    gen = SMACProposalGenerator(knobs, output_dir=str(output_dir), seed=42)
    gen.tell(valid_config, 100.0)
    gen.save_state(gen.state_path)

    with open(gen.state_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["trial_records"].append({"config": None, "tps": None})
    with open(gen.state_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    gen2 = SMACProposalGenerator(knobs, output_dir=str(output_dir), seed=42)

    assert len(gen2.runhistory) >= 1
    assert len(gen2._trial_records) >= 1


def test_true_resume_same_output_dir(tmp_path, knobs, valid_config):
    """Verify that a new generator with the same output_dir auto-resumes saved state."""
    output_dir = str(tmp_path / "smac_run")
    gen1 = SMACProposalGenerator(knobs, output_dir=output_dir, seed=42)
    gen1.tell(valid_config, tps=100.0)
    expected_key = gen1._history_key(valid_config)
    gen1.save_state(gen1.state_path)
    history_len = len(gen1.runhistory)
    del gen1

    gen2 = SMACProposalGenerator(knobs, output_dir=output_dir, seed=42)
    assert len(gen2.runhistory) >= history_len
    assert expected_key in gen2._history_keys


def test_ask_then_tell_feedback_loop(tmp_path, knobs):
    """Verify the ask→tell production path used by Phase1Runner."""
    gen = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s"), seed=42)
    proposals = gen.generate({}, [], knobs, n=1)
    config = proposals[0]
    gen.tell(config, tps=150.0)
    assert len(gen.runhistory) == 1
    assert len(gen._pending_trials) == 0
    assert len(gen._trial_records) == 1


def test_generate_after_direct_tell_without_ask(tmp_path, knobs, valid_config):
    """Verify generate() still works after tell() is called without a prior ask()."""
    gen = SMACProposalGenerator(
        knobs, output_dir=str(tmp_path / "s"), runcount_limit=10, seed=42
    )
    gen.tell(valid_config, tps=100.0)
    proposals = gen.generate({}, [], knobs, n=1)
    assert len(proposals) == 1
    for param, value in proposals[0].items():
        assert knobs[param]["min"] <= value <= knobs[param]["max"]


def test_replay_skips_malformed_record(tmp_path, knobs, valid_config):
    """Verify malformed trial records are skipped without aborting replay."""
    output_dir = str(tmp_path / "smac_run")
    gen = SMACProposalGenerator(knobs, output_dir=output_dir, seed=42)
    gen.tell(valid_config, tps=100.0)
    gen.save_state(gen.state_path)

    with open(gen.state_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["trial_records"].append({"config": None, "tps": None})
    with open(gen.state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    gen2 = SMACProposalGenerator(knobs, output_dir=str(tmp_path / "s2"), seed=42)
    loaded = gen2.load_state(gen.state_path)
    assert loaded is True
    assert len(gen2.runhistory) >= 1
    assert len(gen2._trial_records) >= 1
