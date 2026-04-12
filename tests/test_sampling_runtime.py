"""Unit tests for SamplingRunRecorder."""

from __future__ import annotations

import json

from sampling_runtime import SamplingRunRecorder


def test_build_sample_key_deterministic(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"))
    key1 = recorder.build_sample_key("tpch_0", "random", {"shared_buffers": 1, "work_mem": 2})
    key2 = recorder.build_sample_key("tpch_0", "random", {"shared_buffers": 1, "work_mem": 2})
    assert key1 == key2


def test_build_sample_key_format(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"))
    key = recorder.build_sample_key("tpch_0", "random", {"shared_buffers": 1})
    assert key.startswith("tpch_0:random:")
    assert len(key.split(":")) == 3


def test_build_sample_key_differs_for_different_config(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"))
    key1 = recorder.build_sample_key("w", "r", {"a": 1})
    key2 = recorder.build_sample_key("w", "r", {"a": 2})
    assert key1 != key2


def test_should_skip_false_when_not_resume(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"), resume=False)
    recorder.record({"sample_key": "k1", "status": "success"})
    assert recorder.should_skip("k1") is False


def test_should_skip_false_before_any_record(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"), resume=True)
    assert recorder.should_skip("nonexistent") is False


def test_record_writes_json_line(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    recorder = SamplingRunRecorder(str(metadata_path))
    recorder.record({"sample_key": "k1", "status": "success", "tps": 99.0})

    lines = metadata_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tps"] == 99.0


def test_record_adds_timestamp_if_missing(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    recorder = SamplingRunRecorder(str(metadata_path))
    recorder.record({"sample_key": "k1", "status": "success"})

    record = json.loads(metadata_path.read_text(encoding="utf-8").strip())
    assert "timestamp" in record
    assert isinstance(record["timestamp"], float)


def test_record_success_enables_skip(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"), resume=True)
    recorder.record({"sample_key": "k1", "status": "success"})
    assert recorder.should_skip("k1") is True


def test_record_failed_does_not_enable_skip(tmp_path):
    recorder = SamplingRunRecorder(str(tmp_path / "metadata.jsonl"), resume=True)
    recorder.record({"sample_key": "k1", "status": "failed"})
    assert recorder.should_skip("k1") is False


def test_resume_loads_previous_success(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    key1 = "w1:random:abc"
    key2 = "w2:random:def"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps({"sample_key": key1, "status": "success"}),
                json.dumps({"sample_key": key2, "status": "success"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    recorder = SamplingRunRecorder(str(metadata_path), resume=True)
    assert recorder.should_skip(key1) is True
    assert recorder.should_skip(key2) is True
    assert recorder.should_skip("unknown") is False


def test_resume_false_ignores_existing(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    key = "w1:random:abc"
    metadata_path.write_text(
        json.dumps({"sample_key": key, "status": "success"}) + "\n",
        encoding="utf-8",
    )

    recorder = SamplingRunRecorder(str(metadata_path), resume=False)
    assert recorder.should_skip(key) is False


def test_corrupted_line_skipped_gracefully(tmp_path):
    metadata_path = tmp_path / "metadata.jsonl"
    first_key = "w1:random:abc"
    last_key = "w2:random:def"
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps({"sample_key": first_key, "status": "success"}),
                "NOT_JSON_{{{{",
                json.dumps({"sample_key": last_key, "status": "success"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    recorder = SamplingRunRecorder(str(metadata_path), resume=True)
    assert recorder.should_skip(first_key) is True
    assert recorder.should_skip(last_key) is True
