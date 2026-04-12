"""Integration tests for Phase 1 pipeline (dry_run, no database)."""

from __future__ import annotations

import json
import logging
import os

import pytest

from sampling_runtime import SamplingRunRecorder
from proposal_generators.random_generator import RandomProposalGenerator
from orchestration.phase1_runner import Phase1Runner, Phase1RunSummary
from stress_testing_tool import stress_testing_tool
from training_data_builder import TrainingDataBuilder, BuilderConfig


def make_config(tmp_path):
    sample_prefix = str(tmp_path / "offline_sample" / "samples")
    return {
        "benchmark_config": {
            "workload_path": "",
            "tool": "direct",
            "timeout": "30",
            "fresh_session_per_test": "false",
            "fetch_result_rows": "false",
        },
        "tuning_config": {
            "offline_sample": sample_prefix,
            "knob_config": "knob_config/knob_config.json",
        },
        "database_config": {},
        "parameter_execution": {},
        "ssh_config": {"host": "localhost"},
    }


@pytest.fixture
def tpch_workload_dir(tmp_path):
    wdir = tmp_path / "workloads"
    wdir.mkdir()
    (wdir / "tpch_0.wg").write_text(
        "SELECT count(*) FROM lineitem WHERE l_quantity > 10;\n"
        "SELECT sum(l_extendedprice) FROM lineitem WHERE l_shipmode='MAIL';",
        encoding="utf-8",
    )
    (wdir / "tpch_1.wg").write_text(
        "SELECT l_orderkey, count(*) FROM lineitem GROUP BY l_orderkey ORDER BY l_orderkey;",
        encoding="utf-8",
    )
    return str(wdir)


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


def make_runner(
    tmp_path,
    tpch_workload_dir,
    knobs,
    n_proposals=2,
    max_workloads=None,
    *,
    workload_prefix="tpch",
    metadata_path=None,
    resume=False,
):
    config = make_config(tmp_path)
    os.makedirs(os.path.dirname(config["tuning_config"]["offline_sample"]), exist_ok=True)
    logger = logging.getLogger("test")
    stt = stress_testing_tool(
        config,
        database=None,
        logger=logger,
        sample_path=config["tuning_config"]["offline_sample"],
        parameter_subsystem=None,
    )
    recorder = SamplingRunRecorder(
        str(metadata_path or (tmp_path / "metadata.jsonl")),
        resume=resume,
    )
    generator = RandomProposalGenerator(seed=42)
    return Phase1Runner(
        config=config,
        generator=generator,
        recorder=recorder,
        stt=stt,
        knobs_detail=knobs,
        workload_dir=tpch_workload_dir,
        workload_prefix=workload_prefix,
        n_proposals_per_workload=n_proposals,
        max_workloads=max_workloads,
        dry_run=True,
    )


def test_discover_workloads_finds_tpch_files(tmp_path, tpch_workload_dir, knobs):
    runner = make_runner(tmp_path, tpch_workload_dir, knobs)
    workloads = runner.discover_workloads()
    assert len(workloads) == 2
    assert all(workload.endswith(".wg") for workload in workloads)
    assert all("tpch_" in os.path.basename(workload) for workload in workloads)


def test_discover_workloads_prefix_filter(tmp_path, tpch_workload_dir, knobs):
    extra_path = os.path.join(tpch_workload_dir, "job_0.wg")
    with open(extra_path, "w", encoding="utf-8") as handle:
        handle.write("SELECT 1;")

    runner = make_runner(
        tmp_path,
        tpch_workload_dir,
        knobs,
        workload_prefix="tpch",
    )
    workloads = runner.discover_workloads()
    assert len(workloads) == 2
    assert all(os.path.basename(workload) != "job_0.wg" for workload in workloads)


def test_dryrun_summary_counts(tmp_path, tpch_workload_dir, knobs):
    runner = make_runner(
        tmp_path,
        tpch_workload_dir,
        knobs,
        n_proposals=3,
        max_workloads=2,
    )
    summary = runner.run()
    assert isinstance(summary, Phase1RunSummary)
    assert summary.workloads_processed == 2
    assert summary.total_samples == 6
    assert summary.skipped_samples == 6
    assert summary.successful_samples == 0
    assert summary.failed_samples == 0


def test_dryrun_metadata_written(tmp_path, tpch_workload_dir, knobs):
    runner = make_runner(
        tmp_path,
        tpch_workload_dir,
        knobs,
        n_proposals=3,
        max_workloads=2,
    )
    runner.run()

    metadata_path = tmp_path / "metadata.jsonl"
    assert metadata_path.exists()
    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 6
    for record in records:
        assert "sample_key" in record
        assert "status" in record
        assert "timestamp" in record


def test_dryrun_all_metadata_have_required_fields(tmp_path, tpch_workload_dir, knobs):
    runner = make_runner(
        tmp_path,
        tpch_workload_dir,
        knobs,
        n_proposals=3,
        max_workloads=2,
    )
    runner.run()

    records = [
        json.loads(line)
        for line in (tmp_path / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        "sample_key",
        "workload_id",
        "status",
        "duration_seconds",
        "timestamp",
        "metadata",
    }
    for record in records:
        assert required <= set(record.keys())
        assert record["status"] == "skipped"


def test_resume_skips_success_records(tmp_path, tpch_workload_dir, knobs):
    metadata_path = tmp_path / "metadata.jsonl"
    first_runner = make_runner(
        tmp_path,
        tpch_workload_dir,
        knobs,
        n_proposals=2,
        max_workloads=1,
        metadata_path=metadata_path,
        resume=False,
    )
    first_runner.run()

    first_config = RandomProposalGenerator(seed=42).generate({}, [], knobs, n=1)[0]
    target_key = SamplingRunRecorder(str(metadata_path)).build_sample_key(
        "tpch_0",
        "random",
        first_config,
    )
    with open(metadata_path, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_key": target_key,
                    "workload_id": "tpch_0",
                    "status": "success",
                    "tps": 123.0,
                    "duration_seconds": 0.01,
                    "metadata": {"source": "manual"},
                    "timestamp": 1.0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    second_runner = make_runner(
        tmp_path,
        tpch_workload_dir,
        knobs,
        n_proposals=2,
        max_workloads=1,
        metadata_path=metadata_path,
        resume=True,
    )
    second_runner.run()

    records = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        record["sample_key"] == target_key and record["status"] == "skipped"
        for record in records
    )


def test_training_data_builder_with_mock_offline_samples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    workload_file = tmp_path / "tpch_0.wg"
    workload_file.write_text(
        "SELECT count(*) FROM lineitem WHERE l_quantity > 10;\n"
        "SELECT sum(l_extendedprice) FROM lineitem WHERE l_shipmode='MAIL';",
        encoding="utf-8",
    )

    sample_path = tmp_path / "offline_sample.jsonl"
    output_path = tmp_path / "training_sft_data.jsonl"

    records = []
    for tps in [50.0, 80.0, 120.0, 160.0, 200.0]:
        records.append(
            {
                "workload": "tpch_0.wg",
                "workload_file": "tpch_0.wg",
                "config": {
                    "shared_buffers": 131072,
                    "work_mem": 65536,
                    "checkpoint_completion_target": 0.5,
                },
                "tps": tps,
                "inner_metrics": {
                    "cache_hit_ratio": 0.97,
                    "active_connections": 5,
                    "cpu_usage": 23.0,
                },
                "query_plans": "",
                "y": -tps,
            }
        )

    with open(sample_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    builder = TrainingDataBuilder(
        str(sample_path),
        str(output_path),
        BuilderConfig(min_samples=1, max_samples=10, min_tps=0.001),
    )
    result = builder.build_and_save()

    assert result is True
    assert os.path.exists(output_path)
    training_samples = [
        json.loads(line)
        for line in open(output_path, "r", encoding="utf-8")
        if line.strip()
    ]
    assert len(training_samples) >= 1
    for sample in training_samples:
        assert {"instruction", "input", "output"} <= set(sample.keys())
        json.loads(sample["output"])
        assert "Workload Statistics" in sample["input"]
