"""
Phase 1 entry point for PostgreSQL auto-tuning data generation.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

from Database import Database
from config import parse_config
from knob_config import parse_knob_config
from orchestration.phase1_runner import Phase1RunSummary, Phase1Runner
from orchestration.baseline_store import BaselineStore
from parameter_subsystem import ParameterExecutionSubsystem
from proposal_generators import get_generator
from sampling_runtime import SamplingRunRecorder
from stress_testing_tool import stress_testing_tool


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 PostgreSQL auto-tuning pipeline")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="PostgreSQL host")
    parser.add_argument("--database", type=str, default="tpch", help="Database / workload prefix")
    parser.add_argument("--datapath", type=str, required=True, help="Workload directory containing .wg files")
    parser.add_argument("--resume", action="store_true", help="Resume a previous Phase 1 run")
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="run_metadata/phase1_run.jsonl",
        help="Path to SamplingRunRecorder metadata output",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        choices=["random", "lhs", "heuristic", "smac"],
        help="Proposal generation strategy",
    )
    parser.add_argument(
        "--n-proposals",
        type=int,
        default=30,
        help="Number of proposals to evaluate per workload",
    )
    parser.add_argument(
        "--max-workloads",
        type=int,
        default=None,
        help="Maximum number of workloads to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and validate proposals without touching the database",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.ini",
        help="Path to config.ini",
    )
    return parser.parse_args()


def setup_logging() -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("Phase1Main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(os.path.join("logs", "phase1_main.log"), encoding="utf-8")
        formatter = logging.Formatter("[%(asctime)s - %(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger


def load_config(cli_args: argparse.Namespace) -> dict[str, Any]:
    config = parse_config.parse_args(cli_args.config)
    config.setdefault("database_config", {})
    config.setdefault("benchmark_config", {})
    config.setdefault("tuning_config", {})
    config.setdefault("parameter_execution", {})
    config.setdefault("ssh_config", {})

    config["ssh_config"]["host"] = cli_args.host
    config["database_config"]["host"] = cli_args.host
    config["database_config"]["database"] = cli_args.database
    config["database_config"]["datapath"] = cli_args.datapath
    config["database_config"]["workload_datapath"] = cli_args.datapath
    config["benchmark_config"]["workload_path"] = ""
    config["benchmark_config"]["tool"] = "direct"

    sample_prefix = config["tuning_config"].get("offline_sample", "offline_sample/offline_sample")
    # Store offline samples under a per-database directory so tpch/job runs do not overwrite each other.
    sample_prefix = os.path.join("offline_sample", cli_args.database, os.path.basename(str(sample_prefix)))
    if cli_args.host not in str(sample_prefix):
        sample_prefix = f"{sample_prefix}_{cli_args.host}"
    config["tuning_config"]["offline_sample"] = sample_prefix
    return config


def reset_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8"):
        pass


def build_generator(
    strategy: str,
    knobs_detail: dict[str, Any],
    logger: logging.Logger,
    n_proposals: int,
    database_name: str,
) -> Any:
    generator_kwargs: dict[str, Any] = {
        "knobs_detail": knobs_detail,
        "logger": logger,
        "seed": 42,
    }
    if strategy == "smac":
        generator_kwargs["output_dir"] = os.path.join("smac_state", database_name)
        generator_kwargs["runcount_limit"] = n_proposals
    return get_generator(strategy, **generator_kwargs)


def build_baseline_store(
    config: dict[str, Any],
    dry_run: bool,
) -> BaselineStore | None:
    """Build a BaselineStore co-located with the offline sample output."""
    if dry_run:
        return None
    sample_prefix = config["tuning_config"]["offline_sample"]
    store_dir = os.path.dirname(sample_prefix) or "offline_sample"
    store_path = os.path.join(store_dir, "baseline_records.jsonl")
    return BaselineStore(store_path)


def build_stress_testing_tool(
    config: dict[str, Any],
    logger: logging.Logger,
    dry_run: bool,
) -> stress_testing_tool:
    sample_prefix = config["tuning_config"]["offline_sample"]
    if dry_run:
        return stress_testing_tool(config, None, logger, sample_prefix, parameter_subsystem=None)

    database = Database(config["database_config"])
    parameter_subsystem = ParameterExecutionSubsystem.from_config(config, database, logger)
    return stress_testing_tool(
        config,
        database,
        logger,
        sample_prefix,
        parameter_subsystem=parameter_subsystem,
    )


def print_summary(summary: Phase1RunSummary) -> None:
    print("=" * 70)
    print("Phase 1 Summary")
    print("=" * 70)
    print(f"Workloads processed : {summary.workloads_processed}")
    print(f"Total samples       : {summary.total_samples}")
    print(f"Successful samples  : {summary.successful_samples}")
    print(f"Failed samples      : {summary.failed_samples}")
    print(f"Skipped samples     : {summary.skipped_samples}")
    print("Output files:")
    for output_file in summary.output_files:
        print(f"  - {output_file}")
    print("=" * 70)


def main() -> int:
    cli_args = parse_cli_args()
    logger = setup_logging()
    logger.info("Starting Phase 1 pipeline")

    if not os.path.isdir(cli_args.datapath):
        logger.error("Workload directory does not exist: %s", cli_args.datapath)
        return 1

    config = load_config(cli_args)
    knobs_detail = parse_knob_config.get_knobs(config["tuning_config"]["knob_config"])

    # Keep metadata in a per-database directory so each workload family has an isolated resume log.
    metadata_path = os.path.join("run_metadata", cli_args.database, os.path.basename(cli_args.metadata_path))
    sample_output_path = f"{config['tuning_config']['offline_sample']}.jsonl"
    if not cli_args.resume:
        if not cli_args.dry_run:
            reset_file(sample_output_path)
        reset_file(metadata_path)

    generator = build_generator(
        strategy=cli_args.strategy,
        knobs_detail=knobs_detail,
        logger=logger,
        n_proposals=cli_args.n_proposals,
        database_name=cli_args.database,
    )
    recorder = SamplingRunRecorder(metadata_path, resume=cli_args.resume)
    stt = build_stress_testing_tool(config, logger, dry_run=cli_args.dry_run)
    baseline_store = build_baseline_store(config, dry_run=cli_args.dry_run)

    runner = Phase1Runner(
        config=config,
        generator=generator,
        recorder=recorder,
        stt=stt,
        knobs_detail=knobs_detail,
        workload_dir=cli_args.datapath,
        workload_prefix=cli_args.database,
        n_proposals_per_workload=cli_args.n_proposals,
        max_workloads=cli_args.max_workloads,
        dry_run=cli_args.dry_run,
        baseline_store=baseline_store,
    )

    summary = runner.run()
    print_summary(summary)
    logger.info("Phase 1 pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
