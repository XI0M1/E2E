"""
Phase 1 orchestration runner.

This module turns the Phase 1 data-generation process into a resumable,
failure-isolated pipeline without changing the underlying sampling/execution
components.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import traceback
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List

from feature_extractor import extract_workload_features
from proposal_generators.base import ProposalGenerator
from sampling_runtime import SamplingRunRecorder
from training_data_builder import build_training_data


@dataclass
class SampleResult:
    sample_key: str
    workload_id: str
    config: dict
    tps: float | None
    status: str
    error: str | None
    duration_seconds: float
    metadata: dict


@dataclass
class WorkloadRunResult:
    workload_id: str
    total: int
    success: int
    failed: int
    skipped: int
    best_tps: float | None
    samples: List[SampleResult] = field(default_factory=list)


@dataclass
class Phase1RunSummary:
    workloads_processed: int
    total_samples: int
    successful_samples: int
    failed_samples: int
    skipped_samples: int
    best_configs: dict[str, dict]
    output_files: list[str]


class Phase1Runner:
    def __init__(
        self,
        config: dict,
        generator: ProposalGenerator,
        recorder: SamplingRunRecorder,
        stt,
        knobs_detail: dict,
        workload_dir: str,
        workload_prefix: str,
        n_proposals_per_workload: int = 30,
        max_workloads: int | None = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.generator = generator
        self.recorder = recorder
        self.stt = stt
        self.knobs_detail = knobs_detail
        self.workload_dir = workload_dir
        self.workload_prefix = workload_prefix
        self.n_proposals_per_workload = n_proposals_per_workload
        self.max_workloads = max_workloads
        self.dry_run = dry_run
        self.logger = getattr(stt, "logger", logging.getLogger("Phase1Runner"))

    def discover_workloads(self) -> list[str]:
        workload_files = sorted(
            os.path.join(self.workload_dir, filename)
            for filename in os.listdir(self.workload_dir)
            if filename.endswith(".wg") and filename.startswith(self.workload_prefix)
        )
        if self.max_workloads is not None:
            workload_files = workload_files[: self.max_workloads]
        return workload_files

    def run(self) -> Phase1RunSummary:
        summary = Phase1RunSummary(
            workloads_processed=0,
            total_samples=0,
            successful_samples=0,
            failed_samples=0,
            skipped_samples=0,
            best_configs={},
            output_files=[],
        )

        sample_output_path = self._sample_output_path()
        summary.output_files.append(self.recorder.metadata_path)
        if sample_output_path and os.path.exists(sample_output_path):
            summary.output_files.append(sample_output_path)

        try:
            workloads = self.discover_workloads()
            self.logger.info("Discovered %s workloads for Phase 1", len(workloads))

            for workload_path in workloads:
                try:
                    workload_result = self._run_single_workload(workload_path)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    workload_id = self._workload_id_from_path(workload_path)
                    error_text = self._format_exception(exc)
                    failure_payload = {
                        "sample_key": self.recorder.build_sample_key(
                            workload_id=workload_id,
                            sample_kind="workload_error",
                            config={"workload_path": workload_path},
                        ),
                        "workload_id": workload_id,
                        "status": "failed",
                        "error": error_text,
                        "tps": None,
                        "duration_seconds": 0.0,
                    }
                    self.recorder.record(failure_payload)
                    workload_result = WorkloadRunResult(
                        workload_id=workload_id,
                        total=1,
                        success=0,
                        failed=1,
                        skipped=0,
                        best_tps=None,
                        samples=[],
                    )
                    self.logger.error("Workload failed [%s]: %s", workload_id, error_text)

                summary.workloads_processed += 1
                summary.total_samples += workload_result.total
                summary.successful_samples += workload_result.success
                summary.failed_samples += workload_result.failed
                summary.skipped_samples += workload_result.skipped
                if workload_result.best_tps is not None:
                    best_sample = max(
                        (sample for sample in workload_result.samples if sample.tps is not None),
                        key=lambda sample: sample.tps or float("-inf"),
                        default=None,
                    )
                    if best_sample is not None:
                        summary.best_configs[workload_result.workload_id] = best_sample.config

            if not self.dry_run:
                output_files = self._finalize_outputs(sample_output_path)
                for output_file in output_files:
                    if output_file not in summary.output_files:
                        summary.output_files.append(output_file)

        except KeyboardInterrupt:
            self.logger.warning("Phase 1 interrupted by user, attempting graceful shutdown")
            self.recorder.record(
                {
                    "sample_key": "__phase1_interrupt__",
                    "workload_id": "__phase1__",
                    "status": "failed",
                    "error": "KeyboardInterrupt",
                    "tps": None,
                    "duration_seconds": 0.0,
                }
            )
            self._persist_generator_state()

        return summary

    def _count_completed_proposals(self, workload_id: str) -> int:
        """
        Count how many successful proposals have already been recorded
        for this workload_id in the run metadata.
        """
        count = 0
        for key in self.recorder.completed_keys:
            # sample_key format: "{workload_id}:{sample_kind}:{hash}"
            if key.startswith(f"{workload_id}:"):
                count += 1
        return count

    def _run_single_workload(self, workload_path: str) -> WorkloadRunResult:
        workload_id = self._workload_id_from_path(workload_path)

        remaining_proposals = self.n_proposals_per_workload
        if self.recorder.resume:
            already_done = self._count_completed_proposals(workload_id)
            if already_done >= self.n_proposals_per_workload:
                self.logger.info(
                    "Skipping workload %s: already has %d/%d completed proposals",
                    workload_id,
                    already_done,
                    self.n_proposals_per_workload,
                )
                return WorkloadRunResult(
                    workload_id=workload_id,
                    total=already_done,
                    success=already_done,
                    failed=0,
                    skipped=already_done,
                    best_tps=None,
                    samples=[],
                )
            remaining_proposals = self.n_proposals_per_workload - already_done

        self.logger.info("Running workload: %s", workload_id)

        baseline_result: dict = {}
        if self.stt is not None and not self.dry_run:
            baseline_result = self._run_baseline(workload_path)
        else:
            baseline_result = {"baseline_tps": 0.0, "baseline_runs": []}

        baseline_tps: float = baseline_result.get("baseline_tps", 0.0)

        if self.stt is not None:
            self.stt.benchmark_config["workload_path"] = workload_path
            self.stt.workload_file = workload_path

        samples: List[SampleResult] = []
        history: List[Dict[str, Any]] = []
        success = 0
        failed = 0
        skipped = 0
        best_tps: float | None = None
        workload_features = self._load_workload_features(workload_path)

        for proposal_index in range(remaining_proposals):
            try:
                proposals = self.generator.generate(
                    workload_features=workload_features,
                    history=history,
                    constraints=self.knobs_detail,
                    n=1,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                sample_result = self._build_failed_result(
                    workload_id=workload_id,
                    config={},
                    sample_kind=f"{self.generator.name}_generate",
                    error=self._format_exception(exc),
                    duration_seconds=0.0,
                    metadata={"generator": self.generator.name, "proposal_index": proposal_index},
                )
                self._record_sample_result(sample_result)
                samples.append(sample_result)
                failed += 1
                continue

            if not proposals:
                sample_result = self._build_skipped_result(
                    workload_id=workload_id,
                    config={},
                    sample_kind=f"{self.generator.name}_generate",
                    duration_seconds=0.0,
                    metadata={"generator": self.generator.name, "proposal_index": proposal_index},
                    error="generator returned no proposals",
                )
                self._record_sample_result(sample_result)
                samples.append(sample_result)
                skipped += 1
                continue

            sample_result = self._run_single_config(
                workload_id=workload_id,
                config=proposals[0],
                sample_kind=self.generator.name,
                baseline_tps=baseline_tps,
            )
            sample_result.metadata.setdefault("proposal_index", proposal_index)
            samples.append(sample_result)

            if sample_result.status == "success":
                success += 1
                if sample_result.tps is not None:
                    history.append({"config": sample_result.config, "tps": sample_result.tps})
                    if best_tps is None or sample_result.tps > best_tps:
                        best_tps = sample_result.tps
                    feedback_fn = getattr(self.generator, "tell", None)
                    if callable(feedback_fn):
                        feedback_fn(sample_result.config, sample_result.tps)
            elif sample_result.status == "failed":
                failed += 1
            else:
                skipped += 1

        self._persist_generator_state()
        return WorkloadRunResult(
            workload_id=workload_id,
            total=len(samples),
            success=success,
            failed=failed,
            skipped=skipped,
            best_tps=best_tps,
            samples=samples,
        )

    def _run_baseline(self, workload_path: str) -> dict:
        workload_id = self._workload_id_from_path(workload_path)
        default_config = {
            name: detail["default"]
            for name, detail in self.knobs_detail.items()
        }
        try:
            r1 = self.stt.test_config(default_config)
            r2 = self.stt.test_config(default_config)
            r3 = self.stt.test_config(default_config)
            median_tps = statistics.median([r1, r2, r3])
            self.logger.info(
                "Baseline for %s: tps=%.3f (runs=%.3f,%.3f,%.3f)",
                workload_id,
                median_tps,
                r1,
                r2,
                r3,
            )
            return {
                "baseline_tps": float(median_tps),
                "baseline_runs": [float(r1), float(r2), float(r3)],
            }
        except Exception:
            self.logger.warning(
                "Baseline measurement failed for %s", workload_id, exc_info=True
            )
            return {"baseline_tps": 0.0, "baseline_runs": []}

    def _run_single_config(
        self,
        workload_id: str,
        config: dict,
        sample_kind: str,
        baseline_tps: float = 0.0,
    ) -> SampleResult:
        started = perf_counter()
        metadata = {
            "generator": self.generator.name,
            "sample_kind": sample_kind,
            "dry_run": self.dry_run,
        }

        validation_config = dict(config)
        if not self.dry_run and self.stt is not None and getattr(self.stt, "parameter_subsystem", None) is not None:
            validation_result = self.stt.parameter_subsystem.validate_config(config)
            metadata["validation"] = {"valid": validation_result.get("valid", False)}
            validation_config = validation_result.get("normalized_config", config)
            if not validation_result.get("valid", False):
                return self._finalize_result(
                    self._build_failed_result(
                        workload_id=workload_id,
                        config=validation_config,
                        sample_kind=sample_kind,
                        error="parameter validation rejected config",
                        duration_seconds=perf_counter() - started,
                        metadata=metadata,
                    )
                )

        sample_key = self.recorder.build_sample_key(
            workload_id=workload_id,
            sample_kind=sample_kind,
            config=validation_config,
        )
        if self.recorder.should_skip(sample_key):
            return self._finalize_result(
                self._build_skipped_result(
                    workload_id=workload_id,
                    config=validation_config,
                    sample_kind=sample_kind,
                    duration_seconds=perf_counter() - started,
                    metadata=metadata,
                    sample_key=sample_key,
                )
            )

        if self.dry_run:
            return self._finalize_result(
                self._build_skipped_result(
                    workload_id=workload_id,
                    config=validation_config,
                    sample_kind=sample_kind,
                    duration_seconds=perf_counter() - started,
                    metadata=metadata,
                    sample_key=sample_key,
                    error="dry-run",
                )
            )

        try:
            tps = self.stt.test_config(validation_config)

            # Compute relative improvement over default config baseline.
            # relative_score > 0 means better than default; < 0 means worse.
            relative_score: float = 0.0
            if baseline_tps > 1e-6:
                relative_score = (float(tps) - baseline_tps) / baseline_tps

            metadata["baseline_tps"] = baseline_tps
            metadata["relative_score"] = relative_score
            return self._finalize_result(
                SampleResult(
                    sample_key=sample_key,
                    workload_id=workload_id,
                    config=validation_config,
                    tps=float(tps),
                    status="success",
                    error=None,
                    duration_seconds=perf_counter() - started,
                    metadata=metadata,
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return self._finalize_result(
                self._build_failed_result(
                    workload_id=workload_id,
                    config=validation_config,
                    sample_kind=sample_kind,
                    error=self._format_exception(exc),
                    duration_seconds=perf_counter() - started,
                    metadata=metadata,
                    sample_key=sample_key,
                )
            )

    def _finalize_outputs(self, sample_output_path: str | None) -> list[str]:
        output_files: list[str] = []
        if not sample_output_path or not os.path.exists(sample_output_path):
            return output_files

        if extract_workload_features(sample_output_path, self.workload_prefix):
            feature_path = os.path.join("SuperWG", "feature", f"{self.workload_prefix}.json")
            if os.path.exists(feature_path):
                output_files.append(feature_path)

        training_output_path = os.path.join(
            "training_data",
            f"training_sft_data_{self.workload_prefix}.jsonl",
        )
        if build_training_data(sample_output_path, training_output_path):
            if os.path.exists(training_output_path):
                output_files.append(training_output_path)

        output_files.append(sample_output_path)
        return output_files

    def _sample_output_path(self) -> str | None:
        if self.stt is None:
            return None
        sample_path = getattr(self.stt, "sample_path", None)
        if not sample_path:
            return None
        return sample_path if str(sample_path).endswith(".jsonl") else f"{sample_path}.jsonl"

    def _record_sample_result(self, sample_result: SampleResult) -> None:
        self.recorder.record(
            {
                "sample_key": sample_result.sample_key,
                "workload_id": sample_result.workload_id,
                "status": sample_result.status,
                "error": sample_result.error,
                "tps": sample_result.tps,
                "duration_seconds": sample_result.duration_seconds,
                "metadata": sample_result.metadata,
            }
        )

    def _finalize_result(self, sample_result: SampleResult) -> SampleResult:
        self._record_sample_result(sample_result)
        return sample_result

    def _build_failed_result(
        self,
        workload_id: str,
        config: dict,
        sample_kind: str,
        error: str,
        duration_seconds: float,
        metadata: dict,
        sample_key: str | None = None,
    ) -> SampleResult:
        return SampleResult(
            sample_key=sample_key
            or self.recorder.build_sample_key(workload_id, sample_kind, config or {"status": "failed"}),
            workload_id=workload_id,
            config=config,
            tps=None,
            status="failed",
            error=error,
            duration_seconds=duration_seconds,
            metadata=metadata,
        )

    def _build_skipped_result(
        self,
        workload_id: str,
        config: dict,
        sample_kind: str,
        duration_seconds: float,
        metadata: dict,
        sample_key: str | None = None,
        error: str | None = None,
    ) -> SampleResult:
        return SampleResult(
            sample_key=sample_key
            or self.recorder.build_sample_key(workload_id, sample_kind, config or {"status": "skipped"}),
            workload_id=workload_id,
            config=config,
            tps=None,
            status="skipped",
            error=error,
            duration_seconds=duration_seconds,
            metadata=metadata,
        )

    def _persist_generator_state(self) -> None:
        save_state = getattr(self.generator, "save_state", None)
        state_path = getattr(self.generator, "state_path", None)
        if callable(save_state) and isinstance(state_path, str) and state_path:
            try:
                save_state(state_path)
            except Exception as exc:
                self.logger.warning("Failed to persist generator state: %s", exc)

    def _load_workload_features(self, workload_path: str) -> dict:
        workload_id = self._workload_id_from_path(workload_path)
        features = {
            "workload_path": workload_path,
            "workload_id": workload_id,
            "total_sql": 0,
            "read_ratio": 0.0,
            "write_ratio": 0.0,
            "join_count": 0,
            "aggregation_count": 0,
            "order_by_percent": 0.0,
            "group_by_percent": 0.0,
            "table_count": 0,
        }

        if not os.path.exists(workload_path):
            return features

        try:
            with open(workload_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return features

        statements = [
            statement.strip()
            for statement in content.split(";")
            if statement.strip() and not statement.strip().startswith(("--", "#"))
        ]
        if not statements:
            return features

        read_count = 0
        write_count = 0
        join_count = 0
        aggregation_count = 0
        order_by_count = 0
        group_by_count = 0
        table_names: set[str] = set()

        for statement in statements:
            upper_stmt = statement.upper()
            if upper_stmt.startswith("SELECT") or " SELECT " in f" {upper_stmt} ":
                read_count += 1
            if any(keyword in upper_stmt for keyword in ("INSERT ", "UPDATE ", "DELETE ")):
                write_count += 1

            join_count += upper_stmt.count(" JOIN ")
            aggregation_count += sum(upper_stmt.count(func) for func in ("COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
            order_by_count += 1 if " ORDER BY " in upper_stmt else 0
            group_by_count += 1 if " GROUP BY " in upper_stmt else 0

            table_names.update(re.findall(r"\bFROM\s+([A-Z0-9_]+)", upper_stmt))
            table_names.update(re.findall(r"\bJOIN\s+([A-Z0-9_]+)", upper_stmt))

        total_sql = len(statements)
        features.update(
            {
                "total_sql": total_sql,
                "read_ratio": (read_count / total_sql) * 100.0 if total_sql else 0.0,
                "write_ratio": (write_count / total_sql) * 100.0 if total_sql else 0.0,
                "join_count": join_count,
                "aggregation_count": aggregation_count,
                "order_by_percent": (order_by_count / total_sql) * 100.0 if total_sql else 0.0,
                "group_by_percent": (group_by_count / total_sql) * 100.0 if total_sql else 0.0,
                "table_count": len(table_names),
            }
        )
        return features

    def _workload_id_from_path(self, workload_path: str) -> str:
        return os.path.splitext(os.path.basename(workload_path))[0]

    def _format_exception(self, exc: Exception) -> str:
        trace_text = traceback.format_exc()
        if not trace_text or trace_text.strip() == "NoneType: None":
            trace_text = str(exc)
        return trace_text[-500:]
