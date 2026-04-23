from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("surrogate.evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class OptionalExecutionContext:
    database: Any | None = None
    stress_tool: Any | None = None
    config: dict[str, Any] | None = None
    db_name: str | None = None

    def close(self) -> None:
        if self.database is not None:
            try:
                self.database.close()
            except Exception:
                pass


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_predictions(prediction_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f"Prediction file does not exist: {prediction_path}")

    rows: list[dict[str, Any]] = []
    with open(prediction_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid prediction JSON in {prediction_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Prediction row must be a JSON object in {prediction_path}:{line_number}"
                )
            rows.append(payload)

    return rows


def normalize_workload_id(value: str | None) -> str:
    if not value:
        return ""
    basename = os.path.basename(str(value).strip())
    if not basename:
        return ""
    return Path(basename).stem


def infer_db_type_from_workload_file(workload_file: str | None) -> str:
    workload_id = normalize_workload_id(workload_file).lower()
    if workload_id.startswith("job_"):
        return "job"
    if workload_id.startswith("ssb_"):
        return "ssb"
    if workload_id.startswith("tpch_"):
        return "tpch"
    return ""


def load_baseline_records(baseline_path: str) -> dict[str, float]:
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline file does not exist: {baseline_path}")

    lookup: dict[str, float] = {}
    with open(baseline_path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.warning(
                    "Skipping malformed baseline row %s:%d: %s",
                    baseline_path,
                    line_number,
                    exc,
                )
                continue

            if not isinstance(payload, dict):
                LOGGER.warning(
                    "Skipping non-object baseline row %s:%d",
                    baseline_path,
                    line_number,
                )
                continue

            workload_id = normalize_workload_id(payload.get("workload_id"))
            baseline_tps = payload.get("baseline_tps")
            try:
                baseline_tps_value = float(baseline_tps)
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Skipping baseline row with invalid baseline_tps %s:%d",
                    baseline_path,
                    line_number,
                )
                continue

            if not workload_id:
                LOGGER.warning(
                    "Skipping baseline row with empty workload_id %s:%d",
                    baseline_path,
                    line_number,
                )
                continue

            if workload_id in lookup:
                LOGGER.warning(
                    "Duplicate baseline workload_id '%s' at %s:%d; keeping last value",
                    workload_id,
                    baseline_path,
                    line_number,
                )
            lookup[workload_id] = baseline_tps_value

    return lookup


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_skipped_execution(error: str) -> dict[str, Any]:
    return {
        "attempted": False,
        "applied": False,
        "success": False,
        "error": error,
        "resolved_workload_path": None,
        "tps": None,
        "avg_latency_ms": None,
        "workload_type": None,
        "relative_score": None,
    }


def build_workload_search_dirs() -> list[str]:
    env_dirs = os.environ.get("WORKLOAD_SEARCH_DIRS", "")
    directories = [directory for directory in env_dirs.split(os.pathsep) if directory]
    directories.extend(
        [
            "data",
            os.path.join("data", "olap"),
            os.path.join("data", "oltp"),
            "olap_workloads",
            "oltp_workloads",
            ".",
        ]
    )
    return directories


def resolve_existing_path(path_text: str | None, source_file: str | None = None) -> str | None:
    if not path_text:
        return None

    candidate = os.path.expanduser(str(path_text).strip())
    if not candidate:
        return None

    candidate_paths = [candidate]
    if source_file and not os.path.isabs(candidate):
        candidate_paths.append(os.path.join(os.path.dirname(source_file), candidate))
    if not os.path.isabs(candidate):
        candidate_paths.append(os.path.join(str(PROJECT_ROOT), candidate))

    for candidate_path in candidate_paths:
        normalized = os.path.abspath(candidate_path)
        if os.path.exists(normalized):
            return normalized
    return None


def resolve_workload_path(row: dict[str, Any]) -> tuple[str | None, str | None]:
    source_file = row.get("source_file")
    workload = row.get("workload")
    workload_file = row.get("workload_file")

    for candidate in [workload, workload_file]:
        resolved = resolve_existing_path(candidate, source_file=source_file)
        if resolved:
            return resolved, None

    for candidate in [workload, workload_file]:
        basename = os.path.basename(str(candidate or "").strip())
        if not basename:
            continue
        for directory in build_workload_search_dirs():
            resolved = os.path.abspath(os.path.join(str(PROJECT_ROOT), directory, basename))
            if os.path.exists(resolved):
                return resolved, None

    return None, "unable to resolve workload path"


def build_optional_execution_context(config_path: str | None) -> OptionalExecutionContext:
    if not config_path:
        LOGGER.info("Dynamic execution disabled (no --config provided)")
        return OptionalExecutionContext()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Execution config file does not exist: {config_path}")

    from config.parse_config import parse_args as parse_ini_config
    from Database import Database
    from stress_testing_tool import stress_testing_tool

    config = parse_ini_config(config_path)
    database = Database(config)
    sample_path = os.path.join(
        "surrogate",
        "evaluations",
        "execution_artifacts",
        "shared_runtime",
    )
    stress_tool = stress_testing_tool(
        config,
        database,
        LOGGER,
        sample_path=sample_path,
    )
    return OptionalExecutionContext(
        database=database,
        stress_tool=stress_tool,
        config=config,
        db_name=str(config.get("database_config", {}).get("database", "")).strip() or None,
    )


def build_execution_context_with_db_override(
    config_path: str | None,
    db_override: str | None = None,
) -> OptionalExecutionContext:
    if not config_path:
        LOGGER.info("Dynamic execution disabled (no --config provided)")
        return OptionalExecutionContext()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Execution config file does not exist: {config_path}")

    from config.parse_config import parse_args as parse_ini_config
    from Database import Database
    from stress_testing_tool import stress_testing_tool

    config = parse_ini_config(config_path)
    config.setdefault("database_config", {})

    if db_override:
        config["database_config"]["database"] = db_override

    effective_db_name = str(config.get("database_config", {}).get("database", "")).strip() or None
    database = Database(config)
    sample_path = os.path.join(
        "surrogate",
        "evaluations",
        "execution_artifacts",
        "shared_runtime",
    )
    stress_tool = stress_testing_tool(
        config,
        database,
        LOGGER,
        sample_path=sample_path,
    )
    LOGGER.info(
        "Dynamic execution enabled via %s (database=%s)",
        config_path,
        effective_db_name,
    )
    return OptionalExecutionContext(
        database=database,
        stress_tool=stress_tool,
        config=config,
        db_name=effective_db_name,
    )


def build_execution_sample_path(row: dict[str, Any]) -> str:
    output_dir = os.path.join("surrogate", "evaluations", "execution_artifacts")
    os.makedirs(output_dir, exist_ok=True)
    safe_id = str(row.get("sample_id") or "sample").replace("\\", "_").replace("/", "_").replace(":", "_")
    return os.path.join(output_dir, f"{safe_id}_{os.getpid()}")


def read_last_json_record(jsonl_path: str) -> dict[str, Any] | None:
    if not os.path.exists(jsonl_path):
        return None

    last_record: dict[str, Any] | None = None
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            last_record = json.loads(stripped)
    return last_record


def execute_prediction_row(
    row: dict[str, Any],
    execution_context: OptionalExecutionContext,
) -> dict[str, Any]:
    if execution_context.stress_tool is None:
        return build_skipped_execution("dynamic execution not enabled")

    standardized_config = row.get("standardized_config")
    if not isinstance(standardized_config, dict) or not standardized_config:
        return build_skipped_execution("standardized_config is empty")

    if row.get("ready_for_apply") is False:
        return build_skipped_execution("prediction is not ready_for_apply")

    workload_path, workload_error = resolve_workload_path(row)
    if not workload_path:
        return build_skipped_execution(workload_error or "unknown workload resolution error")

    stress_tool = execution_context.stress_tool
    sample_path = build_execution_sample_path(row)
    jsonl_path = sample_path + ".jsonl"
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    original_sample_path = stress_tool.sample_path
    original_workload_file = getattr(stress_tool, "workload_file", "")
    original_benchmark_config = dict(getattr(stress_tool, "benchmark_config", {}) or {})

    try:
        stress_tool.sample_path = sample_path
        stress_tool.benchmark_config = dict(original_benchmark_config)
        stress_tool.benchmark_config["workload_path"] = workload_path
        stress_tool.workload_file = workload_path

        fallback_tps = float(stress_tool.test_config(standardized_config))
        record = read_last_json_record(jsonl_path)

        if record is None:
            return {
                "attempted": True,
                "applied": False,
                "success": False,
                "error": "no execution record was produced",
                "resolved_workload_path": workload_path,
                "tps": None,
                "avg_latency_ms": None,
                "workload_type": row.get("workload_type"),
                "relative_score": None,
            }

        measured_tps = _normalize_optional_float(
            record.get("tps", record.get("performance", fallback_tps))
        )
        avg_latency_ms = _normalize_optional_float(record.get("avg_latency_ms"))
        workload_type = record.get("workload_type") or row.get("workload_type")
        relative_score = _normalize_optional_float(record.get("relative_score"))
        success = measured_tps is not None and measured_tps > 0.0

        return {
            "attempted": True,
            "applied": True,
            "success": success,
            "error": None if success else "workload executed but produced non-positive TPS",
            "resolved_workload_path": workload_path,
            "tps": measured_tps,
            "avg_latency_ms": avg_latency_ms,
            "workload_type": workload_type,
            "relative_score": relative_score,
        }
    except Exception as exc:
        LOGGER.exception("Dynamic execution failed for sample %s", row.get("sample_id"))
        return {
            "attempted": True,
            "applied": False,
            "success": False,
            "error": str(exc),
            "resolved_workload_path": workload_path,
            "tps": None,
            "avg_latency_ms": None,
            "workload_type": row.get("workload_type"),
            "relative_score": None,
        }
    finally:
        stress_tool.sample_path = original_sample_path
        stress_tool.workload_file = original_workload_file
        stress_tool.benchmark_config = original_benchmark_config


def validate_row_db_match(
    row: dict[str, Any],
    selected_db: str | None,
) -> None:
    if not selected_db:
        return

    workload_hint = row.get("workload_file") or row.get("workload")
    inferred_db = infer_db_type_from_workload_file(workload_hint)
    if inferred_db and inferred_db != selected_db:
        raise ValueError(
            f"Database/workload mismatch: selected_db={selected_db}, "
            f"but workload '{workload_hint}' implies {inferred_db}"
        )


def evaluate_prediction_row(
    row: dict[str, Any],
    baseline_lookup: dict[str, float],
    execution_context: OptionalExecutionContext | None = None,
    execute_predictions: bool = False,
) -> dict[str, Any]:
    evaluated = dict(row)
    workload_value = row.get("workload_file") or row.get("workload")
    baseline_lookup_key = normalize_workload_id(workload_value)
    baseline_tps = baseline_lookup.get(baseline_lookup_key)

    if execute_predictions and execution_context is not None:
        validate_row_db_match(row, execution_context.db_name)
        execution = execute_prediction_row(row, execution_context)
    else:
        execution = build_skipped_execution("dynamic execution not enabled")

    performance_comparison = {
        "performed": False,
        "better_than_baseline": None,
        "tps_delta_abs": None,
        "tps_delta_ratio": None,
    }

    execution_tps = _normalize_optional_float(execution.get("tps"))
    if execution_tps is not None and baseline_tps is not None and bool(execution.get("success")):
        tps_delta_abs = execution_tps - baseline_tps
        tps_delta_ratio = None
        if baseline_tps > 1e-6:
            tps_delta_ratio = tps_delta_abs / baseline_tps
        performance_comparison = {
            "performed": True,
            "better_than_baseline": execution_tps > baseline_tps,
            "tps_delta_abs": tps_delta_abs,
            "tps_delta_ratio": tps_delta_ratio,
        }

    evaluated["baseline_lookup_key"] = baseline_lookup_key
    evaluated["baseline_tps"] = baseline_tps
    evaluated["execution"] = execution
    evaluated["performance_comparison"] = performance_comparison
    return evaluated


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    parse_success = sum(1 for row in rows if bool(row.get("parse_ok")))
    validation_attempts = sum(
        1
        for row in rows
        if isinstance(row.get("validation"), dict) and row["validation"].get("performed") is True
    )
    validation_success = sum(
        1
        for row in rows
        if isinstance(row.get("validation"), dict) and row["validation"].get("valid") is True
    )
    baseline_lookup_hits = sum(1 for row in rows if row.get("baseline_tps") is not None)
    execution_attempts = sum(
        1
        for row in rows
        if isinstance(row.get("execution"), dict) and row["execution"].get("attempted") is True
    )
    execution_success = sum(
        1
        for row in rows
        if isinstance(row.get("execution"), dict) and row["execution"].get("success") is True
    )
    performance_comparison_attempts = sum(
        1
        for row in rows
        if isinstance(row.get("performance_comparison"), dict)
        and row["performance_comparison"].get("performed") is True
    )
    better_than_baseline = sum(
        1
        for row in rows
        if isinstance(row.get("performance_comparison"), dict)
        and row["performance_comparison"].get("better_than_baseline") is True
    )

    return {
        "total_samples": total,
        "parse_success": parse_success,
        "parse_success_rate": round(parse_success / total, 4) if total else 0.0,
        "validation_attempts": validation_attempts,
        "validation_success": validation_success,
        "validation_success_rate": round(validation_success / validation_attempts, 4)
        if validation_attempts
        else 0.0,
        "baseline_lookup_hits": baseline_lookup_hits,
        "baseline_lookup_hit_rate": round(baseline_lookup_hits / total, 4) if total else 0.0,
        "execution_attempts": execution_attempts,
        "execution_success": execution_success,
        "execution_success_rate": round(execution_success / execution_attempts, 4)
        if execution_attempts
        else 0.0,
        "performance_comparison_attempts": performance_comparison_attempts,
        "better_than_baseline": better_than_baseline,
        "better_than_baseline_rate": round(
            better_than_baseline / performance_comparison_attempts, 4
        )
        if performance_comparison_attempts
        else 0.0,
    }


def build_plot_data(rows: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    series: list[dict[str, Any]] = []
    for row in rows:
        series.append(
            {
                "sample_id": row.get("sample_id"),
                "workload_id": row.get("baseline_lookup_key") or normalize_workload_id(
                    row.get("workload_file") or row.get("workload")
                ),
                "baseline_tps": row.get("baseline_tps"),
                "predicted_tps": (
                    row.get("execution", {}).get("tps")
                    if isinstance(row.get("execution"), dict)
                    else None
                ),
                "avg_latency_ms": (
                    row.get("execution", {}).get("avg_latency_ms")
                    if isinstance(row.get("execution"), dict)
                    else None
                ),
                "relative_score": (
                    row.get("execution", {}).get("relative_score")
                    if isinstance(row.get("execution"), dict)
                    else None
                ),
                "tps_delta_abs": (
                    row.get("performance_comparison", {}).get("tps_delta_abs")
                    if isinstance(row.get("performance_comparison"), dict)
                    else None
                ),
                "tps_delta_ratio": (
                    row.get("performance_comparison", {}).get("tps_delta_ratio")
                    if isinstance(row.get("performance_comparison"), dict)
                    else None
                ),
                "parse_ok": row.get("parse_ok"),
                "validation_valid": (
                    row.get("validation", {}).get("valid")
                    if isinstance(row.get("validation"), dict)
                    else None
                ),
                "execution_success": (
                    row.get("execution", {}).get("success")
                    if isinstance(row.get("execution"), dict)
                    else None
                ),
            }
        )

    return {
        "metrics": metrics,
        "series": series,
    }


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, payload: dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate surrogate prediction JSONL against baseline records")
    parser.add_argument(
        "--prediction-file",
        type=str,
        required=True,
        help="Prediction JSONL generated by surrogate/inference.py",
    )
    parser.add_argument(
        "--baseline-file",
        type=str,
        required=True,
        help="baseline_records.jsonl file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional database config INI for dynamic execution evaluation",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        choices=["job", "tpch", "ssb"],
        help="Override database name from config.ini for dynamic execution",
    )
    parser.add_argument(
        "--execute-predictions",
        action="store_true",
        help="Apply standardized_config to PostgreSQL and execute workload during evaluation",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path for evaluated JSONL output",
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Optional path for summary JSON output",
    )
    parser.add_argument(
        "--plot-data-file",
        type=str,
        default=None,
        help="Optional JSON output for plotting interfaces",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.execute_predictions and not args.config:
        raise ValueError("dynamic execution requires --config")

    prediction_path = args.prediction_file
    prediction_stem = Path(prediction_path).stem
    output_file = args.output_file or os.path.join(
        "surrogate",
        "evaluations",
        f"{prediction_stem}_evaluated.jsonl",
    )
    summary_file = args.summary_file or os.path.join(
        "surrogate",
        "evaluations",
        f"{prediction_stem}_summary.json",
    )
    plot_data_file = args.plot_data_file or os.path.join(
        "surrogate",
        "evaluations",
        f"{prediction_stem}_plot_data.json",
    )

    predictions = load_predictions(prediction_path)
    baseline_lookup = load_baseline_records(args.baseline_file)
    execution_context = (
        build_execution_context_with_db_override(args.config, db_override=args.db)
        if args.execute_predictions
        else OptionalExecutionContext()
    )

    try:
        evaluated_rows = [
            evaluate_prediction_row(
                row,
                baseline_lookup,
                execution_context=execution_context,
                execute_predictions=args.execute_predictions,
            )
            for row in predictions
        ]
    finally:
        execution_context.close()

    metrics = compute_metrics(evaluated_rows)
    plot_data = build_plot_data(evaluated_rows, metrics)

    save_jsonl(output_file, evaluated_rows)
    save_json(summary_file, metrics)
    save_json(plot_data_file, plot_data)
    LOGGER.info("Evaluation summary: %s", metrics)


if __name__ == "__main__":
    main()
