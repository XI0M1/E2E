from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("surrogate.evaluation")


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


def evaluate_prediction_row(
    row: dict[str, Any],
    baseline_lookup: dict[str, float],
) -> dict[str, Any]:
    evaluated = dict(row)
    workload_value = row.get("workload_file")
    baseline_lookup_key = normalize_workload_id(workload_value)
    baseline_tps = baseline_lookup.get(baseline_lookup_key)

    performance_comparison = {
        "performed": False,
        "better_than_baseline": None,
        "tps_delta_abs": None,
        "tps_delta_ratio": None,
    }

    execution = row.get("execution")
    if isinstance(execution, dict):
        try:
            execution_tps = float(execution.get("tps"))
        except (TypeError, ValueError):
            execution_tps = None

        if execution_tps is not None and baseline_tps is not None:
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
        "performance_comparison_attempts": performance_comparison_attempts,
        "better_than_baseline": better_than_baseline,
        "better_than_baseline_rate": round(
            better_than_baseline / performance_comparison_attempts, 4
        )
        if performance_comparison_attempts
        else 0.0,
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
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

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

    predictions = load_predictions(prediction_path)
    baseline_lookup = load_baseline_records(args.baseline_file)
    evaluated_rows = [
        evaluate_prediction_row(row, baseline_lookup)
        for row in predictions
    ]
    metrics = compute_metrics(evaluated_rows)

    save_jsonl(output_file, evaluated_rows)
    save_json(summary_file, metrics)
    LOGGER.info("Evaluation summary: %s", metrics)


if __name__ == "__main__":
    main()
