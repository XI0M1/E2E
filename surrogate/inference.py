"""
Dependencies: transformers, peft, torch, numpy
Optional validation dependency when --config is provided: psycopg2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surrogate.train_surrogate import DEFAULT_MODEL_ALIAS
from surrogate.train_surrogate import DEFAULT_LOCAL_MODEL_PATH
from surrogate.train_surrogate import discover_training_files
from surrogate.train_surrogate import resolve_model_name_or_path
from training_data_builder import TrainingDataBuilder


LOGGER = logging.getLogger("surrogate.inference")
DEFAULT_KNOB_CONFIG_PATH = "knob_config/knob_config.json"


@dataclass
class InferenceSample:
    instruction: str
    input_text: str
    source_file: str
    line_number: int
    sample_id: str
    expected_output: str | None = None
    workload: str | None = None
    workload_file: str | None = None
    workload_type: str | None = None
    baseline_tps: float | None = None

    @property
    def prompt(self) -> str:
        return f"{self.instruction}\n\n{self.input_text}\n\nOutput JSON:"


@dataclass
class OptionalValidatorContext:
    validator: Any | None = None
    database: Any | None = None

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


def _normalize_output_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _discover_input_files(data_dir: str, input_file: str | None) -> list[str]:
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        return [input_file]
    return discover_training_files(data_dir)


def load_inference_samples(
    data_dir: str,
    input_file: str | None = None,
    limit: int | None = None,
) -> list[InferenceSample]:
    samples: list[InferenceSample] = []
    for file_path in _discover_input_files(data_dir=data_dir, input_file=input_file):
        with open(file_path, "r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                stripped_line = raw_line.strip()
                if not stripped_line:
                    continue

                try:
                    payload = json.loads(stripped_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {file_path}:{line_number}: {exc}"
                    ) from exc

                if "instruction" not in payload or "input" not in payload:
                    raise ValueError(
                        f"Missing required fields instruction/input in {file_path}:{line_number}"
                    )

                instruction = str(payload["instruction"]).strip()
                input_text = str(payload["input"]).strip()
                if not instruction or not input_text:
                    LOGGER.warning(
                        "Skipping empty inference sample from %s:%d",
                        file_path,
                        line_number,
                    )
                    continue

                sample = InferenceSample(
                    instruction=instruction,
                    input_text=input_text,
                    source_file=file_path,
                    line_number=line_number,
                    sample_id=f"{os.path.basename(file_path)}:{line_number}",
                    expected_output=_normalize_output_text(payload.get("output")),
                    workload=_normalize_optional_text(
                        payload.get("workload") or payload.get("workload_path")
                    ),
                    workload_file=_normalize_optional_text(payload.get("workload_file")),
                    workload_type=_normalize_optional_text(payload.get("workload_type")),
                    baseline_tps=_normalize_optional_float(payload.get("baseline_tps")),
                )
                samples.append(sample)

                if limit is not None and len(samples) >= limit:
                    LOGGER.info("Reached inference limit=%d", limit)
                    return samples

    if not samples:
        raise ValueError("No usable inference samples were loaded")

    LOGGER.info("Loaded %d inference sample(s)", len(samples))
    return samples


def load_tokenizer(model_name_or_path: str):
    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.eos_token is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide eos_token and eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return tokenizer


def load_model_and_adapter(
    model_name_or_path: str,
    adapter_path: str,
):
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        local_files_only=True,
    )
    model.eval()
    LOGGER.info(
        "Loaded base model from %s and LoRA adapter from %s",
        resolved_model_path,
        adapter_path,
    )
    return model


def load_knob_specs(knob_config_path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(knob_config_path):
        raise FileNotFoundError(f"Knob config file does not exist: {knob_config_path}")

    with open(knob_config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Knob config at {knob_config_path} is not a JSON object")

    LOGGER.info("Loaded %d knob definitions from %s", len(payload), knob_config_path)
    return payload


def build_optional_validator(config_path: str | None) -> OptionalValidatorContext:
    if not config_path:
        LOGGER.info("Database-backed validation disabled (no --config provided)")
        return OptionalValidatorContext()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Validation config file does not exist: {config_path}")

    from config.parse_config import parse_args as parse_ini_config
    from Database import Database
    from parameter_validation import ParameterConstraintValidator

    config = parse_ini_config(config_path)
    database = Database(config)
    validator = ParameterConstraintValidator(database, LOGGER)
    LOGGER.info("Database-backed validation enabled via %s", config_path)
    return OptionalValidatorContext(validator=validator, database=database)


def generate_response_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    tokenized = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    model_device = next(model.parameters()).device
    tokenized = {key: value.to(model_device) for key, value in tokenized.items()}

    generation_kwargs: dict[str, Any] = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated_ids = model.generate(**generation_kwargs)

    prompt_token_count = tokenized["input_ids"].shape[1]
    new_token_ids = generated_ids[0][prompt_token_count:]
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return generated_text


def parse_generated_json(generated_text: str) -> tuple[dict[str, Any] | None, str | None, str]:
    if not generated_text:
        return None, "Generated text is empty", "empty"

    try:
        payload = json.loads(generated_text)
        if isinstance(payload, dict):
            return payload, None, "strict"
        return None, "Generated JSON is not an object", "strict"
    except json.JSONDecodeError:
        pass

    stripped = generated_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(
                line
                for line in lines
                if not line.strip().startswith("```")
            ).strip()
            try:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    return payload, None, "code_fence"
                return None, "Generated fenced JSON is not an object", "code_fence"
            except json.JSONDecodeError:
                pass

    first_brace = generated_text.find("{")
    last_brace = generated_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = generated_text[first_brace:last_brace + 1]
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload, None, "substring"
            return None, "Generated JSON substring is not an object", "substring"
        except json.JSONDecodeError as exc:
            return None, f"Failed to parse extracted JSON substring: {exc}", "substring"

    return None, "No JSON object found in generated text", "none"


def _coerce_numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Numeric value is empty")
    return float(text)


def decode_percentile_knob_value(
    knob_name: str,
    raw_value: Any,
    knob_spec: dict[str, Any],
) -> Any:
    knob_type = str(knob_spec.get("type", "integer"))
    minimum = float(knob_spec.get("min", 0))
    maximum = float(knob_spec.get("max", 0))
    step = float(knob_spec.get("step", 1) or 1)
    range_size = maximum - minimum

    numeric_value = _coerce_numeric(raw_value)

    if numeric_value == -1.0 and minimum <= -1.0 <= maximum:
        return -1

    if knob_type == "float":
        percentile = int(round(max(0.0, min(100.0, numeric_value))))
        decoded_value = TrainingDataBuilder._decode_from_percentile(
            percentile,
            minimum,
            maximum,
            step=step,
        )
        precision = 0
        step_text = f"{step:.12f}".rstrip("0").rstrip(".")
        if "." in step_text:
            precision = len(step_text.split(".")[1])
        return round(float(decoded_value), precision)

    if knob_type == "integer":
        if range_size <= 20:
            actual_value = int(round(numeric_value))
            return max(int(round(minimum)), min(int(round(maximum)), actual_value))

        percentile = int(round(max(0.0, min(100.0, numeric_value))))
        decoded_value = TrainingDataBuilder._decode_from_percentile(
            percentile,
            minimum,
            maximum,
            step=step,
        )
        return int(round(decoded_value))

    if knob_type == "bool":
        return bool(int(round(numeric_value)))

    return raw_value


def decode_generated_knob_config(
    generated_payload: dict[str, Any],
    knob_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    decoded: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    dropped_unknown_knobs: list[str] = []

    for knob_name, raw_value in generated_payload.items():
        knob_spec = knob_specs.get(knob_name)
        if knob_spec is None:
            dropped_unknown_knobs.append(knob_name)
            issues.append(
                {
                    "severity": "warning",
                    "rule": "decoder.unknown_knob",
                    "message": f"Dropped unknown knob generated by the model: {knob_name}",
                    "parameter": knob_name,
                }
            )
            continue

        try:
            decoded[knob_name] = decode_percentile_knob_value(
                knob_name=knob_name,
                raw_value=raw_value,
                knob_spec=knob_spec,
            )
        except Exception as exc:
            issues.append(
                {
                    "severity": "error",
                    "rule": "decoder.decode_failed",
                    "message": f"Failed to decode knob {knob_name}: {exc}",
                    "parameter": knob_name,
                }
            )

    return decoded, issues, dropped_unknown_knobs


def validate_standardized_config(
    standardized_config: dict[str, Any],
    validator_context: OptionalValidatorContext,
) -> dict[str, Any]:
    if validator_context.validator is None:
        return {
            "performed": False,
            "valid": None,
            "normalized_config": standardized_config,
            "issues": [],
            "derived": {},
        }

    result = validator_context.validator.validate_payload(standardized_config)
    return {
        "performed": True,
        "valid": result.valid,
        "normalized_config": result.normalized_config,
        "issues": [issue.to_dict() for issue in result.issues],
        "derived": result.derived,
    }


def normalize_json_object(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
        return None
    return None


def build_skipped_execution(error: str | None) -> dict[str, Any]:
    return {
        "attempted": False,
        "applied": False,
        "success": False,
        "error": error,
        "tps": None,
        "avg_latency_ms": None,
        "workload_type": None,
        "relative_score": None,
        "baseline_tps": None,
    }


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


def find_workload_by_basename(basename: str) -> str | None:
    if not basename:
        return None

    search_roots = [
        os.path.join(str(PROJECT_ROOT), "data"),
        os.path.join(str(PROJECT_ROOT), "SuperWG"),
        str(PROJECT_ROOT),
    ]
    for search_root in search_roots:
        if not os.path.isdir(search_root):
            continue
        for root, _, files in os.walk(search_root):
            if basename in files:
                return os.path.join(root, basename)
    return None


def resolve_workload_path_for_sample(
    sample: InferenceSample,
    runtime_config: dict[str, Any],
) -> tuple[str | None, str | None]:
    candidate_values = [
        sample.workload,
        sample.workload_file,
        runtime_config.get("benchmark_config", {}).get("workload_path"),
    ]

    for candidate in candidate_values:
        resolved = resolve_existing_path(candidate, source_file=sample.source_file)
        if resolved:
            return resolved, None

    for candidate in [sample.workload, sample.workload_file]:
        basename = os.path.basename(str(candidate or "").strip())
        if not basename:
            continue
        resolved = find_workload_by_basename(basename)
        if resolved:
            return resolved, None

    return None, (
        "Execution skipped: unable to determine workload path from sample metadata "
        "or benchmark_config.workload_path"
    )


def build_execution_sample_path(sample: InferenceSample, suffix: str) -> str:
    output_dir = os.path.join("surrogate", "predictions", "execution_artifacts")
    os.makedirs(output_dir, exist_ok=True)
    safe_id = sample.sample_id.replace("\\", "_").replace("/", "_").replace(":", "_")
    return os.path.join(output_dir, f"{safe_id}_{suffix}_{os.getpid()}")


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


def run_workload_measurement(
    config_dict: dict[str, Any],
    runtime_config: dict[str, Any],
    sample: InferenceSample,
    workload_path: str,
    sample_path_suffix: str,
) -> dict[str, Any]:
    from Database import Database
    from stress_testing_tool import stress_testing_tool

    benchmark_config = dict(runtime_config.get("benchmark_config", {}))
    benchmark_config["workload_path"] = workload_path
    execution_config = dict(runtime_config)
    execution_config["benchmark_config"] = benchmark_config

    database = None
    sample_path = build_execution_sample_path(sample, sample_path_suffix)
    jsonl_path = sample_path + ".jsonl"
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    try:
        database = Database(execution_config)
        tester = stress_testing_tool(
            execution_config,
            database,
            LOGGER,
            sample_path=sample_path,
        )
        fallback_tps = float(tester.test_config(config_dict))
        record = read_last_json_record(jsonl_path)

        if record is None:
            return {
                "attempted": True,
                "applied": False,
                "success": False,
                "error": "Configuration execution produced no measurable result record",
                "tps": None,
                "avg_latency_ms": None,
                "workload_type": sample.workload_type,
                "relative_score": None,
            }

        measured_tps = _normalize_optional_float(
            record.get("tps", record.get("performance", fallback_tps))
        )
        avg_latency_ms = _normalize_optional_float(record.get("avg_latency_ms"))
        workload_type = _normalize_optional_text(record.get("workload_type")) or sample.workload_type
        relative_score = _normalize_optional_float(record.get("relative_score"))
        success = measured_tps is not None and measured_tps > 0.0

        return {
            "attempted": True,
            "applied": True,
            "success": success,
            "error": None if success else "Workload executed but did not produce a positive TPS measurement",
            "tps": measured_tps,
            "avg_latency_ms": avg_latency_ms,
            "workload_type": workload_type,
            "relative_score": relative_score,
        }
    except Exception as exc:
        LOGGER.exception("Execution evaluation failed for sample %s", sample.sample_id)
        return {
            "attempted": True,
            "applied": False,
            "success": False,
            "error": str(exc),
            "tps": None,
            "avg_latency_ms": None,
            "workload_type": sample.workload_type,
            "relative_score": None,
        }
    finally:
        if database is not None:
            try:
                database.close()
            except Exception:
                pass


def evaluate_predicted_config(
    standardized_config: dict[str, Any],
    sample: InferenceSample,
    config_path: str | None,
    baseline_mode: str = "default_knobs",
) -> dict[str, Any]:
    if not config_path:
        return build_skipped_execution("Execution skipped: --config not provided")

    if not standardized_config:
        return build_skipped_execution("Execution skipped: standardized_config is empty")

    from config.parse_config import parse_args as parse_ini_config

    runtime_config = parse_ini_config(config_path)
    workload_path, workload_error = resolve_workload_path_for_sample(sample, runtime_config)
    if not workload_path:
        return build_skipped_execution(workload_error)

    baseline_tps = None
    if baseline_mode == "metadata":
        baseline_tps = sample.baseline_tps
    elif baseline_mode == "default_knobs":
        baseline_result = run_workload_measurement(
            config_dict={},
            runtime_config=runtime_config,
            sample=sample,
            workload_path=workload_path,
            sample_path_suffix="baseline",
        )
        if baseline_result["success"]:
            baseline_tps = baseline_result["tps"]

    prediction_result = run_workload_measurement(
        config_dict=standardized_config,
        runtime_config=runtime_config,
        sample=sample,
        workload_path=workload_path,
        sample_path_suffix="prediction",
    )
    prediction_result["baseline_tps"] = baseline_tps
    return prediction_result


def evaluate_exact_match(
    expected_output: str | None,
    generated_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not expected_output:
        return {"performed": False, "result": None}

    try:
        expected_payload = normalize_json_object(expected_output)
    except Exception:
        expected_payload = None

    if expected_payload is None or generated_payload is None:
        return {"performed": True, "result": False}

    return {
        "performed": True,
        "result": expected_payload == generated_payload,
    }


def build_performance_comparison(execution: dict[str, Any]) -> dict[str, Any]:
    if not execution.get("success"):
        return {
            "performed": False,
            "better_than_baseline": None,
            "tps_delta_abs": None,
            "tps_delta_ratio": None,
        }

    predicted_tps = _normalize_optional_float(execution.get("tps"))
    baseline_tps = _normalize_optional_float(execution.get("baseline_tps"))
    if predicted_tps is None or baseline_tps is None:
        return {
            "performed": False,
            "better_than_baseline": None,
            "tps_delta_abs": None,
            "tps_delta_ratio": None,
        }

    delta_abs = predicted_tps - baseline_tps
    delta_ratio = None
    if baseline_tps > 1e-6:
        delta_ratio = delta_abs / baseline_tps

    return {
        "performed": True,
        "better_than_baseline": predicted_tps > baseline_tps,
        "tps_delta_abs": delta_abs,
        "tps_delta_ratio": delta_ratio,
    }


def save_results_jsonl(output_path: str, results: list[dict[str, Any]]) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    LOGGER.info("Saved %d inference result(s) to %s", len(results), output_path)


def run_inference(
    model_name_or_path: str,
    adapter_path: str,
    samples: list[InferenceSample],
    knob_specs: dict[str, dict[str, Any]],
    output_path: str,
    max_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    config_path: str | None = None,
    execute_predictions: bool = False,
    baseline_mode: str = "metadata",
) -> dict[str, Any]:
    set_seed(seed)
    tokenizer = load_tokenizer(model_name_or_path)
    model = load_model_and_adapter(model_name_or_path, adapter_path)
    validator_context = build_optional_validator(config_path)

    results: list[dict[str, Any]] = []
    parse_success = 0
    validation_success = 0
    validation_attempts = 0
    exact_match = 0
    exact_match_comparable = 0
    execution_attempts = 0
    execution_success = 0
    performance_comparison_attempts = 0
    better_than_baseline = 0

    try:
        for index, sample in enumerate(samples, start=1):
            LOGGER.info(
                "Running inference for sample %d/%d (%s)",
                index,
                len(samples),
                sample.sample_id,
            )

            try:
                generated_text = generate_response_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=sample.prompt,
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                parsed_payload, parse_error, parse_mode = parse_generated_json(generated_text)
                parse_ok = parsed_payload is not None
                if parse_ok:
                    parse_success += 1

                decoded_config: dict[str, Any] = {}
                decode_issues: list[dict[str, Any]] = []
                dropped_unknown_knobs: list[str] = []
                if parsed_payload is not None:
                    decoded_config, decode_issues, dropped_unknown_knobs = decode_generated_knob_config(
                        generated_payload=parsed_payload,
                        knob_specs=knob_specs,
                    )

                validation = validate_standardized_config(
                    standardized_config=decoded_config,
                    validator_context=validator_context,
                )
                if validation["performed"]:
                    validation_attempts += 1
                    if validation["valid"]:
                        validation_success += 1

                standardized_config = validation["normalized_config"]
                ready_for_apply = bool(standardized_config) and (
                    not validation["performed"] or bool(validation["valid"])
                )
                exact_match_block = evaluate_exact_match(
                    expected_output=sample.expected_output,
                    generated_payload=parsed_payload,
                )
                if exact_match_block["performed"]:
                    exact_match_comparable += 1
                    if exact_match_block["result"]:
                        exact_match += 1

                if execute_predictions and config_path and ready_for_apply:
                    execution = evaluate_predicted_config(
                        standardized_config=standardized_config,
                        sample=sample,
                        config_path=config_path,
                        baseline_mode=baseline_mode,
                    )
                elif execute_predictions and not config_path:
                    execution = build_skipped_execution(
                        "Execution skipped: --execute-predictions requires --config"
                    )
                elif execute_predictions and not ready_for_apply:
                    execution = build_skipped_execution(
                        "Execution skipped: config is not ready_for_apply"
                    )
                else:
                    execution = build_skipped_execution(
                        "Execution skipped: --execute-predictions not enabled"
                    )

                if execution["attempted"]:
                    execution_attempts += 1
                    if execution["success"]:
                        execution_success += 1

                performance_comparison = build_performance_comparison(execution)
                if performance_comparison["performed"]:
                    performance_comparison_attempts += 1
                    if performance_comparison["better_than_baseline"]:
                        better_than_baseline += 1

                result = {
                    "sample_id": sample.sample_id,
                    "source_file": sample.source_file,
                    "line_number": sample.line_number,
                    "instruction": sample.instruction,
                    "input": sample.input_text,
                    "prompt": sample.prompt,
                    "expected_output": sample.expected_output,
                    "workload": sample.workload,
                    "workload_file": sample.workload_file,
                    "workload_type": sample.workload_type,
                    "baseline_tps": sample.baseline_tps,
                    "generated_text": generated_text,
                    "parse_ok": parse_ok,
                    "parse_error": parse_error,
                    "parse_mode": parse_mode,
                    "generated_payload": parsed_payload,
                    "decoded_config": decoded_config,
                    "decode_issues": decode_issues,
                    "dropped_unknown_knobs": dropped_unknown_knobs,
                    "validation": validation,
                    "standardized_config": standardized_config,
                    "ready_for_apply": ready_for_apply,
                    "execution": execution,
                    "exact_match": exact_match_block,
                    "performance_comparison": performance_comparison,
                }
            except Exception as exc:
                LOGGER.exception("Inference failed for sample %s", sample.sample_id)
                result = {
                    "sample_id": sample.sample_id,
                    "source_file": sample.source_file,
                    "line_number": sample.line_number,
                    "instruction": sample.instruction,
                    "input": sample.input_text,
                    "prompt": sample.prompt,
                    "expected_output": sample.expected_output,
                    "workload": sample.workload,
                    "workload_file": sample.workload_file,
                    "workload_type": sample.workload_type,
                    "baseline_tps": sample.baseline_tps,
                    "generated_text": "",
                    "parse_ok": False,
                    "parse_error": str(exc),
                    "parse_mode": "error",
                    "generated_payload": None,
                    "decoded_config": {},
                    "decode_issues": [],
                    "dropped_unknown_knobs": [],
                    "validation": {
                        "performed": False,
                        "valid": None,
                        "normalized_config": {},
                        "issues": [],
                        "derived": {},
                    },
                    "standardized_config": {},
                    "ready_for_apply": False,
                    "execution": build_skipped_execution(f"Inference failed before execution: {exc}"),
                    "exact_match": {"performed": False, "result": None},
                    "performance_comparison": {
                        "performed": False,
                        "better_than_baseline": None,
                        "tps_delta_abs": None,
                        "tps_delta_ratio": None,
                    },
                }

            results.append(result)
    finally:
        validator_context.close()

    save_results_jsonl(output_path, results)

    total = len(results)
    parse_success_rate = (parse_success / total) if total else 0.0
    validation_success_rate = (
        validation_success / validation_attempts
        if validation_attempts
        else 0.0
    )
    execution_success_rate = (
        execution_success / execution_attempts
        if execution_attempts
        else 0.0
    )
    better_than_baseline_rate = (
        better_than_baseline / performance_comparison_attempts
        if performance_comparison_attempts
        else 0.0
    )
    summary = {
        "total_samples": total,
        "parse_success": parse_success,
        "parse_success_rate": round(parse_success_rate, 4),
        "validation_attempts": validation_attempts,
        "validation_success": validation_success,
        "validation_success_rate": round(validation_success_rate, 4),
        "execution_attempts": execution_attempts,
        "execution_success": execution_success,
        "execution_success_rate": round(execution_success_rate, 4),
        "exact_match_comparable": exact_match_comparable,
        "exact_match": exact_match,
        "exact_match_rate": round(exact_match / exact_match_comparable, 4) if exact_match_comparable else 0.0,
        "performance_comparison_attempts": performance_comparison_attempts,
        "better_than_baseline": better_than_baseline,
        "better_than_baseline_rate": round(better_than_baseline_rate, 4),
        "output_path": output_path,
    }
    LOGGER.info("Inference summary: %s", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA inference for the PostgreSQL surrogate model")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        help=(
            "Base model path or HuggingFace ID. "
            f"The default alias is always resolved to the local path {DEFAULT_LOCAL_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter directory",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="tpch",
        help="Used for naming the default output file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training_data",
        help="Directory containing training_sft_data_*.jsonl files",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Optional explicit JSONL file with instruction/input records",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save inference JSONL results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config INI for database-backed parameter validation",
    )
    parser.add_argument(
        "--knob-config",
        type=str,
        default=DEFAULT_KNOB_CONFIG_PATH,
        help="Knob configuration JSON path",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
        help=(
            "Maximum prompt token length during inference. "
            "Must match the --max-length used during training (default: 768)."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to run",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when --do-sample is enabled",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold when --do-sample is enabled",
    )
    parser.add_argument(
        "--execute-predictions",
        action="store_true",
        help="Actually apply predicted configs and execute the workload",
    )
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="metadata",
        choices=["metadata", "default_knobs"],
        help=(
            "How to obtain baseline_tps for performance comparison: "
            "'metadata' uses baseline_tps from inference sample metadata if present; "
            "'default_knobs' executes the workload with default knob config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    samples = load_inference_samples(
        data_dir=args.data_dir,
        input_file=args.input_file,
        limit=args.limit,
    )
    knob_specs = load_knob_specs(args.knob_config)

    output_file = args.output_file
    if output_file is None:
        output_file = os.path.join(
            "surrogate",
            "predictions",
            f"{args.database}_predictions.jsonl",
        )

    summary = run_inference(
        model_name_or_path=args.model,
        adapter_path=args.adapter_path,
        samples=samples,
        knob_specs=knob_specs,
        output_path=output_file,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        config_path=args.config,
        execute_predictions=args.execute_predictions,
        baseline_mode=args.baseline_mode,
    )
    LOGGER.info("Inference completed successfully: %s", summary)


if __name__ == "__main__":
    main()
