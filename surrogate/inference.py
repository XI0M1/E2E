"""
Dependencies: transformers, peft, torch, numpy
Optional validation dependency when --config is provided: psycopg2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import set_seed

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
) -> dict[str, Any]:
    set_seed(seed)
    tokenizer = load_tokenizer(model_name_or_path)
    model = load_model_and_adapter(model_name_or_path, adapter_path)
    validator_context = build_optional_validator(config_path)

    results: list[dict[str, Any]] = []
    parse_success = 0
    validation_success = 0
    validation_attempts = 0

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
                result = {
                    "sample_id": sample.sample_id,
                    "source_file": sample.source_file,
                    "line_number": sample.line_number,
                    "instruction": sample.instruction,
                    "input": sample.input_text,
                    "prompt": sample.prompt,
                    "expected_output": sample.expected_output,
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
    summary = {
        "total_samples": total,
        "parse_success": parse_success,
        "parse_success_rate": round(parse_success_rate, 4),
        "validation_attempts": validation_attempts,
        "validation_success": validation_success,
        "validation_success_rate": round(validation_success_rate, 4),
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
        default=2048,
        help="Maximum prompt token length during inference",
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
    )
    LOGGER.info("Inference completed successfully: %s", summary)


if __name__ == "__main__":
    main()
