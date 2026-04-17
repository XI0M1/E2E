"""
Analyze token length distribution for SFT training data without loading model weights.

Usage:
  python tools/analyze_token_distribution.py --max-length 768 --prompt-floor 200
  python tools/analyze_token_distribution.py --max-length 512 --prompt-floor 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surrogate.train_surrogate import DEFAULT_MODEL_ALIAS
from surrogate.train_surrogate import _normalize_output_text
from surrogate.train_surrogate import discover_training_files
from surrogate.train_surrogate import load_tokenizer


def load_raw_records(data_dir: str) -> list[dict]:
    records: list[dict] = []
    for file_path in discover_training_files(data_dir):
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

                if "instruction" not in payload or "input" not in payload or "output" not in payload:
                    raise ValueError(
                        f"Missing required fields in {file_path}:{line_number}"
                    )

                instruction = str(payload["instruction"]).strip()
                input_text = str(payload["input"]).strip()
                response = _normalize_output_text(payload["output"])
                prompt = f"{instruction}\n\n{input_text}\n\nOutput JSON:"

                if not prompt.strip() or not response.strip():
                    continue

                records.append(
                    {
                        "prompt": prompt,
                        "response": response,
                    }
                )

    if not records:
        raise ValueError(f"No usable records found under {data_dir}")
    return records


def simulate_truncation(
    prompt_ids,
    response_ids,
    max_length,
    prompt_floor,
) -> tuple[int, int, bool, bool]:
    prompt_budget = max(prompt_floor, max_length - len(response_ids) - 1)
    prompt_budget = min(prompt_budget, max_length - 2)
    prompt_budget = max(prompt_budget, 0)

    truncated_prompt_ids = prompt_ids[:prompt_budget]

    response_budget = max_length - min(len(prompt_ids), prompt_budget) - 1
    response_budget = max(response_budget, 1)
    truncated_response_ids = response_ids[:response_budget]

    actual_prompt_len = len(truncated_prompt_ids)
    actual_response_len = len(truncated_response_ids)
    prompt_was_cut = actual_prompt_len < len(prompt_ids)
    response_was_cut = actual_response_len < len(response_ids)
    return actual_prompt_len, actual_response_len, prompt_was_cut, response_was_cut


def _percentile_summary(values: list[int]) -> str:
    array = np.asarray(values, dtype=float)
    points = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    percentile_values = np.percentile(array, points)
    return (
        f"mean={array.mean():.1f}  "
        + "  ".join(
            f"p{point}={int(round(value))}"
            for point, value in zip(points, percentile_values)
        )
    )


def _coverage_threshold(values: list[int], coverage: int) -> int:
    array = np.asarray(values, dtype=float)
    return int(np.ceil(np.percentile(array, coverage)))


def analyze(data_dir, max_length, prompt_floor, model_path) -> None:
    records = load_raw_records(data_dir)
    tokenizer = load_tokenizer(model_path)

    first_prompt = str(records[0]["prompt"])
    first_instruction = first_prompt.split("\n\n", 1)[0]
    instruction_token_count = len(
        tokenizer(first_instruction, add_special_tokens=False)["input_ids"]
    )

    raw_prompt_lengths: list[int] = []
    raw_response_lengths: list[int] = []
    raw_total_lengths: list[int] = []
    actual_prompt_lengths: list[int] = []
    prompt_cut_count = 0
    response_cut_count = 0
    prompt_lt_256_count = 0
    prompt_le_instruction_count = 0
    fit_without_truncation_count = 0

    eos_tokens = 1

    for record in records:
        prompt_ids = tokenizer(
            str(record["prompt"]),
            add_special_tokens=False,
        )["input_ids"]
        response_ids = tokenizer(
            str(record["response"]),
            add_special_tokens=False,
        )["input_ids"]

        raw_prompt_len = len(prompt_ids)
        raw_response_len = len(response_ids)
        raw_total_len = raw_prompt_len + raw_response_len + eos_tokens

        raw_prompt_lengths.append(raw_prompt_len)
        raw_response_lengths.append(raw_response_len)
        raw_total_lengths.append(raw_total_len)

        if raw_total_len <= max_length:
            fit_without_truncation_count += 1

        (
            actual_prompt_len,
            actual_response_len,
            prompt_was_cut,
            response_was_cut,
        ) = simulate_truncation(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            max_length=max_length,
            prompt_floor=prompt_floor,
        )

        actual_prompt_lengths.append(actual_prompt_len)
        if prompt_was_cut:
            prompt_cut_count += 1
        if response_was_cut:
            response_cut_count += 1
        if actual_prompt_len < 256:
            prompt_lt_256_count += 1
        if actual_prompt_len <= instruction_token_count:
            prompt_le_instruction_count += 1

        _ = actual_response_len

    total_records = len(records)

    print("─── Raw token lengths (before truncation) ───")
    print(f"  Prompt   : {_percentile_summary(raw_prompt_lengths)}")
    print(f"  Response : {_percentile_summary(raw_response_lengths)}")
    print(f"  Total    : {_percentile_summary(raw_total_lengths)}")
    print()

    print(f"─── After truncation (max_length={max_length}, prompt_floor={prompt_floor}) ───")
    print(f"  Actual prompt len     : {_percentile_summary(actual_prompt_lengths)}")
    print(
        f"  Prompt cut            : {prompt_cut_count}/{total_records} "
        f"({prompt_cut_count / total_records * 100:.1f}%)"
    )
    print(
        f"  Response cut          : {response_cut_count}/{total_records} "
        f"({response_cut_count / total_records * 100:.1f}%)"
    )
    print(
        f"  Prompt < 256 tokens   : {prompt_lt_256_count}/{total_records} "
        f"({prompt_lt_256_count / total_records * 100:.1f}%)"
    )
    print(
        f"  Prompt ≤ instruction  : {prompt_le_instruction_count}/{total_records} "
        f"({prompt_le_instruction_count / total_records * 100:.1f}%)"
    )
    print()

    print("─── Samples that fit WITHOUT truncation ───")
    print(
        f"  {fit_without_truncation_count}/{total_records} "
        f"({fit_without_truncation_count / total_records * 100:.1f}%) fit within {max_length} tokens"
    )
    print()

    print("─── Recommended max_length thresholds ───")
    for coverage in (80, 90, 95, 99):
        threshold = _coverage_threshold(raw_total_lengths, coverage)
        print(f"  {coverage}% coverage → max_length = {threshold}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution and truncation effects for SFT data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training_data",
        help="Training data directory",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
        help="Simulated max_length",
    )
    parser.add_argument(
        "--prompt-floor",
        type=int,
        default=200,
        help="Simulated prompt_floor",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        help="Tokenizer path or alias",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(
        data_dir=args.data_dir,
        max_length=args.max_length,
        prompt_floor=args.prompt_floor,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
