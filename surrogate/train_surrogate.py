"""
Dependencies: transformers, peft, datasets, torch, numpy
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass
from glob import glob
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from datasets import DatasetDict
from peft import LoraConfig
from peft import TaskType
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed


LOGGER = logging.getLogger("surrogate.train_surrogate")
DEFAULT_MODEL_ALIAS = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_MODEL_PATH = "/root/autodl-tmp/llm/data/models/Qwen2.5-7B-Instruct"
REQUIRED_FIELDS = ("instruction", "input", "output")


@dataclass
class TrainingRecord:
    prompt: str
    response: str
    source_file: str
    line_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "source_file": self.source_file,
            "line_number": self.line_number,
        }


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def resolve_model_name_or_path(model_name_or_path: str) -> str:
    if model_name_or_path == DEFAULT_MODEL_ALIAS:
        if not os.path.exists(DEFAULT_LOCAL_MODEL_PATH):
            raise FileNotFoundError(
                f"Default local model path does not exist: {DEFAULT_LOCAL_MODEL_PATH}"
            )
        LOGGER.info(
            "Resolved default model alias %s to local path %s",
            DEFAULT_MODEL_ALIAS,
            DEFAULT_LOCAL_MODEL_PATH,
        )
        return DEFAULT_LOCAL_MODEL_PATH

    LOGGER.info(
        "Using explicit model path or cache identifier: %s (local_files_only=True)",
        model_name_or_path,
    )
    return model_name_or_path


def discover_training_files(data_dir: str) -> list[str]:
    pattern = os.path.join(data_dir, "training_sft_data_*.jsonl")
    files = sorted(
        path
        for path in glob(pattern)
        if os.path.basename(path) != "dataset_stats.json"
    )
    if not files:
        raise FileNotFoundError(
            f"No training files matching training_sft_data_*.jsonl were found in {data_dir}"
        )
    LOGGER.info("Discovered %d training file(s) under %s", len(files), data_dir)
    for path in files:
        LOGGER.info("Training data file: %s", path)
    return files


def _normalize_output_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False)


def load_sft_records(data_dir: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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

                missing_fields = [
                    field_name for field_name in REQUIRED_FIELDS if field_name not in payload
                ]
                if missing_fields:
                    raise ValueError(
                        f"Missing field(s) {missing_fields} in {file_path}:{line_number}"
                    )

                instruction = str(payload["instruction"]).strip()
                input_text = str(payload["input"]).strip()
                output_text = _normalize_output_text(payload["output"])

                if not instruction or not input_text or not output_text:
                    LOGGER.warning(
                        "Skipping empty record from %s:%d",
                        file_path,
                        line_number,
                    )
                    continue

                record = TrainingRecord(
                    prompt=f"{instruction}\n\n{input_text}",
                    response=output_text,
                    source_file=file_path,
                    line_number=line_number,
                )
                records.append(record.to_dict())

    if not records:
        raise ValueError(f"No usable SFT records were loaded from {data_dir}")

    LOGGER.info("Loaded %d usable SFT record(s)", len(records))
    return records


def _compute_split_counts(total_size: int) -> tuple[int, int, int]:
    if total_size < 3:
        raise ValueError(
            f"At least 3 records are required for an 8:1:1 split, but got {total_size}"
        )

    target = np.array([0.8, 0.1, 0.1], dtype=float) * float(total_size)
    counts = np.floor(target).astype(int)
    remainders = target - counts

    while int(counts.sum()) < total_size:
        index = int(np.argmax(remainders))
        counts[index] += 1
        remainders[index] = -1.0

    for index in range(3):
        if counts[index] > 0:
            continue
        donor_index = int(np.argmax(counts))
        if counts[donor_index] <= 1:
            raise ValueError(
                f"Unable to allocate a non-empty split for {total_size} record(s)"
            )
        counts[donor_index] -= 1
        counts[index] += 1

    train_count, val_count, test_count = (int(value) for value in counts.tolist())
    return train_count, val_count, test_count


def build_dataset_dict(records: list[dict[str, Any]], random_seed: int = 42) -> DatasetDict:
    if not records:
        raise ValueError("Cannot build DatasetDict from an empty record list")

    train_count, val_count, test_count = _compute_split_counts(len(records))
    indices = np.random.default_rng(random_seed).permutation(len(records))
    shuffled_records = [records[int(index)] for index in indices]

    train_records = shuffled_records[:train_count]
    val_records = shuffled_records[train_count:train_count + val_count]
    test_records = shuffled_records[train_count + val_count:]

    if len(test_records) != test_count:
        raise RuntimeError(
            f"Unexpected test split size: expected {test_count}, got {len(test_records)}"
        )

    LOGGER.info(
        "Dataset split sizes -> train: %d, val: %d, test: %d",
        len(train_records),
        len(val_records),
        len(test_records),
    )

    return DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "val": Dataset.from_list(val_records),
            "test": Dataset.from_list(test_records),
        }
    )


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
    tokenizer.padding_side = "right"
    return tokenizer


def _tokenize_batch(
    batch: dict[str, list[Any]],
    tokenizer,
    max_length: int,
) -> dict[str, list[list[int]]]:
    if max_length < 2:
        raise ValueError(f"max_length must be >= 2, but got {max_length}")

    eos_token_id = tokenizer.eos_token_id
    input_id_batch: list[list[int]] = []
    attention_mask_batch: list[list[int]] = []
    label_batch: list[list[int]] = []

    prompt_texts = batch["prompt"]
    response_texts = batch["response"]

    for prompt_text, response_text in zip(prompt_texts, response_texts):
        prompt_ids = tokenizer(str(prompt_text), add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(str(response_text), add_special_tokens=False)["input_ids"]

        max_response_length = max_length - 1
        if len(response_ids) > max_response_length:
            response_ids = response_ids[:max_response_length]

        prompt_budget = max_length - len(response_ids) - 1
        prompt_budget = max(prompt_budget, 0)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[:prompt_budget]

        if not response_ids:
            continue

        input_ids = prompt_ids + response_ids + [eos_token_id]
        labels = ([-100] * len(prompt_ids)) + response_ids + [eos_token_id]
        attention_mask = [1] * len(input_ids)

        if not any(label != -100 for label in labels):
            continue

        input_id_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        label_batch.append(labels)

    return {
        "input_ids": input_id_batch,
        "attention_mask": attention_mask_batch,
        "labels": label_batch,
    }


def tokenize_dataset_dict(
    dataset_dict: DatasetDict,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    tokenized_splits: dict[str, Dataset] = {}

    for split_name, split_dataset in dataset_dict.items():
        tokenized_dataset = split_dataset.map(
            lambda batch: _tokenize_batch(batch, tokenizer=tokenizer, max_length=max_length),
            batched=True,
            remove_columns=split_dataset.column_names,
            desc=f"Tokenizing {split_name} split",
        )
        if len(tokenized_dataset) == 0:
            raise ValueError(f"All samples in split '{split_name}' were filtered out")
        tokenized_splits[split_name] = tokenized_dataset
        LOGGER.info("Tokenized split '%s' -> %d sample(s)", split_name, len(tokenized_dataset))

    return DatasetDict(tokenized_splits)


def load_model_with_lora(model_name_or_path: str, lora_r: int):
    resolved_model_path = resolve_model_name_or_path(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    trainable_parameters = 0
    total_parameters = 0
    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total_parameters += parameter_count
        if parameter.requires_grad:
            trainable_parameters += parameter_count

    LOGGER.info(
        "Loaded model with LoRA. Trainable parameters: %d / %d (%.4f%%)",
        trainable_parameters,
        total_parameters,
        (trainable_parameters / total_parameters) * 100.0,
    )
    return model


def build_training_arguments(
    output_dir: str,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainingArguments:
    training_kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": 8,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "bf16": True,
        "fp16": False,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "seed": seed,
        "report_to": [],
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**training_kwargs)


class SurrogateModelTrainer:
    def __init__(
        self,
        database: str,
        model_name_or_path: str = DEFAULT_MODEL_ALIAS,
        data_dir: str = "training_data",
        output_dir: str = "surrogate/checkpoints",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        max_length: int = 2048,
        random_seed: int = 42,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
    ) -> None:
        self.database = database
        self.model_name_or_path = model_name_or_path
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, database)
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.max_length = max_length
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.lora_r = lora_r

    def train(self) -> str:
        set_seed(self.random_seed)
        os.makedirs(self.output_dir, exist_ok=True)

        records = load_sft_records(self.data_dir)
        dataset_dict = build_dataset_dict(records, random_seed=self.random_seed)
        tokenizer = load_tokenizer(self.model_name_or_path)
        tokenized_dataset_dict = tokenize_dataset_dict(
            dataset_dict,
            tokenizer=tokenizer,
            max_length=self.max_length,
        )
        model = load_model_with_lora(
            self.model_name_or_path,
            lora_r=self.lora_r,
        )
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        )
        training_args = build_training_arguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            seed=self.random_seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset_dict["train"],
            eval_dataset=tokenized_dataset_dict["val"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        LOGGER.info("Starting Qwen2.5 + LoRA SFT training")
        train_result = trainer.train()
        LOGGER.info("Training metrics: %s", train_result.metrics)

        test_metrics = trainer.evaluate(
            eval_dataset=tokenized_dataset_dict["test"],
            metric_key_prefix="test",
        )
        LOGGER.info("Test metrics: %s", test_metrics)

        trainer.model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        trainer.save_state()
        LOGGER.info("Saved LoRA adapter and tokenizer to %s", self.output_dir)
        return self.output_dir


def train_surrogate(database: str) -> str:
    trainer = SurrogateModelTrainer(database=database)
    return trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen2.5 with LoRA on SFT tuning data")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ALIAS,
        help=(
            "Model path or HuggingFace ID. "
            f"The default alias is always resolved to the local path {DEFAULT_LOCAL_MODEL_PATH}."
        ),
    )
    parser.add_argument(
        "--database",
        type=str,
        default="tpch",
        help="Used for naming the checkpoint directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training_data",
        help="Directory containing training_sft_data_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="surrogate/checkpoints",
        help="Checkpoint output root directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train and eval batch size",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length after tokenization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling and training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    trainer = SurrogateModelTrainer(
        database=args.database,
        model_name_or_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        max_length=args.max_length,
        random_seed=args.seed,
        learning_rate=args.lr,
        lora_r=args.lora_r,
    )
    final_output_dir = trainer.train()
    LOGGER.info("Training completed successfully. Output directory: %s", final_output_dir)


if __name__ == "__main__":
    main()
