from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surrogate.train_surrogate import DEFAULT_MODEL_ALIAS
from surrogate.train_surrogate import build_dataset_dict
from surrogate.train_surrogate import load_sft_records
from surrogate.train_surrogate import load_tokenizer
from surrogate.train_surrogate import tokenize_dataset_dict


LOGGER = logging.getLogger("tools.smoke_test_train_surrogate")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    data_dir = PROJECT_ROOT / "training_data"
    records = load_sft_records(str(data_dir))
    if len(records) < 1:
        raise RuntimeError("No training records were loaded")
    LOGGER.info("Loaded %d record(s) from %s", len(records), data_dir)

    dataset_dict = build_dataset_dict(records, random_seed=42)
    tokenizer = load_tokenizer(DEFAULT_MODEL_ALIAS)
    tokenized_dataset_dict = tokenize_dataset_dict(dataset_dict, tokenizer, max_length=2048)

    sample = None
    sample_split = None
    for split_name in ("train", "val", "test"):
        for index in range(len(tokenized_dataset_dict[split_name])):
            candidate = tokenized_dataset_dict[split_name][index]
            labels = candidate["labels"]
            non_masked_token_count = sum(1 for token_id in labels if token_id != -100)
            if -100 in labels and non_masked_token_count > 0:
                sample = candidate
                sample_split = split_name
                break
        if sample is not None:
            break

    if sample is None or sample_split is None:
        raise RuntimeError("No tokenized sample satisfied the smoke test checks")

    labels = sample["labels"]
    non_masked_token_count = sum(1 for token_id in labels if token_id != -100)

    print(f"loaded_records={len(records)}")
    print(f"sample_split={sample_split}")
    print(f"masked_label_present={-100 in labels}")
    print(f"non_masked_label_tokens={non_masked_token_count}")
    print("smoke test passed")


if __name__ == "__main__":
    main()
