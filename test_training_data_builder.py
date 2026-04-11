import json

from training_data_builder import BuilderConfig, TrainingDataBuilder


def test_select_high_quality_samples_with_dedup(tmp_path):
    builder = TrainingDataBuilder(
        offline_sample_path="unused.jsonl",
        output_path=str(tmp_path / "training.jsonl"),
        builder_config=BuilderConfig(
            min_samples=1,
            max_samples=10,
            deduplicate=True,
            min_tps=0.1,
            top_fraction=1.0,
            secondary_fraction=0.0,
        ),
        random_seed=7,
    )
    builder.samples = [
        {"workload": "w1", "config": {"shared_buffers": 1}, "tps": 10.0},
        {"workload": "w1", "config": {"shared_buffers": 2}, "tps": 9.0},
        {"workload": "w1", "config": {"shared_buffers": 3}, "tps": 0.05},
        {"workload": "w2", "config": {"shared_buffers": 1}, "tps": 8.0},
        {"workload": "w2", "config": {"shared_buffers": 4}, "tps": 7.0},
    ]

    selected = builder.select_high_quality_samples()

    assert len(selected) == 3
    assert all(float(sample["tps"]) >= 0.1 for sample in selected)
    assert builder.deduplication_removed == 1
    assert len({json.dumps(sample["config"], sort_keys=True) for sample in selected}) == 3


def test_format_config_as_output_raw_and_human(tmp_path):
    config = {
        "shared_buffers": 1024,
        "checkpoint_completion_target": 0.5,
        "max_connections": 100,
    }

    raw_builder = TrainingDataBuilder(
        offline_sample_path="unused.jsonl",
        output_path=str(tmp_path / "raw.jsonl"),
        builder_config=BuilderConfig(output_format="raw"),
    )
    human_builder = TrainingDataBuilder(
        offline_sample_path="unused.jsonl",
        output_path=str(tmp_path / "human.jsonl"),
        builder_config=BuilderConfig(output_format="human"),
    )

    raw_output = json.loads(raw_builder.format_config_as_output(config))
    human_output = json.loads(human_builder.format_config_as_output(config))

    assert raw_output["shared_buffers"] == 1024
    assert raw_output["checkpoint_completion_target"] == 0.5
    assert isinstance(raw_output["max_connections"], int)

    assert human_output["shared_buffers"] == "1.0MB"
    assert human_output["checkpoint_completion_target"] == "0.50"
    assert human_output["max_connections"] == "100"
    assert all(isinstance(value, str) for value in human_output.values())


def test_output_json_validation(tmp_path):
    builder = TrainingDataBuilder(
        offline_sample_path="unused.jsonl",
        output_path=str(tmp_path / "training.jsonl"),
    )

    assert builder.validate_output_json('{"a": 1}', sample_index=1) is True
    assert builder.validation_errors == 0

    assert builder.validate_output_json('{"a": }', sample_index=2) is False
    assert builder.validation_errors == 1
