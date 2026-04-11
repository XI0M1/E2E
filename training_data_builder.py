"""
SFT training data builder.

This module converts offline PostgreSQL tuning samples into SFT-style JSONL
records for downstream LLM fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

MEMORY_PARAMS = {
    "shared_buffers",
    "work_mem",
    "maintenance_work_mem",
    "effective_cache_size",
    "temp_buffers",
    "wal_buffers",
}


@dataclass
class BuilderConfig:
    min_samples: int = 200
    max_samples: int = 300
    max_plan_chars: int = 1000
    max_plans_per_sample: int = 3
    instruction_lang: str = "zh"
    output_format: str = "human"
    deduplicate: bool = True
    min_tps: float = 0.1
    top_fraction: float = 0.33
    secondary_fraction: float = 0.167


@dataclass
class DatasetStats:
    total_samples: int
    workload_distribution: dict[str, int]
    tps_min: float
    tps_max: float
    tps_mean: float
    tps_p50: float
    tps_p90: float
    avg_input_chars: int
    avg_output_params: int
    validation_errors: int
    deduplication_removed: int


class TrainingDataBuilder:
    """Builds SFT training samples from offline evaluation data."""

    def __init__(
        self,
        offline_sample_path: str,
        output_path: str,
        builder_config: BuilderConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        self.offline_sample_path = offline_sample_path
        self.output_path = output_path
        self.builder_config = builder_config or BuilderConfig()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.samples: list[dict[str, Any]] = []
        self.training_data: list[dict[str, str]] = []
        self.validation_errors = 0
        self.deduplication_removed = 0
        self.last_stats: DatasetStats | None = None
        self._last_valid_source_samples: list[dict[str, Any]] = []
        self.workload_search_dirs = self._build_workload_search_dirs()

    def _build_workload_search_dirs(self) -> list[str]:
        """Build workload search directories for both old and new layouts."""
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

        seen: set[str] = set()
        result: list[str] = []
        for directory in directories:
            normalized = os.path.normpath(directory)
            if normalized not in seen:
                seen.add(normalized)
                result.append(directory)
        return result

    def resolve_workload_path(self, sample: dict[str, Any]) -> str:
        """Prefer real paths from samples and fall back to name-based search."""
        workload = sample.get("workload", "")
        workload_file = sample.get("workload_file", "")

        for candidate in [workload, workload_file]:
            if candidate and os.path.exists(candidate):
                return str(candidate)

        basename = os.path.basename(str(workload_file or workload))
        if not basename:
            return ""

        for directory in self.workload_search_dirs:
            candidate = os.path.join(directory, basename)
            if os.path.exists(candidate):
                return candidate

        return ""

    def load_samples(self) -> bool:
        """Load offline samples from JSONL."""
        self.samples = []
        try:
            with open(self.offline_sample_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    self.samples.append(json.loads(line))
            print(f"✓ 加载了 {len(self.samples)} 个采样数据")
            return True
        except FileNotFoundError:
            print(f"✗ 离线采样文件不存在: {self.offline_sample_path}")
            return False
        except Exception as exc:
            print(f"✗ 加载采样数据失败: {exc}")
            return False

    def _config_hash(self, config: dict[str, Any]) -> str:
        """Create a stable SHA1 hash for deduplication."""
        payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _sample_identity(self, sample: dict[str, Any]) -> str:
        """Create a stable identity for one sample candidate."""
        workload = str(sample.get("workload", ""))
        tps = float(sample.get("tps", 0.0))
        return f"{workload}:{tps:.6f}:{self._config_hash(sample.get('config', {}))}"

    def _deduplicate_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicated configs while keeping the best-TPS sample."""
        deduplicated: list[dict[str, Any]] = []
        seen_config_hashes: set[str] = set()
        removed = 0

        for sample in sorted(samples, key=lambda item: float(item.get("tps", 0.0)), reverse=True):
            config_hash = self._config_hash(sample.get("config", {}))
            if config_hash in seen_config_hashes:
                removed += 1
                continue
            seen_config_hashes.add(config_hash)
            deduplicated.append(sample)

        self.deduplication_removed = removed
        return deduplicated

    def select_high_quality_samples(
        self,
        min_count: int | None = None,
        max_count: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Select high-quality samples grouped by workload.

        Strategy:
        - Filter out samples below `min_tps`
        - Group by workload
        - Take top `top_fraction` per workload
        - Sample `secondary_fraction` from the remainder
        - Optionally deduplicate by config hash
        - Backfill to `min_samples` using best remaining samples
        """
        cfg = self.builder_config
        effective_min = cfg.min_samples if min_count is None else min_count
        effective_max = cfg.max_samples if max_count is None else max_count

        print("\n=== 筛选高质量样本 ===\n")
        self.deduplication_removed = 0

        eligible_samples = [
            sample
            for sample in self.samples
            if float(sample.get("tps", 0.0)) >= cfg.min_tps
        ]

        workload_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in eligible_samples:
            workload_samples[str(sample.get("workload", "unknown"))].append(sample)

        selected_candidates: list[dict[str, Any]] = []
        for workload, samples in sorted(workload_samples.items()):
            sorted_samples = sorted(samples, key=lambda item: float(item.get("tps", 0.0)), reverse=True)
            top_count = min(
                len(sorted_samples),
                max(1, int(np.ceil(len(sorted_samples) * cfg.top_fraction))),
            )
            chosen = list(sorted_samples[:top_count])

            remaining = sorted_samples[top_count:]
            secondary_count = min(
                len(remaining),
                int(np.ceil(len(remaining) * cfg.secondary_fraction)),
            )
            if secondary_count > 0:
                chosen.extend(
                    self.rng.choice(remaining, size=secondary_count, replace=False).tolist()
                )

            selected_candidates.extend(chosen)
            print(
                f"  {workload}: {len(sorted_samples)} -> {len(chosen)} "
                f"(Top {top_count} + {secondary_count} secondary)"
            )

        if cfg.deduplicate:
            selected_samples = self._deduplicate_samples(selected_candidates)
        else:
            selected_samples = sorted(
                selected_candidates,
                key=lambda item: float(item.get("tps", 0.0)),
                reverse=True,
            )

        if len(selected_samples) < effective_min:
            selected_hashes = {self._sample_identity(sample) for sample in selected_samples}
            selected_config_hashes = {
                self._config_hash(sample.get("config", {})) for sample in selected_samples
            }
            for sample in sorted(
                eligible_samples,
                key=lambda item: float(item.get("tps", 0.0)),
                reverse=True,
            ):
                if len(selected_samples) >= effective_min:
                    break
                sample_identity = self._sample_identity(sample)
                if sample_identity in selected_hashes:
                    continue
                if cfg.deduplicate:
                    config_hash = self._config_hash(sample.get("config", {}))
                    if config_hash in selected_config_hashes:
                        continue
                    selected_config_hashes.add(config_hash)
                selected_hashes.add(sample_identity)
                selected_samples.append(sample)

        selected_samples = sorted(
            selected_samples,
            key=lambda item: float(item.get("tps", 0.0)),
            reverse=True,
        )
        if len(selected_samples) > effective_max:
            selected_samples = selected_samples[:effective_max]

        print(f"\n✓ 最终筛选样本数: {len(selected_samples)}/{len(self.samples)}")
        return selected_samples

    def extract_workload_statistics(self, workload_file: str) -> dict[str, Any]:
        """Extract simple SQL statistics from a workload file."""
        stats: dict[str, Any] = {
            "total_sql": 0,
            "read_ratio": 0.0,
            "write_ratio": 0.0,
            "order_by_percent": 0.0,
            "group_by_percent": 0.0,
            "join_count": 0,
            "aggregation_count": 0,
            "table_count": 0,
        }

        try:
            if not workload_file or not os.path.exists(workload_file):
                return stats

            with open(workload_file, "r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read().upper()

            lines = content.splitlines()
            cleaned_lines = [
                line
                for line in lines
                if not line.strip().startswith("--") and not line.strip().startswith("#")
            ]
            content = "\n".join(cleaned_lines)

            sqls = [sql.strip() for sql in content.split(";") if sql.strip()]
            stats["total_sql"] = len(sqls)

            select_count = sum(1 for sql in sqls if sql.startswith("SELECT"))
            write_count = sum(
                1
                for sql in sqls
                if any(sql.startswith(op) for op in ["INSERT", "UPDATE", "DELETE"])
            )
            total_ops = select_count + write_count if write_count > 0 else len(sqls)

            stats["read_ratio"] = round(select_count / total_ops * 100, 1) if total_ops > 0 else 100.0
            stats["write_ratio"] = round(write_count / total_ops * 100, 1) if total_ops > 0 else 0.0

            order_by_count = sum(1 for sql in sqls if "ORDER BY" in sql)
            stats["order_by_percent"] = (
                round(order_by_count / len(sqls) * 100, 1) if sqls else 0.0
            )

            group_by_count = sum(1 for sql in sqls if "GROUP BY" in sql)
            stats["group_by_percent"] = (
                round(group_by_count / len(sqls) * 100, 1) if sqls else 0.0
            )

            stats["join_count"] = sum(sql.count("JOIN") for sql in sqls)

            agg_pattern = r"(COUNT|SUM|AVG|MAX|MIN|GROUP_CONCAT)\s*\("
            stats["aggregation_count"] = sum(len(re.findall(agg_pattern, sql)) for sql in sqls)

            table_pattern = r"FROM\s+(\w+)|JOIN\s+(\w+)"
            all_tables: set[str] = set()
            for sql in sqls:
                for match in re.findall(table_pattern, sql):
                    table = match[0] if match[0] else match[1]
                    if table:
                        all_tables.add(table)
            stats["table_count"] = len(all_tables)
        except Exception as exc:
            print(f"  ! 提取 workload 统计失败: {exc}")

        return stats

    def format_metrics_text(self, inner_metrics: dict[str, Any]) -> str:
        """Format internal metrics for the prompt input."""
        lines: list[str] = []
        priority_metrics = [
            ("cache_hit_ratio", "Cache Hit Ratio", lambda value: f"{float(value) * 100:.1f}%"),
            ("xact_commit", "Committed Transactions", lambda value: f"{float(value) / 1000:.1f}k"),
            ("active_connections", "Active Connections", lambda value: f"{int(value)}"),
            ("tup_returned", "Returned Tuples", lambda value: f"{float(value) / 1000:.1f}k"),
            ("disk_read_count", "Disk Read Count", lambda value: f"{float(value) / 1e6:.1f}M"),
            ("cpu_usage", "CPU Usage", lambda value: f"{float(value):.1f}%"),
        ]

        priority_keys = {metric[0] for metric in priority_metrics}
        for key, label, formatter in priority_metrics:
            if key in inner_metrics:
                try:
                    lines.append(f"- {label}: {formatter(inner_metrics[key])}")
                except Exception:
                    continue

        for key, value in inner_metrics.items():
            if key in priority_keys:
                continue
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_memory_value(self, value: int | float) -> str:
        """Format KB values into a human-readable unit string."""
        kb_value = float(value)
        if kb_value >= 1048576:
            return f"{kb_value / 1048576:.1f}GB"
        if kb_value >= 1024:
            return f"{kb_value / 1024:.1f}MB"
        return f"{int(round(kb_value))}KB"

    def _format_scalar_as_string(self, key: str, value: Any) -> str:
        """Format one config value into a consistent human-readable string."""
        if isinstance(value, bool):
            return "true" if value else "false"
        if key in MEMORY_PARAMS and isinstance(value, (int, float)):
            return self._format_memory_value(value)
        if isinstance(value, float):
            return f"{value:.2f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    def format_config_as_output(self, config: dict[str, Any]) -> str:
        """
        Format a config into the configured output JSON mode.

        raw mode:
            preserves native ints/floats and is easier for LLM post-processing.
        human mode:
            converts every value to a string for consistent prompt targets.
        """
        if self.builder_config.output_format == "raw":
            normalized = {
                key: (value.item() if isinstance(value, np.generic) else value)
                for key, value in sorted(config.items())
            }
            return json.dumps(normalized, ensure_ascii=False, indent=2)

        if self.builder_config.output_format != "human":
            raise ValueError(
                f"Unsupported output_format: {self.builder_config.output_format}"
            )

        normalized = {
            key: self._format_scalar_as_string(key, value)
            for key, value in sorted(config.items())
        }
        return json.dumps(normalized, ensure_ascii=False, indent=2)

    def format_config_as_human_readable(self, config: dict[str, Any]) -> str:
        """Backward-compatible wrapper around the new configurable formatter."""
        original_mode = self.builder_config.output_format
        try:
            self.builder_config.output_format = "human"
            return self.format_config_as_output(config)
        finally:
            self.builder_config.output_format = original_mode

    def validate_output_json(self, output_text: str, sample_index: int) -> bool:
        """Validate that the generated output is parseable JSON."""
        try:
            json.loads(output_text)
            return True
        except json.JSONDecodeError as exc:
            self.validation_errors += 1
            logger.warning("Sample %s output JSON validation failed: %s", sample_index, exc)
            return False

    def _get_instruction_text(self) -> str:
        """Return a bilingual instruction prompt."""
        if self.builder_config.instruction_lang == "en":
            return (
                "You are a senior PostgreSQL tuning expert. Based on the workload\n"
                "characteristics and performance metrics provided, recommend the\n"
                "optimal database parameter configuration.\n"
                "Output must be valid JSON only, no explanations."
            )

        return (
            "你是一个资深的PostgreSQL参数调优专家。根据工作负载特征和性能指标，\n"
            "推荐最优的数据库参数配置。输出必须是严格的JSON格式。"
        )

    def _select_query_plans_text(self, query_plans_text: str) -> str:
        """Limit query plans by count and by total prompt length."""
        if not query_plans_text:
            return "N/A"

        plans_list = [item for item in query_plans_text.split("=== SQL") if item.strip()]
        selected_plans = plans_list[: self.builder_config.max_plans_per_sample]
        if not selected_plans:
            return "N/A"

        combined = "=== SQL" + "=== SQL".join(selected_plans)
        return combined[: self.builder_config.max_plan_chars]

    def build_training_samples(
        self,
        selected_samples: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build SFT-format training samples."""
        print("\n=== 构建 SFT 训练数据 ===\n")
        self.validation_errors = 0
        self._last_valid_source_samples = []

        training_samples: list[dict[str, str]] = []
        instruction = self._get_instruction_text()

        for index, sample in enumerate(selected_samples, 1):
            try:
                workload_path = self.resolve_workload_path(sample)
                workload_stats = self.extract_workload_statistics(workload_path)
                query_plans_text = self._select_query_plans_text(sample.get("query_plans", ""))
                metrics_text = self.format_metrics_text(sample.get("inner_metrics", {}))

                input_text = (
                    "Workload Statistics:\n"
                    f"- Total SQL: {workload_stats['total_sql']}\n"
                    f"- Read-Write Ratio: {workload_stats['read_ratio']}% / {workload_stats['write_ratio']}%\n"
                    f"- ORDER BY Proportion: {workload_stats['order_by_percent']}%\n"
                    f"- GROUP BY Proportion: {workload_stats['group_by_percent']}%\n"
                    f"- JOIN Count: {workload_stats['join_count']}\n"
                    f"- Aggregation Functions: {workload_stats['aggregation_count']}\n"
                    f"- Table Count: {workload_stats['table_count']}\n\n"
                    "Query Plans:\n"
                    f"{query_plans_text}\n\n"
                    "Internal Metrics:\n"
                    f"{metrics_text or 'N/A'}"
                )

                output_json = self.format_config_as_output(sample["config"])
                if not self.validate_output_json(output_json, index):
                    continue

                training_samples.append(
                    {
                        "instruction": instruction,
                        "input": input_text,
                        "output": output_json,
                    }
                )
                self._last_valid_source_samples.append(sample)

                if index % 50 == 0:
                    print(f"  已处理 {index}/{len(selected_samples)} 个样本")
            except Exception as exc:
                print(f"  ! 样本 {index} 构建失败: {exc}")
                continue

        print(f"\n✓ 构建完成，共 {len(training_samples)} 个训练样本")
        return training_samples

    def _build_dataset_stats(
        self,
        training_samples: list[dict[str, str]],
    ) -> DatasetStats:
        """Compute lightweight dataset statistics for quality inspection."""
        if not training_samples or not self._last_valid_source_samples:
            return DatasetStats(
                total_samples=0,
                workload_distribution={},
                tps_min=0.0,
                tps_max=0.0,
                tps_mean=0.0,
                tps_p50=0.0,
                tps_p90=0.0,
                avg_input_chars=0,
                avg_output_params=0,
                validation_errors=self.validation_errors,
                deduplication_removed=self.deduplication_removed,
            )

        workload_distribution: dict[str, int] = defaultdict(int)
        tps_values: list[float] = []
        input_lengths: list[int] = []
        output_param_counts: list[int] = []

        for training_sample, source_sample in zip(training_samples, self._last_valid_source_samples):
            workload_id = str(source_sample.get("workload") or source_sample.get("workload_file") or "unknown")
            workload_distribution[workload_id] += 1
            tps_values.append(float(source_sample.get("tps", 0.0)))
            input_lengths.append(len(training_sample["input"]))
            output_param_counts.append(len(json.loads(training_sample["output"])))

        tps_array = np.array(tps_values, dtype=float)
        return DatasetStats(
            total_samples=len(training_samples),
            workload_distribution=dict(workload_distribution),
            tps_min=float(np.min(tps_array)),
            tps_max=float(np.max(tps_array)),
            tps_mean=float(np.mean(tps_array)),
            tps_p50=float(np.percentile(tps_array, 50)),
            tps_p90=float(np.percentile(tps_array, 90)),
            avg_input_chars=int(round(float(np.mean(input_lengths)))),
            avg_output_params=int(round(float(np.mean(output_param_counts)))),
            validation_errors=self.validation_errors,
            deduplication_removed=self.deduplication_removed,
        )

    def _dataset_stats_path(self) -> str:
        """Return the dataset stats output path."""
        output_dir = os.path.dirname(self.output_path) or "training_data"
        return os.path.join(output_dir, "dataset_stats.json")

    def save_training_data(self, training_samples: list[dict[str, str]]) -> bool:
        """Save training data and stats report."""
        try:
            output_dir = os.path.dirname(self.output_path) or "."
            os.makedirs(output_dir, exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as handle:
                for sample in training_samples:
                    handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

            self.last_stats = self._build_dataset_stats(training_samples)
            stats_path = self._dataset_stats_path()
            os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as handle:
                json.dump(asdict(self.last_stats), handle, ensure_ascii=False, indent=2)

            print(f"\n✓ 训练数据已保存到: {self.output_path}")
            print(
                f"  共 {len(training_samples)} 个样本，"
                f"文件大小: {os.path.getsize(self.output_path) / 1024 / 1024:.2f}MB"
            )
            print(f"✓ 数据集统计已保存到: {stats_path}")
            print(json.dumps(asdict(self.last_stats), ensure_ascii=False, indent=2))
            return True
        except Exception as exc:
            print(f"✗ 保存训练数据失败: {exc}")
            return False

    def build_and_save(
        self,
        min_samples: int | None = None,
        max_samples: int | None = None,
    ) -> bool:
        """End-to-end build flow for SFT training data."""
        print("\n" + "=" * 70)
        print("SFT 训练数据构建")
        print("=" * 70)

        if not self.load_samples():
            return False

        selected_samples = self.select_high_quality_samples(min_samples, max_samples)
        training_samples = self.build_training_samples(selected_samples)
        if not self.save_training_data(training_samples):
            return False

        print("\n✓ 训练数据构建完成\n")
        return True


def build_training_data(offline_sample_path: str, output_path: str) -> bool:
    """
    Convenience wrapper to build training data with backward-compatible args.

    Example:
        build_training_data(
            "offline_sample/offline_sample_tpch.jsonl",
            "training_data/training_sft_data.jsonl",
        )
    """
    builder = TrainingDataBuilder(offline_sample_path, output_path)
    return builder.build_and_save()


if __name__ == "__main__":
    build_training_data(
        offline_sample_path="offline_sample/offline_sample_tpch.jsonl",
        output_path="training_data/training_sft_data.jsonl",
    )
