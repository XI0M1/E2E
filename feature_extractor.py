"""
Workload feature extraction from offline samples.

This implementation is intentionally defensive for large-scale batch runs:
- all numeric aggregations are safe on empty arrays
- non-numeric metric values are ignored instead of poisoning the vector
- extraction produces a small metadata report for downstream auditing
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class WorkloadFeatureExtractor:
    """Extract workload-level feature vectors from offline sampling data."""

    VECTOR_DIM = 30

    def __init__(self, offline_sample_path: str, database: str):
        self.offline_sample_path = offline_sample_path
        self.database = database
        self.features: Dict[str, List[float]] = {}
        self.workload_data = defaultdict(list)
        self.feature_report: Dict[str, Dict[str, object]] = {}

    def _log(self, message: str):
        print(message)

    def _numeric_array(self, values: List[object]) -> np.ndarray:
        numeric_values = []
        for value in values:
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                numeric_values.append(float(value))
        if not numeric_values:
            return np.array([], dtype=float)
        return np.asarray(numeric_values, dtype=float)

    def _safe_mean(self, values: List[object], default: float = 0.0) -> float:
        arr = self._numeric_array(values)
        return float(arr.mean()) if arr.size > 0 else float(default)

    def _safe_std(self, values: List[object], default: float = 0.0) -> float:
        arr = self._numeric_array(values)
        return float(arr.std()) if arr.size > 0 else float(default)

    def _safe_min(self, values: List[object], default: float = 0.0) -> float:
        arr = self._numeric_array(values)
        return float(arr.min()) if arr.size > 0 else float(default)

    def _safe_max(self, values: List[object], default: float = 0.0) -> float:
        arr = self._numeric_array(values)
        return float(arr.max()) if arr.size > 0 else float(default)

    def _ratio(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        if denominator == 0:
            return float(default)
        return float(numerator / denominator)

    def _clip_feature(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(max(0.0, min(1.0, value)))

    def load_samples(self) -> bool:
        try:
            with open(self.offline_sample_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    workload = sample.get("workload", "unknown")
                    self.workload_data[workload].append(sample)

            self._log(f"Loaded samples for {len(self.workload_data)} workloads")
            for workload_id, samples in self.workload_data.items():
                self._log(f"  {workload_id}: {len(samples)} samples")
            return True
        except FileNotFoundError:
            self._log(f"Offline sample file not found: {self.offline_sample_path}")
            return False
        except Exception as exc:
            self._log(f"Failed to load offline samples: {exc}")
            return False

    def extract_features(self) -> Dict[str, List[float]]:
        self._log("\n=== Extracting workload feature vectors ===\n")

        for workload_id, samples in self.workload_data.items():
            self._log(f"Processing {workload_id} ({len(samples)} samples)")
            feature = [0.0] * self.VECTOR_DIM
            report = {
                "workload_id": workload_id,
                "sample_count": len(samples),
                "status": "success",
                "empty_metrics": [],
            }

            shared_buffers = [s.get("config", {}).get("shared_buffers", 131072) for s in samples]
            work_mem = [s.get("config", {}).get("work_mem", 4096) for s in samples]
            effective_cache_size = [s.get("config", {}).get("effective_cache_size", 524288) for s in samples]
            maintenance_work_mem = [s.get("config", {}).get("maintenance_work_mem", 65536) for s in samples]
            checkpoint_completion_target = [
                s.get("config", {}).get("checkpoint_completion_target", 0.5) for s in samples
            ]

            feature[0] = self._clip_feature(self._safe_mean(shared_buffers, 131072.0) / 2000000.0)
            feature[1] = self._clip_feature(self._safe_mean(work_mem, 4096.0) / 1000000.0)
            feature[2] = self._clip_feature(self._safe_mean(effective_cache_size, 524288.0) / 2000000.0)
            feature[3] = self._clip_feature(self._safe_mean(maintenance_work_mem, 65536.0) / 1000000.0)
            feature[4] = self._clip_feature(self._safe_mean(checkpoint_completion_target, 0.5))

            param_variance = self._compute_parameter_sensitivity(samples)
            for index, (_, variance) in enumerate(sorted(param_variance.items())[:11]):
                feature[5 + index] = self._clip_feature(variance)

            tps_values = [s.get("tps", 0.0) for s in samples]
            tps_mean = self._safe_mean(tps_values, 0.0)
            tps_std = self._safe_std(tps_values, 0.0)
            feature[16] = self._clip_feature(self._ratio(tps_std, tps_mean, 0.0))

            tps_min = self._safe_min(tps_values, 0.0)
            tps_max = self._safe_max(tps_values, 0.0)
            feature[17] = self._clip_feature(self._ratio(tps_max - tps_min, tps_min, 0.0))

            cache_ratios = [s.get("inner_metrics", {}).get("cache_hit_ratio") for s in samples]
            if self._numeric_array(cache_ratios).size == 0:
                report["empty_metrics"].append("cache_hit_ratio")
            cache_mean = self._safe_mean(cache_ratios, 0.5)
            cache_std = self._safe_std(cache_ratios, 0.0)
            feature[18] = self._clip_feature(cache_mean)
            feature[19] = self._clip_feature(self._ratio(cache_std, cache_mean, 0.0))

            cpu_usages = [s.get("inner_metrics", {}).get("cpu_usage") for s in samples]
            if self._numeric_array(cpu_usages).size == 0:
                report["empty_metrics"].append("cpu_usage")
            feature[20] = self._clip_feature(self._safe_mean(cpu_usages, 50.0) / 100.0)
            feature[26] = self._clip_feature(self._safe_std(cpu_usages, 0.0) / 100.0)

            active_connections = [s.get("inner_metrics", {}).get("active_connections") for s in samples]
            if self._numeric_array(active_connections).size == 0:
                report["empty_metrics"].append("active_connections")
            feature[21] = self._clip_feature(self._safe_mean(active_connections, 5.0) / 100.0)

            database_sizes = [s.get("inner_metrics", {}).get("database_size") for s in samples]
            if self._numeric_array(database_sizes).size == 0:
                report["empty_metrics"].append("database_size")
            feature[22] = self._clip_feature(self._safe_mean(database_sizes, 5 * 10**9) / (100 * 10**9))

            feature[23] = self._clip_feature(1.0 / (1.0 + feature[16]))
            feature[24] = 0.5
            feature[25] = self._clip_feature(self._compute_sensitivity_score(feature[16], feature[19]))

            latencies = [s.get("inner_metrics", {}).get("latency_ms") for s in samples]
            if self._numeric_array(latencies).size == 0:
                report["empty_metrics"].append("latency_ms")
            feature[27] = self._clip_feature(self._safe_mean(latencies, 5.0) / 100.0)
            feature[28] = self._clip_feature(self._safe_std(latencies, 0.0) / 100.0)

            feature[29] = self._clip_feature(float(len(samples)) / 50.0)

            self.features[workload_id] = [self._clip_feature(value) for value in feature]
            self.feature_report[workload_id] = report
            self._log(
                f"  OK: dim={len(feature)}, range=[{min(feature):.3f}, {max(feature):.3f}], "
                f"empty_metrics={report['empty_metrics']}"
            )

        return self.features

    def _compute_parameter_sensitivity(self, samples: List[Dict]) -> Dict[str, float]:
        if len(samples) < 2:
            return {}

        all_params = set()
        for sample in samples:
            all_params.update(sample.get("config", {}).keys())

        tps_values = self._numeric_array([sample.get("tps", 0.0) for sample in samples])
        if tps_values.size < 2 or float(tps_values.std()) == 0.0:
            return {}

        param_tps_corr: Dict[str, float] = {}
        for param in list(all_params)[:20]:
            param_values = self._numeric_array(
                [sample.get("config", {}).get(param) for sample in samples]
            )
            if param_values.size != tps_values.size:
                continue
            if param_values.size < 2 or float(param_values.std()) == 0.0:
                continue

            corr_matrix = np.corrcoef(param_values, tps_values)
            corr = float(abs(corr_matrix[0, 1]))
            if np.isfinite(corr):
                param_tps_corr[param] = corr

        return param_tps_corr

    def _compute_sensitivity_score(self, tps_variance: float, cache_variance: float) -> float:
        return min((tps_variance * 0.6 + cache_variance * 0.4), 1.0)

    def save_features(self, output_dir: str = "SuperWG/feature") -> bool:
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.database}.json")
            report_path = os.path.join(output_dir, f"{self.database}_feature_report.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.features, f, indent=2, ensure_ascii=False)

            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.feature_report, f, indent=2, ensure_ascii=False)

            self._log(f"\nSaved feature vectors to: {output_path}")
            self._log(f"Saved feature extraction report to: {report_path}")
            return True
        except Exception as exc:
            self._log(f"Failed to save feature vectors: {exc}")
            return False

    def extract_and_save(self) -> bool:
        self._log("\n" + "=" * 70)
        self._log("Workload Feature Vector Extraction")
        self._log("=" * 70)

        if not self.load_samples():
            return False

        self.extract_features()

        if not self.save_features():
            return False

        self._log("\nFeature extraction completed.\n")
        return True


def extract_workload_features(offline_sample_path: str, database: str) -> bool:
    extractor = WorkloadFeatureExtractor(offline_sample_path, database)
    return extractor.extract_and_save()


if __name__ == "__main__":
    extract_workload_features(
        offline_sample_path="offline_sample/offline_sample_tpch.jsonl",
        database="tpch",
    )

