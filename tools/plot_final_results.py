import argparse
import json
import os
import statistics
from typing import Any, List, Dict

import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_plot_series(paths: List[str]) -> List[Dict[str, Any]]:
    series = []
    for path in paths:
        payload = load_json(path)
        series.extend(payload.get("series", []))
    return series


def infer_benchmark(workload_id: str | None) -> str:
    text = (workload_id or "").lower()
    if text.startswith("tpch"):
        return "TPCH"
    if text.startswith("job"):
        return "JOB"
    if text.startswith("ssb"):
        return "SSB"
    return "OTHER"


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def plot_training_loss(trainer_state_path: str, output_dir: str) -> None:
    if not trainer_state_path or not os.path.exists(trainer_state_path):
        print(f"[skip] trainer_state.json not found: {trainer_state_path}")
        return

    payload = load_json(trainer_state_path)
    log_history = payload.get("log_history", [])

    train_steps = []
    train_loss = []
    eval_steps = []
    eval_loss = []

    for item in log_history:
        if "loss" in item and "step" in item:
            train_steps.append(item["step"])
            train_loss.append(item["loss"])
        if "eval_loss" in item and "step" in item:
            eval_steps.append(item["step"])
            eval_loss.append(item["eval_loss"])

    if not train_steps and not eval_steps:
        print("[skip] no train/eval loss found in trainer_state.json")
        return

    plt.figure(figsize=(8, 5))
    if train_steps:
        plt.plot(train_steps, train_loss, label="train_loss", linewidth=2)
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="eval_loss", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] saved {out_path}")


def plot_success_rates(series: List[Dict[str, Any]], output_dir: str) -> None:
    total = len(series)
    if total == 0:
        print("[skip] no evaluation series for success rate plot")
        return

    json_success = sum(1 for row in series if bool(row.get("parse_ok")))
    param_valid = sum(1 for row in series if bool(row.get("validation_valid")))
    execution_success = sum(1 for row in series if bool(row.get("execution_success")))
    better_than_baseline = sum(
        1 for row in series
        if row.get("tps_delta_ratio") is not None and (row.get("tps_delta_ratio") > 0)
    )

    labels = [
        "JSON Success",
        "Parameter Valid",
        "Execution Success",
        "Better than Baseline",
    ]
    values = [
        json_success / total * 100,
        param_valid / total * 100,
        execution_success / total * 100,
        better_than_baseline / total * 100,
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    plt.ylabel("Rate (%)")
    plt.ylim(0, 100)
    plt.title("Success Rate Distribution")
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "success_rates.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] saved {out_path}")


def plot_improvement_by_benchmark(series: List[Dict[str, Any]], output_dir: str) -> None:
    grouped: Dict[str, List[float]] = {}
    for row in series:
        ratio = row.get("tps_delta_ratio")
        if ratio is None:
            continue
        benchmark = infer_benchmark(row.get("workload_id"))
        grouped.setdefault(benchmark, []).append(ratio * 100.0)

    if not grouped:
        print("[skip] no tps_delta_ratio found for benchmark improvement plot")
        return

    benchmarks = []
    mean_values = []
    median_values = []

    for benchmark in ["TPCH", "JOB", "SSB", "OTHER"]:
        values = grouped.get(benchmark, [])
        if not values:
            continue
        benchmarks.append(benchmark)
        mean_values.append(safe_mean(values))
        median_values.append(safe_median(values))

    x = range(len(benchmarks))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], mean_values, width=width, label="Mean Improvement (%)", color="#4C78A8")
    plt.bar([i + width / 2 for i in x], median_values, width=width, label="Median Improvement (%)", color="#F58518")

    plt.xticks(list(x), benchmarks)
    plt.ylabel("Performance Improvement (%)")
    plt.title("Performance Improvement by Benchmark")
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, "improvement_by_benchmark.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] saved {out_path}")


def plot_workload_comparison(series: List[Dict[str, Any]], output_dir: str, top_n: int = 20) -> None:
    rows = []
    for row in series:
        baseline_tps = row.get("baseline_tps")
        predicted_tps = row.get("predicted_tps")
        if baseline_tps is None or predicted_tps is None:
            continue
        rows.append(
            {
                "workload_id": row.get("workload_id", "unknown"),
                "baseline_tps": baseline_tps,
                "predicted_tps": predicted_tps,
                "gain": predicted_tps - baseline_tps,
            }
        )

    if not rows:
        print("[skip] no baseline/predicted TPS pairs found for workload comparison")
        return

    rows.sort(key=lambda x: x["gain"], reverse=True)
    rows = rows[:top_n]

    workload_ids = [row["workload_id"] for row in rows]
    baseline = [row["baseline_tps"] for row in rows]
    predicted = [row["predicted_tps"] for row in rows]

    x = range(len(rows))
    width = 0.4

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], baseline, width=width, label="Baseline TPS", color="#9ECAE1")
    plt.bar([i + width / 2 for i in x], predicted, width=width, label="Predicted TPS", color="#3182BD")

    plt.xticks(list(x), workload_ids, rotation=45, ha="right")
    plt.ylabel("TPS")
    plt.title(f"Baseline vs Predicted TPS (Top {top_n} Workloads)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "workload_tps_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[ok] saved {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final experiment figures from plot_data.json")
    parser.add_argument(
        "--trainer-state",
        type=str,
        required=True,
        help="Path to trainer_state.json under the final checkpoint directory",
    )
    parser.add_argument(
        "--plot-data",
        type=str,
        nargs="+",
        required=True,
        help="One or more plot_data.json files produced by surrogate/evaluation.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save paper figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    series = load_plot_series(args.plot_data)

    plot_training_loss(args.trainer_state, args.output_dir)
    plot_success_rates(series, args.output_dir)
    plot_improvement_by_benchmark(series, args.output_dir)
    plot_workload_comparison(series, args.output_dir, top_n=20)


if __name__ == "__main__":
    main()
