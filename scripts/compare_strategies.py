"""
Compare random vs SMAC sampling strategy quality from existing offline samples.

Usage (two separate files):
    python scripts/compare_strategies.py \
        --random-file offline_sample/offline_sample_localhost.jsonl \
        --smac-file   offline_sample/offline_sample_smac_localhost.jsonl

Usage (auto-detect, single file, breakdown by workload):
    python scripts/compare_strategies.py --auto
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load_samples(path: str) -> list[dict]:
    samples: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    except FileNotFoundError:
        print(f"  [WARN] File not found: {path}")
    return samples


def compute_stats(samples: list[dict], label: str) -> dict:
    valid = [float(s.get("tps", 0.0)) for s in samples if float(s.get("tps", 0.0)) > 0.001]
    if not valid:
        print(f"  [{label}] No valid samples (TPS > 0.001)")
        return {}
    arr = np.array(valid)
    return {
        "label": label,
        "n_total": len(samples),
        "n_valid": len(valid),
        "tps_min":    float(arr.min()),
        "tps_max":    float(arr.max()),
        "tps_mean":   float(arr.mean()),
        "tps_median": float(np.median(arr)),
        "tps_std":    float(arr.std()),
        "tps_p75":    float(np.percentile(arr, 75)),
        "tps_p90":    float(np.percentile(arr, 90)),
        "tps_p95":    float(np.percentile(arr, 95)),
        "improvement_ratio": float((arr.max() - arr.min()) / arr.min()) if arr.min() > 0 else 0.0,
    }


def print_comparison(a: dict, b: dict) -> None:
    if not a or not b:
        return
    metrics = ["n_valid", "tps_max", "tps_mean", "tps_median", "tps_p90", "tps_p95", "improvement_ratio"]
    col_w = [24, 14, 14, 8]
    hdr = f"{'Metric':<24}{a['label']:<14}{b['label']:<14}{'Winner':<8}"
    print("\n" + "=" * 60)
    print(f"  {a['label']}  vs  {b['label']}")
    print("=" * 60)
    print(hdr)
    print("-" * 60)
    for key in metrics:
        av, bv = a.get(key, 0), b.get(key, 0)
        winner = a["label"] if av > bv else (b["label"] if bv > av else "tie")
        print(f"  {key:<22}{av:<14.4f}{bv:<14.4f}{winner}")
    print("=" * 60)
    better = b["label"] if b["tps_p90"] > a["tps_p90"] else a["label"]
    print(f"  P90 TPS winner: {better}  (margin: {abs(a['tps_p90'] - b['tps_p90']):.4f})\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sampling strategy TPS quality")
    parser.add_argument("--random-file", type=str)
    parser.add_argument("--smac-file",   type=str)
    parser.add_argument("--auto",        action="store_true",
                        help="Auto-detect offline_sample/*.jsonl and show per-workload breakdown")
    args = parser.parse_args()

    if args.auto:
        files = sorted(glob.glob("offline_sample/*.jsonl"))
        if not files:
            print("No offline_sample/*.jsonl files found. Run Phase 1 first.")
            return
        for path in files:
            samples = load_samples(path)
            from collections import defaultdict
            by_wl: dict[str, list] = defaultdict(list)
            for s in samples:
                by_wl[str(s.get("workload_file", "?"))].append(s)
            print(f"\nFile: {path}  ({len(samples)} total samples)")
            for wl, wl_samples in sorted(by_wl.items()):
                st = compute_stats(wl_samples, wl)
                if st:
                    print(f"  {wl:<30} n={st['n_valid']:>3}  "
                          f"mean={st['tps_mean']:.4f}  max={st['tps_max']:.4f}  "
                          f"p90={st['tps_p90']:.4f}")
        return

    if not (args.random_file and args.smac_file):
        parser.print_help()
        return

    stats_random = compute_stats(load_samples(args.random_file), "random")
    stats_smac   = compute_stats(load_samples(args.smac_file),   "smac")
    print_comparison(stats_random, stats_smac)

    os.makedirs("results", exist_ok=True)
    out = {"random": stats_random, "smac": stats_smac}
    out_path = "results/strategy_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  Full stats saved to: {out_path}")


if __name__ == "__main__":
    main()
