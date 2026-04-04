from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/data/codebooks")
    parser.add_argument("--out", type=str, default="coverage_curve.png")
    args = parser.parse_args()

    stats_dir = Path(args.dir)
    stats_files = sorted(stats_dir.glob("*.stats.json"))
    if not stats_files:
        raise SystemExit(f"No .stats.json files found in {stats_dir}")

    all_k_values: set[int] = set()
    layer_curves: dict[str, dict[int, float]] = {}

    for sf in stats_files:
        with sf.open() as f:
            stats = json.load(f)
        curve = stats.get("coverage_curve", {})
        name = sf.stem.replace(".stats", "")
        parsed = {int(k): v for k, v in curve.items()}
        layer_curves[name] = parsed
        all_k_values.update(parsed.keys())

    k_sorted = sorted(all_k_values)
    if not k_sorted:
        raise SystemExit("No coverage_curve data found")

    coverage_matrix = np.zeros((len(layer_curves), len(k_sorted)))
    layer_names = sorted(layer_curves.keys())
    for i, name in enumerate(layer_names):
        for j, k in enumerate(k_sorted):
            coverage_matrix[i, j] = layer_curves[name].get(k, float("nan"))

    avg_coverage = np.nanmean(coverage_matrix, axis=0)
    min_coverage = np.nanmin(coverage_matrix, axis=0)
    max_coverage = np.nanmax(coverage_matrix, axis=0)
    p25 = np.nanpercentile(coverage_matrix, 25, axis=0)
    p75 = np.nanpercentile(coverage_matrix, 75, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(
        k_sorted,
        min_coverage * 100,
        max_coverage * 100,
        alpha=0.1,
        color="tab:blue",
        label="min–max",
    )
    ax.fill_between(
        k_sorted, p25 * 100, p75 * 100, alpha=0.25, color="tab:blue", label="p25–p75"
    )
    ax.plot(
        k_sorted, avg_coverage * 100, "o-", color="tab:blue", linewidth=2, label="mean"
    )

    ax.set_xlabel("Top-k codebooks")
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Top-k codebook coverage ({len(layer_names)} layers)")
    ax.set_ylim(0, 105)
    ax.set_xticks(k_sorted)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

    print(f"\n{'k':>6}  {'mean':>8}  {'min':>8}  {'max':>8}  {'p25':>8}  {'p75':>8}")
    for j, k in enumerate(k_sorted):
        print(
            f"{k:>6}  {avg_coverage[j] * 100:>7.1f}%  {min_coverage[j] * 100:>7.1f}%  {max_coverage[j] * 100:>7.1f}%  {p25[j] * 100:>7.1f}%  {p75[j] * 100:>7.1f}%"
        )


if __name__ == "__main__":
    main()
