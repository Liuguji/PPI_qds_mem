#!/usr/bin/env python3
"""
Aggregate CPU/GPU flow JSON summaries by n and generate seaborn charts.

Default behavior:
- Input dir: <repo_root>/liubo_test_CPU_GPU_results
- Output dir root: <repo_root>/liubo_diag

Generated charts:
1) avg_cpu_full_flow_mem_by_n               (line plot)
2) avg_cpu_gpu_elapsed_by_n_clustered_bar   (clustered bar)
3) avg_speedup_gpu_vs_cpu_by_n_with_ref_y1  (line plot + y=1 reference)

Each chart is saved in its own subdirectory under liubo_diag.
Filename format: <file_stem>_<timestamp>.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


REQUIRED_KEYS = [
    "n",
    "cpu_elapsed_s",
    "gpu_elapsed_s",
    "speedup_gpu_vs_cpu",
    "cpu_full_flow_mem_gb",
]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return data


def _load_rows(input_dir: Path) -> list[dict[str, Any]]:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    rows: list[dict[str, Any]] = []
    for fp in json_files:
        row = _load_json(fp)
        missing = [k for k in REQUIRED_KEYS if k not in row]
        if missing:
            raise KeyError(f"Missing keys {missing} in file: {fp}")
        rows.append(row)
    return rows


def _build_aggregated_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)

    # Ensure numeric types for aggregation.
    numeric_cols = REQUIRED_KEYS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid rows that cannot be used for averaging.
    df = df.dropna(subset=numeric_cols)
    if df.empty:
        raise ValueError("No valid rows left after numeric conversion.")

    agg = (
        df.groupby("n", as_index=False)[
            ["cpu_elapsed_s", "gpu_elapsed_s", "speedup_gpu_vs_cpu", "cpu_full_flow_mem_gb"]
        ]
        .mean()
        .sort_values("n")
    )

    return agg


def _save_line_mem(df_agg: pd.DataFrame, out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"avg_cpu_full_flow_mem_by_n_{ts}.png"

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df_agg, x="n", y="cpu_full_flow_mem_gb", marker="o", linewidth=2)
    plt.title("Average CPU Full-Flow Memory by n")
    plt.xlabel("n")
    plt.ylabel("avg cpu_full_flow_mem_gb")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _save_cluster_elapsed(df_agg: pd.DataFrame, out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"avg_cpu_gpu_elapsed_by_n_clustered_bar_{ts}.png"

    df_long = df_agg.melt(
        id_vars=["n"],
        value_vars=["cpu_elapsed_s", "gpu_elapsed_s"],
        var_name="series",
        value_name="elapsed_s",
    )

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=df_long, x="n", y="elapsed_s", hue="series")
    plt.title("Average CPU/GPU Elapsed Time by n")
    plt.xlabel("n")
    plt.ylabel("avg elapsed_s")
    plt.legend(title="metric")

    # Add value labels on bars with two decimal places.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)

    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _save_line_mem_with_ref(df_agg: pd.DataFrame, out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"avg_speedup_gpu_vs_cpu_by_n_with_ref_y1_{ts}.png"

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=df_agg, x="n", y="speedup_gpu_vs_cpu", marker="o", linewidth=2)
    plt.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="y = 1")
    plt.title("Average Speedup (GPU vs CPU) by n (y=1 reference)")
    plt.xlabel("n")
    plt.ylabel("avg speedup_gpu_vs_cpu")
    # Do not force uniform y-ticks; let matplotlib choose based on data.
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main() -> None:
    sns.set_theme(style="whitegrid")

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]

    default_input_dir = repo_root / "liubo_test_CPU_GPU_results"
    default_diag_root = repo_root / "liubo_diag"

    parser = argparse.ArgumentParser(description="Aggregate JSON summaries by n and plot seaborn charts")
    parser.add_argument("--input-dir", type=Path, default=default_input_dir, help="Directory containing JSON summaries")
    parser.add_argument("--diag-root", type=Path, default=default_diag_root, help="Root directory for output chart folders")
    parser.add_argument("--save-agg-csv", action="store_true", help="Also save aggregated n-level table to CSV")
    args = parser.parse_args()

    input_dir = args.input_dir
    diag_root = args.diag_root

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rows = _load_rows(input_dir)
    df_agg = _build_aggregated_df(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")

    dir1 = diag_root / "avg_cpu_full_flow_mem_by_n"
    dir2 = diag_root / "avg_cpu_gpu_elapsed_by_n_clustered_bar"
    dir3 = diag_root / "avg_speedup_gpu_vs_cpu_by_n_with_ref_y1"

    p1 = _save_line_mem(df_agg, dir1, ts)
    p2 = _save_cluster_elapsed(df_agg, dir2, ts)
    p3 = _save_line_mem_with_ref(df_agg, dir3, ts)

    if args.save_agg_csv:
        agg_csv = diag_root / f"n_level_avg_metrics_{ts}.csv"
        agg_csv.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(agg_csv, index=False, encoding="utf-8")
        print(f"Aggregated CSV: {agg_csv}")

    print(f"Loaded JSON files: {len(rows)}")
    print("Averaged metrics by n:")
    print(df_agg.to_string(index=False))
    print(f"Chart 1: {p1}")
    print(f"Chart 2: {p2}")
    print(f"Chart 3: {p3}")


if __name__ == "__main__":
    main()
