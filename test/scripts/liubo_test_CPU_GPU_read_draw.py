#!/usr/bin/env python3
"""
Aggregate CPU/GPU flow JSON summaries by n and generate seaborn charts.

Default behavior:
- Input dir: <repo_root>/liubo_test_CPU_GPU_data
- Output dir root: <repo_root>/liubo_diag

Generated charts:
1) avg_flow_memory_by_n_three_lines         (line plot, 3 series)
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

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from scipy.interpolate import make_interp_spline
except Exception:
    make_interp_spline = None


REQUIRED_KEYS = [
    "n",
    "cpu_elapsed_s",
    "gpu_elapsed_s",
    "speedup_gpu_vs_cpu",
    "cpu_full_flow_mem_gb",
    "cpu_ckpt_flow_mem_gb",
    "gpu_ckpt_flow_mem_gb",
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
            [
                "cpu_elapsed_s",
                "gpu_elapsed_s",
                "speedup_gpu_vs_cpu",
                "cpu_full_flow_mem_gb",
                "cpu_ckpt_flow_mem_gb",
                "gpu_ckpt_flow_mem_gb",
            ]
        ]
        .mean()
        .sort_values("n")
    )

    return agg


def _plot_smooth_series(ax, x, y, *, label: str, marker: str = "o", linewidth: float = 2.0) -> None:
    """Plot a smooth curve through points when possible; otherwise fall back to polyline."""
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)

    valid = ~(pd.isna(x_arr) | pd.isna(y_arr))
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if len(x_arr) == 0:
        return

    order = x_arr.argsort()
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    # Cubic spline needs scipy and at least 4 points with distinct x.
    can_smooth = (
        make_interp_spline is not None
        and len(x_arr) >= 4
        and len(set(x_arr.tolist())) == len(x_arr)
    )
    if can_smooth:
        x_dense = np.linspace(float(x_arr.min()), float(x_arr.max()), 300)
        spline = make_interp_spline(x_arr, y_arr, k=3)
        y_dense = spline(x_dense)
        ax.plot(x_dense, y_dense, label=label, linewidth=linewidth)
        ax.plot(x_arr, y_arr, linestyle="", marker=marker, markersize=5)
    else:
        ax.plot(x_arr, y_arr, label=label, marker=marker, linewidth=linewidth)


def _save_line_avg_mem_by_n(df_agg: pd.DataFrame, out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"avg_flow_memory_by_n_three_lines_{ts}.png"

    df_mem = df_agg.melt(
        id_vars=["n"],
        value_vars=["cpu_full_flow_mem_gb", "cpu_ckpt_flow_mem_gb", "gpu_ckpt_flow_mem_gb"],
        var_name="series",
        value_name="mem_gb",
    )

    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    for series_name in ["cpu_full_flow_mem_gb", "cpu_ckpt_flow_mem_gb", "gpu_ckpt_flow_mem_gb"]:
        part = df_mem[df_mem["series"] == series_name]
        _plot_smooth_series(
            ax,
            part["n"],
            part["mem_gb"],
            label=series_name,
            marker="o",
            linewidth=2,
        )
    plt.title("Average Flow Memory by n (CPU full / CPU ckpt / GPU ckpt)")
    plt.xlabel("n")
    plt.ylabel("avg memory (GB)")
    # Use logarithmic y-axis so large scale differences are readable.
    plt.yscale("log")
    plt.legend(title="metric")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def _save_cluster_avg_elapsed_by_n(df_agg: pd.DataFrame, out_dir: Path, ts: str) -> Path:
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


def _save_line_avg_speedup_by_n_with_ref(df_agg: pd.DataFrame, out_dir: Path, ts: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"avg_speedup_gpu_vs_cpu_by_n_with_ref_y1_{ts}.png"

    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    _plot_smooth_series(
        ax,
        df_agg["n"],
        df_agg["speedup_gpu_vs_cpu"],
        label="speedup_gpu_vs_cpu",
        marker="o",
        linewidth=2,
    )
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

    default_input_dir = repo_root / "liubo_test_CPU_GPU_data"
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

    dir1 = diag_root / "avg_flow_memory_by_n_three_lines"
    dir2 = diag_root / "avg_cpu_gpu_elapsed_by_n_clustered_bar"
    dir3 = diag_root / "avg_speedup_gpu_vs_cpu_by_n_with_ref_y1"

    chart_mem_path = _save_line_avg_mem_by_n(df_agg, dir1, ts)
    chart_elapsed_path = _save_cluster_avg_elapsed_by_n(df_agg, dir2, ts)
    chart_speedup_path = _save_line_avg_speedup_by_n_with_ref(df_agg, dir3, ts)

    if args.save_agg_csv:
        agg_csv = diag_root / f"n_level_avg_metrics_{ts}.csv"
        agg_csv.parent.mkdir(parents=True, exist_ok=True)
        df_agg.to_csv(agg_csv, index=False, encoding="utf-8")
        print(f"Aggregated CSV: {agg_csv}")

    print(f"Loaded JSON files: {len(rows)}")
    print("Averaged metrics by n:")
    print(df_agg.to_string(index=False))
    print(f"Chart 1: {chart_mem_path}")
    print(f"Chart 2: {chart_elapsed_path}")
    print(f"Chart 3: {chart_speedup_path}")


if __name__ == "__main__":
    main()
