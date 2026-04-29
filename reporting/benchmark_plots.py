# =============================================================================
# Plotting helpers for benchmark experiments.
#
# Purpose
# -------
# This module visualizes solver-level benchmark summary metrics so that
# cross-solver comparisons are easier to interpret than from tables alone.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reporting.export import save_figure


def plot_benchmark_summary(
    summary_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/benchmark",
    filename_stem: str = "benchmark_summary",
) -> dict[str, Path] | None:
    """
    Create a three-panel solver benchmark summary figure.

    Panels
    ------
    1. Average objective
    2. Average runtime
    3. Average MIP gap

    Parameters
    ----------
    summary_df : pd.DataFrame
        Solver-level benchmark summary table.
    output_dir : str | Path, default="results/figures/benchmark"
        Output directory for the figure.
    filename_stem : str, default="benchmark_summary"
        Base filename for saved figure outputs.

    Returns
    -------
    dict[str, Path] | None
        Saved output paths, or `None` if the input table is empty.
    """
    if summary_df.empty:
        return None

    df = summary_df.copy()

    required_cols = [
        "solver_name",
        "avg_objective",
        "avg_runtime_s",
        "avg_mip_gap",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in summary_df: {col}")

    colors = {
        "Xpress": "steelblue",
        "HiGHS": "tomato",
        "CBC": "seagreen",
    }
    bar_colors = [colors.get(s, "gray") for s in df["solver_name"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Solver Benchmark Summary",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # 1. Average objective
    # -----------------------------------------------------------------
    ax = axes[0]
    values = df["avg_objective"] / 1000.0
    bars = ax.bar(
        df["solver_name"],
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Average Objective")
    ax.set_ylabel("USD ($000s)")
    ax.grid(axis="y", alpha=0.3)

    # -----------------------------------------------------------------
    # 2. Average runtime
    # -----------------------------------------------------------------
    ax = axes[1]
    values = df["avg_runtime_s"]
    bars = ax.bar(
        df["solver_name"],
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Average Runtime")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.3)

    # -----------------------------------------------------------------
    # 3. Average MIP gap
    # -----------------------------------------------------------------
    ax = axes[2]
    values = df["avg_mip_gap"].fillna(0) * 100.0
    bars = ax.bar(
        df["solver_name"],
        values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.setTitle = None
    ax.set_title("Average MIP Gap")
    ax.set_ylabel("Percent")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved_paths = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
        dpi=150,
        save_png=True,
        save_pdf=True,
    )
    plt.close(fig)

    return saved_paths