# =============================================================================
# Reporting helpers for benchmark experiments.
#
# Purpose
# -------
# This module handles benchmark-oriented table export and compact console
# summaries. It is a lightweight reporting layer on top of the benchmark
# experiment outputs.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reporting.export import save_dataframe


def save_benchmark_outputs(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/tables/benchmark",
    raw_filename: str = "benchmark_raw",
    summary_filename: str = "benchmark_summary",
) -> dict[str, dict]:
    """
    Save raw and summary benchmark tables.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Benchmark run-level table.
    summary_df : pd.DataFrame
        Benchmark solver-level summary table.
    output_dir : str | Path, default="results/tables/benchmark"
        Output directory.
    raw_filename : str, default="benchmark_raw"
        Base filename for the raw table.
    summary_filename : str, default="benchmark_summary"
        Base filename for the summary table.

    Returns
    -------
    dict[str, dict]
        Nested dictionary of saved file paths with keys:
        - `raw`
        - `summary`
    """
    raw_paths = save_dataframe(
        raw_df,
        output_dir=output_dir,
        filename_stem=raw_filename,
        index=False,
        save_csv=True,
        save_excel=True,
    )

    summary_paths = save_dataframe(
        summary_df,
        output_dir=output_dir,
        filename_stem=summary_filename,
        index=False,
        save_csv=True,
        save_excel=True,
    )

    return {
        "raw": raw_paths,
        "summary": summary_paths,
    }


def print_benchmark_summary(summary_df: pd.DataFrame) -> None:
    """
    Print a compact benchmark summary to the console.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Solver-level benchmark summary table.
    """
    if summary_df.empty:
        print("\nNo benchmark summary available.")
        return

    print("\n" + "=" * 88)
    print("BENCHMARK SUMMARY")
    print("=" * 88)
    print(summary_df.to_string(index=False))
    print("=" * 88)