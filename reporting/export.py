# =============================================================================
# Shared export helpers for tables and figure artifacts in the VSRP codebase.
#
# Purpose
# -------
# This module centralizes the low-level save logic used across reporting
# modules so that table and figure export behavior is consistent.
#
# Main responsibilities
# ---------------------
# - create output directories
# - save pandas DataFrames to CSV and Excel
# - save matplotlib figures to PNG and PDF
#
# Architectural role
# ------------------
# This file is the utility foundation of the reporting layer. It prevents
# repeated save logic from being duplicated across plotting and reporting
# modules.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """
    Ensure that a directory exists and return the resulting path.

    Parameters
    ----------
    path : Path
        Directory path to create if necessary.

    Returns
    -------
    Path
        The same directory path, guaranteed to exist.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    filename_stem: str,
    index: bool = False,
    save_csv: bool = True,
    save_excel: bool = True,
) -> dict[str, Path]:
    """
    Save a DataFrame to CSV and/or Excel.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save.
    output_dir : str | Path
        Target output directory.
    filename_stem : str
        Base filename without extension.
    index : bool, default=False
        Whether to write the DataFrame index.
    save_csv : bool, default=True
        Whether to save a CSV version.
    save_excel : bool, default=True
        Whether to save an Excel version.

    Returns
    -------
    dict[str, Path]
        Mapping from file type to saved file path.
    """
    output_dir = ensure_directory(Path(output_dir))
    saved_paths: dict[str, Path] = {}

    if save_csv:
        csv_path = output_dir / f"{filename_stem}.csv"
        df.to_csv(csv_path, index=index)
        saved_paths["csv"] = csv_path

    if save_excel:
        xlsx_path = output_dir / f"{filename_stem}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=index, sheet_name=filename_stem[:31])
        saved_paths["xlsx"] = xlsx_path

    return saved_paths


def save_figure(
    fig,
    *,
    output_dir: "str | Path",
    filename_stem: str,
    dpi: int = 150,
    save_png: bool = True,
    save_pdf: bool = True,
) -> "dict[str, Path]":
    """
    Save a matplotlib figure to PNG and/or PDF.

    Parameters
    ----------
    fig
        Matplotlib figure object.
    output_dir : str | Path
        Target output directory.
    filename_stem : str
        Base filename without extension.
    dpi : int, default=150
        Resolution used for PNG export.
    save_png : bool, default=True
        Whether to save a PNG version.
    save_pdf : bool, default=True
        Whether to save a PDF version.

    Returns
    -------
    dict[str, Path]
        Mapping from file type to saved file path.
    """
    output_dir = ensure_directory(Path(output_dir))
    saved_paths: dict[str, Path] = {}

    if save_png:
        png_path = output_dir / f"{filename_stem}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        saved_paths["png"] = png_path

    if save_pdf:
        pdf_path = output_dir / f"{filename_stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        saved_paths["pdf"] = pdf_path

    return saved_paths