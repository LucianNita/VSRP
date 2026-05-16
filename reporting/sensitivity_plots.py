# =============================================================================
# Plotting helpers for deterministic sensitivity analysis in the VSRP codebase.
#
# Purpose
# -------
# This module visualizes the main comparative-static experiments:
# - alpha sweep
# - initial-delay sweep
# - alpha trade-off frontier
#
# Architectural role
# ------------------
# The functions in this file consume the tidy DataFrames returned by
# `experiments/sensitivity.py` and convert them into reusable figure
# artifacts for the reporting layer.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reporting.export import save_figure


# =============================================================================
# 1. ALPHA SWEEP PLOT
# =============================================================================

def plot_alpha_sweep(
    alpha_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/sensitivity",
    filename_stem: str = "alpha_sensitivity",
    exclude_alpha_1: bool = True,
) -> dict[str, Path] | None:
    """
    Create a 2x2 figure summarizing the alpha sweep.

    Panels
    ------
    1. objective vs alpha
    2. operational vs service cost vs alpha
    3. delayed-container and skipped-port counts vs alpha
    4. total CO2 vs alpha

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Output of the alpha sweep experiment.
    output_dir : str | Path, default="results/figures/sensitivity"
        Figure output directory.
    filename_stem : str, default="alpha_sensitivity"
        Base filename for saved figure files.
    exclude_alpha_1 : bool, default=True
        Whether to omit the pure service-weight endpoint from the plot
        for readability.
    """
    if alpha_df.empty:
        return None

    df = alpha_df.copy()
    df = df[df["feasible"] == True].copy()

    if exclude_alpha_1:
        df = df[df["alpha"] < 1.0].copy()

    if df.empty:
        return None

    df = df.sort_values("alpha")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Alpha Sensitivity Analysis",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # 1. Objective vs alpha
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(
        df["alpha"],
        df["objective_value"] / 1000.0,
        color="steelblue",
        marker="o",
        linewidth=2,
    )
    ax.set_title("(a) Objective vs Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Objective ($000s)")
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # 2. Cost decomposition vs alpha
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(
        df["alpha"],
        df["operational_cost_usd"] / 1000.0,
        color="darkgreen",
        marker="o",
        linewidth=2,
        label="Operational cost",
    )
    ax.plot(
        df["alpha"],
        df["service_cost_usd"] / 1000.0,
        color="tomato",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Service cost",
    )
    ax.set_title("(b) Cost Decomposition vs Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Cost ($000s)")
    ax.legend()
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # 3. Service outcomes vs alpha
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(
        df["alpha"],
        df["n_delayed"],
        color="purple",
        marker="o",
        linewidth=2,
        label="Delayed containers",
    )
    ax.plot(
        df["alpha"],
        df["n_skipped"],
        color="orange",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Skipped ports",
    )
    ax.set_title("(c) Service Outcomes vs Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # 4. CO2 vs alpha
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(
        df["alpha"],
        df["total_co2_t"],
        color="black",
        marker="o",
        linewidth=2,
    )
    ax.set_title("(d) CO2 Emissions vs Alpha")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("CO2 (tonnes)")
    ax.grid(alpha=0.3)

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


# =============================================================================
# 2. DELAY SWEEP PLOT
# =============================================================================

def plot_delay_sweep(
    delay_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/sensitivity",
    filename_stem: str = "delay_sensitivity",
) -> dict[str, Path] | None:
    """
    Create a 2x2 figure summarizing the initial-delay sweep.

    Panels
    ------
    1. objective vs initial delay
    2. operational vs service cost vs initial delay
    3. delayed-container and skipped-port counts vs initial delay
    4. total CO2 and ETS cost vs initial delay

    Parameters
    ----------
    delay_df : pd.DataFrame
        Output of the delay sweep experiment.
    output_dir : str | Path, default="results/figures/sensitivity"
        Figure output directory.
    filename_stem : str, default="delay_sensitivity"
        Base filename for saved figure files.
    """
    if delay_df.empty:
        return None

    df = delay_df.copy()
    df = df[df["feasible"] == True].copy()

    if df.empty:
        return None

    df = df.sort_values("initial_delay_h")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Initial Delay Sensitivity Analysis",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # 1. Objective vs delay
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(
        df["initial_delay_h"],
        df["objective_value"] / 1000.0,
        color="steelblue",
        marker="o",
        linewidth=2,
    )
    ax.set_title("(a) Objective vs Initial Delay")
    ax.set_xlabel("Initial delay (hours)")
    ax.set_ylabel("Objective ($000s)")
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # 2. Cost decomposition vs delay
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(
        df["initial_delay_h"],
        df["operational_cost_usd"] / 1000.0,
        color="darkgreen",
        marker="o",
        linewidth=2,
        label="Operational cost",
    )
    ax.plot(
        df["initial_delay_h"],
        df["service_cost_usd"] / 1000.0,
        color="tomato",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Service cost",
    )
    ax.set_title("(b) Cost Decomposition vs Delay")
    ax.set_xlabel("Initial delay (hours)")
    ax.set_ylabel("Cost ($000s)")
    ax.legend()
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # 3. Service outcomes vs delay
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(
        df["initial_delay_h"],
        df["n_delayed"],
        color="purple",
        marker="o",
        linewidth=2,
        label="Delayed containers",
    )
    ax.plot(
        df["initial_delay_h"],
        df["n_skipped"],
        color="orange",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Skipped ports",
    )
    ax.set_title("(c) Service Outcomes vs Delay")
    ax.set_xlabel("Initial delay (hours)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # 4. CO2 and ETS vs delay
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    ax2 = ax.twinx()

    l1 = ax.plot(
        df["initial_delay_h"],
        df["total_co2_t"],
        color="black",
        marker="o",
        linewidth=2,
        label="CO2 (t)",
    )
    l2 = ax2.plot(
        df["initial_delay_h"],
        df["total_ets_eur"] / 1000.0,
        color="red",
        marker="s",
        linewidth=2,
        linestyle="--",
        label="ETS (€000s)",
    )

    ax.set_title("(d) Emissions vs Delay")
    ax.set_xlabel("Initial delay (hours)")
    ax.set_ylabel("CO2 (tonnes)", color="black")
    ax2.set_ylabel("ETS (€000s)", color="red")
    ax.grid(alpha=0.3)

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper right")

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


def plot_alpha_pareto(
    alpha_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/sensitivity",
    filename_stem: str = "alpha_pareto",
    exclude_alpha_1: bool = True,
) -> dict[str, Path] | None:
    """
    Create a Pareto-style operational-cost vs service-cost trade-off plot.

    Visual encoding
    ---------------
    - x-axis: operational cost
    - y-axis: service cost
    - point color: alpha value
    - annotations: numeric alpha labels

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Output of the alpha sweep experiment.
    output_dir : str | Path, default="results/figures/sensitivity"
        Figure output directory.
    filename_stem : str, default="alpha_pareto"
        Base filename for saved figure files.
    exclude_alpha_1 : bool, default=True
        Whether to omit the pure service-weight endpoint from the plot
        for readability.
    """
    if alpha_df.empty:
        return None

    df = alpha_df.copy()
    df = df[df["feasible"] == True].copy()

    if exclude_alpha_1:
        df = df[df["alpha"] < 1.0].copy()

    if df.empty:
        return None

    df = df.sort_values("alpha")

    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(
        df["operational_cost_usd"] / 1000.0,
        df["service_cost_usd"] / 1000.0,
        c=df["alpha"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    # Connect points in alpha order to emphasize the trade-off frontier.
    ax.plot(
        df["operational_cost_usd"] / 1000.0,
        df["service_cost_usd"] / 1000.0,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    )

    for _, row in df.iterrows():
        x = row["operational_cost_usd"] / 1000.0
        y = row["service_cost_usd"] / 1000.0
        alpha = row["alpha"]

        # Small manual offsets to keep labels readable where points cluster.
        if alpha in {0.1, 0.2}:
            offset = (6, 10 if alpha == 0.1 else -2)
        elif alpha in {0.7, 0.8, 0.9}:
            offset = (6, 8 + int((alpha - 0.7) * 20))
        else:
            offset = (5, 5)

        ax.annotate(
            f"{alpha:.1f}",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Alpha", fontsize=10)

    ax.set_title(
        "Alpha Trade-off Frontier: Operational vs Service Cost",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Operational Cost ($000s)")
    ax.set_ylabel("Service Cost ($000s)")
    ax.grid(alpha=0.3)

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