from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reporting.export import save_figure


def plot_penalty_sweep_overview(
    df: pd.DataFrame,
    *,
    penalty_name: str,
    output_dir: str | Path = "results/figures/penalty_sensitivity",
    filename_stem: str | None = None,
) -> dict[str, Path] | None:
    """
    Plot a 2x2 overview of a penalty sensitivity experiment.

    This function is especially useful for multi-seed penalty sweeps,
    where it summarizes the average response of the system and the
    fraction of seeds that experience structural route or strategy change.

    Expected columns
    ----------------
    The input table is expected to contain most or all of:
    - `penalty_value`
    - `objective_value`
    - `n_delayed`
    - `n_skipped`
    - `n_swapped`
    - `total_co2_t`
    - `total_ets_eur`
    - `route_changed_vs_first`
    - `strategy_changed_vs_first`

    Panels
    ------
    (a) average objective
    (b) average service / strategy counts
    (c) average emissions and ETS cost
    (d) fraction of seeds with route/strategy change
    """
    if df.empty:
        return None

    filename_stem = filename_stem or f"{penalty_name}_overview"

    plot_df = df.copy()
    if "feasible" in plot_df.columns:
        plot_df = plot_df[plot_df["feasible"] == True].copy()

    if plot_df.empty:
        return None

    agg = (
        plot_df.groupby("penalty_value").agg(
            avg_objective=("objective_value", "mean"),
            avg_n_delayed=("n_delayed", "mean"),
            avg_n_skipped=("n_skipped", "mean"),
            avg_n_swapped=("n_swapped", "mean"),
            avg_total_co2_t=("total_co2_t", "mean"),
            avg_total_ets_eur=("total_ets_eur", "mean"),
            route_change_rate=("route_changed_vs_first", "mean"),
            strategy_change_rate=("strategy_changed_vs_first", "mean"),
        ).reset_index().sort_values("penalty_value")
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Penalty Sensitivity Overview: {penalty_name}",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # (a) Objective
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(
        agg["penalty_value"],
        agg["avg_objective"] / 1000.0,
        marker="o",
        linewidth=2,
        color="steelblue",
    )
    ax.set_title("(a) Average Objective")
    ax.set_xlabel("Penalty value (USD)")
    ax.set_ylabel("Objective ($000s)")
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # (b) Service / strategy outcomes
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(
        agg["penalty_value"],
        agg["avg_n_delayed"],
        marker="o",
        linewidth=2,
        color="tomato",
        label="Delayed",
    )
    ax.plot(
        agg["penalty_value"],
        agg["avg_n_skipped"],
        marker="s",
        linewidth=2,
        color="orange",
        linestyle="--",
        label="Skipped",
    )
    ax.plot(
        agg["penalty_value"],
        agg["avg_n_swapped"],
        marker="^",
        linewidth=2,
        color="purple",
        linestyle=":",
        label="Swapped",
    )
    ax.set_title("(b) Average Service / Strategy Counts")
    ax.set_xlabel("Penalty value (USD)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # (c) Emissions
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(
        agg["penalty_value"],
        agg["avg_total_co2_t"],
        marker="o",
        linewidth=2,
        color="black",
        label="CO2 (t)",
    )
    ax2 = ax.twinx()
    ax2.plot(
        agg["penalty_value"],
        agg["avg_total_ets_eur"] / 1000.0,
        marker="s",
        linewidth=2,
        color="darkred",
        linestyle="--",
        label="ETS (€000s)",
    )
    ax.set_title("(c) Average Emissions")
    ax.set_xlabel("Penalty value (USD)")
    ax.set_ylabel("CO2 (tonnes)", color="black")
    ax2.set_ylabel("ETS (€000s)", color="darkred")
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # (d) Change rates
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(
        agg["penalty_value"],
        agg["route_change_rate"] * 100.0,
        marker="o",
        linewidth=2,
        color="seagreen",
        label="Route change rate",
    )
    ax.plot(
        agg["penalty_value"],
        agg["strategy_change_rate"] * 100.0,
        marker="s",
        linewidth=2,
        color="mediumpurple",
        linestyle="--",
        label="Strategy change rate",
    )
    ax.set_title("(d) Fraction of Seeds with Change")
    ax.set_xlabel("Penalty value (USD)")
    ax.set_ylabel("Percent of seeds (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved