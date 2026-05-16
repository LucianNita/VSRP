from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting.export import save_figure


def plot_fleet_scenario_comparison(
    raw_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/fleet",
    filename_stem: str = "fleet_scenario_comparison",
) -> dict[str, Path] | None:
    """
    Compare fleet-level metrics across scenarios and solver backends.

    Panels
    ------
    (a) Fleet objective by scenario and solver
    (b) Total delayed and misconnected containers by scenario
    (c) Total fleet CO2 by scenario and solver
    (d) Average per-vessel runtime by scenario and solver

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw output from run_fleet_benchmark(), one row per
        (solver, scenario).
    output_dir : str | Path
        Output directory for saved figure files.
    filename_stem : str
        Base filename without extension.

    Returns
    -------
    dict[str, Path] | None
        Saved output paths, or None if the input table is empty.
    """
    if raw_df.empty:
        return None

    df = raw_df[raw_df["feasible"] == True].copy()
    if df.empty:
        return None

    scenarios = df["scenario"].unique()
    solvers = df["solver_name"].unique()
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(solvers), 1)

    solver_colors = {
        "Xpress": "steelblue",
        "HiGHS": "tomato",
        "CBC": "seagreen",
    }
    colors = [solver_colors.get(s, "gray") for s in solvers]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Fleet Disruption Scenario Comparison",
        fontsize=14,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # (a) Fleet objective
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for s_idx, (solver, color) in enumerate(zip(solvers, colors)):
        sdf = df[df["solver_name"] == solver]
        values = [
            sdf[sdf["scenario"] == sc]["fleet_objective_value"].mean()
            / 1000.0
            for sc in scenarios
        ]
        bars = ax.bar(
            x + s_idx * width,
            values,
            width,
            label=solver,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_title("(a) Fleet Objective by Scenario")
    ax.set_xticks(x + width * (len(solvers) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Fleet objective ($000s)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # (b) Service impact
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    delayed_vals = [
        df[df["scenario"] == sc]["total_delayed"].mean()
        for sc in scenarios
    ]
    misconn_vals = [
        df[df["scenario"] == sc]["total_misconnected"].mean()
        for sc in scenarios
    ]
    bar_w = 0.35
    bars1 = ax.bar(
        x - bar_w / 2,
        delayed_vals,
        bar_w,
        label="Delayed",
        color="tomato",
        edgecolor="black",
        linewidth=0.6,
    )
    bars2 = ax.bar(
        x + bar_w / 2,
        misconn_vals,
        bar_w,
        label="Misconnected",
        color="steelblue",
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, val in zip(list(bars1) + list(bars2), delayed_vals + misconn_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("(b) Fleet Service Impact by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Container count (avg across solvers)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # (c) Total CO2
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    has_co2 = "total_co2_t" in df.columns and df["total_co2_t"].notna().any()

    if has_co2:
        for s_idx, (solver, color) in enumerate(zip(solvers, colors)):
            sdf = df[df["solver_name"] == solver]
            values = [
                sdf[sdf["scenario"] == sc]["total_co2_t"].mean()
                for sc in scenarios
            ]
            ax.bar(
                x + s_idx * width,
                values,
                width,
                label=solver,
                color=color,
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_xticks(x + width * (len(solvers) - 1) / 2)
        ax.set_xticklabels(scenarios, rotation=12, ha="right", fontsize=9)
        ax.legend(fontsize=9)
    else:
        ax.text(
            0.5,
            0.5,
            "CO2 data not available\n(open-source backends)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="gray",
        )

    ax.set_title("(c) Total Fleet CO2 by Scenario")
    ax.set_ylabel("CO2 (tonnes)")
    ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # (d) Runtime
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    for s_idx, (solver, color) in enumerate(zip(solvers, colors)):
        sdf = df[df["solver_name"] == solver]
        values = [
            sdf[sdf["scenario"] == sc]["avg_runtime_s"].mean()
            for sc in scenarios
        ]
        bars = ax.bar(
            x + s_idx * width,
            values,
            width,
            label=solver,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_title("(d) Avg Per-Vessel Runtime by Scenario")
    ax.set_xticks(x + width * (len(solvers) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


def plot_fleet_per_vessel_breakdown(
    per_vessel_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/fleet",
    filename_stem: str = "fleet_per_vessel_breakdown",
) -> dict[str, Path] | None:
    """
    Per-vessel breakdown of objective, service impact, and emissions
    within each fleet scenario.

    One row of panels per scenario. Left panel shows per-vessel
    objective. Right panel shows delayed containers and CO2.

    Parameters
    ----------
    per_vessel_df : pd.DataFrame
        Per-vessel results table with columns including scenario,
        vessel_idx, objective_value, n_delayed, and total_co2_t.
    output_dir : str | Path
        Output directory for saved figure files.
    filename_stem : str
        Base filename without extension.

    Returns
    -------
    dict[str, Path] | None
        Saved output paths, or None if the input table is empty.
    """
    if per_vessel_df.empty:
        return None

    scenarios = per_vessel_df["scenario"].unique()
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(
        n_scenarios,
        2,
        figsize=(14, 5 * n_scenarios),
        squeeze=False,
    )
    fig.suptitle(
        "Per-Vessel Breakdown by Fleet Scenario",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, scenario in enumerate(scenarios):
        sdf = (
            per_vessel_df[per_vessel_df["scenario"] == scenario]
            .sort_values("vessel_idx")
        )
        vessel_labels = [f"V{int(v)}" for v in sdf["vessel_idx"]]

        # Left: objective per vessel
        ax = axes[row_idx, 0]
        bars = ax.bar(
            vessel_labels,
            sdf["objective_value"] / 1000.0,
            color="steelblue",
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, sdf["objective_value"] / 1000.0):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"${val:.0f}k",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(f"{scenario} — Objective per Vessel")
        ax.set_ylabel("Objective ($000s)")
        ax.grid(axis="y", alpha=0.3)

        # Right: service and emissions
        ax = axes[row_idx, 1]
        ax2 = ax.twinx()

        ax.bar(
            vessel_labels,
            sdf["n_delayed"],
            color="tomato",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.75,
            label="Delayed",
            width=0.4,
        )
        ax.bar(
            [str(v) + " " for v in vessel_labels],
            sdf["n_misconnected"] if "n_misconnected" in sdf.columns
            else [0] * len(vessel_labels),
            color="orange",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.75,
            label="Misconnected",
            width=0.4,
        )

        has_co2 = (
            "total_co2_t" in sdf.columns
            and sdf["total_co2_t"].notna().any()
        )
        if has_co2:
            ax2.plot(
                vessel_labels,
                sdf["total_co2_t"],
                color="black",
                marker="o",
                linewidth=2,
                label="CO2 (t)",
            )
            ax2.set_ylabel("CO2 (tonnes)", color="black")

        ax.set_title(f"{scenario} — Service & Emissions per Vessel")
        ax.set_ylabel("Container count", color="tomato")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


def plot_fleet_benchmark_summary(
    summary_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/fleet",
    filename_stem: str = "fleet_benchmark_summary",
) -> dict[str, Path] | None:
    """
    Three-panel fleet benchmark summary figure.

    Panels
    ------
    (a) Average fleet objective by solver
    (b) Average total runtime by solver
    (c) Feasibility and optimality rates by solver

    Parameters
    ----------
    summary_df : pd.DataFrame
        Solver-level summary from summarize_fleet_benchmark().
    output_dir : str | Path
        Output directory for saved figure files.
    filename_stem : str
        Base filename without extension.

    Returns
    -------
    dict[str, Path] | None
        Saved output paths, or None if the input table is empty.
    """
    if summary_df.empty:
        return None

    solver_colors = {
        "Xpress": "steelblue",
        "HiGHS": "tomato",
        "CBC": "seagreen",
    }
    bar_colors = [
        solver_colors.get(s, "gray")
        for s in summary_df["solver_name"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Fleet Benchmark Summary",
        fontsize=13,
        fontweight="bold",
    )

    # (a) Average fleet objective
    ax = axes[0]
    if "avg_fleet_objective" in summary_df.columns:
        values = summary_df["avg_fleet_objective"] / 1000.0
        bars = ax.bar(
            summary_df["solver_name"],
            values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"${val:.0f}k",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.set_title("(a) Avg Fleet Objective")
    ax.set_ylabel("USD ($000s)")
    ax.grid(axis="y", alpha=0.3)

    # (b) Average total runtime
    ax = axes[1]
    if "avg_total_runtime_s" in summary_df.columns:
        values = summary_df["avg_total_runtime_s"].fillna(0)
        bars = ax.bar(
            summary_df["solver_name"],
            values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_title("(b) Avg Total Fleet Runtime")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.3)

    # (c) Feasibility and optimality rates
    ax = axes[2]
    x = np.arange(len(summary_df))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        summary_df["pct_feasible"] * 100,
        width,
        label="Feasible %",
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    bars2 = ax.bar(
        x + width / 2,
        summary_df["pct_optimal"] * 100,
        width,
        label="Optimal %",
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["solver_name"])
    ax.set_ylim(0, 115)
    ax.set_title("(c) Feasibility and Optimality Rates")
    ax.set_ylabel("Percent")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved