# =============================================================================
# Green recovery and regulatory-exposure plotting helpers for the VSRP.
#
# Plots produced
# --------------
# 1. plot_green_recovery_tradeoff
#    Cost / service / emissions trade-off across alpha values
#
# 2. plot_emissions_vs_delay_penalty
#    CO2 emissions vs delay penalty across disruption scenarios
#
# 3. plot_strategy_mix_under_ets
#    Strategy counts and emissions across ETS carbon-price scenarios
#
# 4. plot_ets_phase_in_impact
#    ETS cost vs regulatory year under the EU ETS phase-in schedule
#
# 5. plot_cii_profile
#    Attained vs required CII across delay scenarios
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from reporting.export import save_figure


# =============================================================================
# 1. GREEN RECOVERY TRADE-OFF
# =============================================================================

def plot_green_recovery_tradeoff(
    alpha_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/emissions",
    filename_stem: str = "green_recovery_tradeoff",
    exclude_alpha_1: bool = True,
) -> dict[str, Path] | None:
    """
    Plot cost, emissions, and regulatory-cost proxies across alpha values.

    The figure highlights the trade-off between:
    - operational cost
    - service cost
    - emissions / regulatory burden

    Parameters
    ----------
    alpha_df : pd.DataFrame
        Output of the alpha sensitivity experiment.
    exclude_alpha_1 : bool
        When True, exclude the pure service-weight endpoint
        $$\alpha = 1.0$$ from the plot for readability.
    """
    if alpha_df.empty:
        return None

    df = alpha_df[alpha_df["feasible"] == True].copy()
    if exclude_alpha_1:
        df = df[df["alpha"] < 1.0].copy()
    if df.empty:
        return None

    df = df.sort_values("alpha")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Green Recovery Trade-off: Cost, Service, and Emissions",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # Panel (a): operational vs service cost
    # -----------------------------------------------------------------
    ax = axes[0]
    ax.plot(
        df["alpha"],
        df["operational_cost_usd"] / 1000.0,
        color="steelblue",
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
    ax.set_title("(a) Cost Components vs Alpha")
    ax.set_xlabel("Alpha (service weight)")
    ax.set_ylabel("Cost ($000s)")
    ax.legend()
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # Panel (b): CO2 and ETS cost vs alpha
    # -----------------------------------------------------------------
    ax = axes[1]
    line1 = ax.plot(
        df["alpha"],
        df["total_co2_t"],
        color="black",
        marker="o",
        linewidth=2,
        label="CO2 (t)",
    )
    ax.set_title("(b) Emissions vs Alpha")
    ax.set_xlabel("Alpha (service weight)")
    ax.set_ylabel("CO2 (tonnes)", color="black")
    ax.grid(alpha=0.3)

    lines = list(line1)
    labels = [line1[0].get_label()]

    if "total_ets_eur" in df.columns:
        ax2 = ax.twinx()
        line2 = ax2.plot(
            df["alpha"],
            df["total_ets_eur"] / 1000.0,
            color="darkred",
            marker="^",
            linewidth=1.5,
            linestyle=":",
            label="ETS (€000s)",
        )
        ax2.set_ylabel("ETS cost (€000s)", color="darkred")
        lines += list(line2)
        labels += [line2[0].get_label()]

    ax.legend(lines, labels, loc="best")

    # -----------------------------------------------------------------
    # Panel (c): FuelEU penalty or delayed containers vs alpha
    # -----------------------------------------------------------------
    ax = axes[2]
    if "total_fueleu_penalty_usd" in df.columns:
        ax.plot(
            df["alpha"],
            df["total_fueleu_penalty_usd"] / 1000.0,
            color="darkorange",
            marker="D",
            linewidth=2,
            label="FuelEU penalty",
        )
        ax.set_ylabel("FuelEU penalty ($000s)")
        ax.set_title("(c) FuelEU Cost vs Alpha")
    else:
        ax.plot(
            df["alpha"],
            df["n_delayed"],
            color="purple",
            marker="o",
            linewidth=2,
            label="Delayed containers",
        )
        ax.set_ylabel("Count")
        ax.set_title("(c) Delayed Containers vs Alpha")

    ax.set_xlabel("Alpha (service weight)")
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


# =============================================================================
# 2. EMISSIONS VS DELAY PENALTY SCATTER
# =============================================================================

def plot_emissions_vs_delay_penalty(
    delay_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/emissions",
    filename_stem: str = "emissions_vs_delay_penalty",
) -> dict[str, Path] | None:
    """
    Plot CO2 emissions against delay penalty across disruption scenarios.

    Each point is labelled by the initial disruption delay. This plot is
    intended as a compact visualisation of the trade-off between:
    - environmental burden
    - service penalty exposure
    """
    if delay_df.empty:
        return None

    df = delay_df[delay_df["feasible"] == True].copy()
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(
        df["total_co2_t"],
        df["delay_cost_usd"] / 1000.0,
        c=df["initial_delay_h"],
        cmap="plasma",
        s=90,
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"{int(row['initial_delay_h'])}h",
            (row["total_co2_t"], row["delay_cost_usd"] / 1000.0),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Initial delay (hours)", fontsize=10)

    ax.set_title(
        "Emissions vs Delay Penalty Across Disruption Scenarios",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Total CO2 (tonnes)")
    ax.set_ylabel("Delay penalty cost ($000s)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 3. STRATEGY MIX UNDER ETS TIGHTENING
# =============================================================================

def plot_strategy_mix_under_ets(
    ets_sweep_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/emissions",
    filename_stem: str = "strategy_mix_ets",
) -> dict[str, Path] | None:
    """
    Plot strategy counts and emissions across ETS carbon-price scenarios.

    Expected columns
    ----------------
    carbon_price_eur, n_skipped, n_swapped, n_expedited, n_speed_up

    Note
    ----
    In the current implementation, ETS carbon price is varied in the
    post-solve reporting layer rather than being directly internalised
    into the optimization objective. The figure should therefore be
    interpreted as an ETS exposure experiment, not as a fully endogenous
    carbon-price-in-objective response curve.
    """
    required = {"carbon_price_eur", "n_skipped"}
    if ets_sweep_df.empty or not required.issubset(ets_sweep_df.columns):
        return None

    df = ets_sweep_df.sort_values("carbon_price_eur").copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Strategy Mix and ETS Exposure Across Carbon Prices",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # Panel (a): stacked strategy counts
    # -----------------------------------------------------------------
    ax = axes[0]
    x = df["carbon_price_eur"].values
    width = (x[1] - x[0]) * 0.6 if len(x) > 1 else 10.0

    strategy_cols = [
        ("n_skipped", "Port Omission", "tomato"),
        ("n_swapped", "Port Swap", "steelblue"),
        ("n_expedited", "Expedited Port", "seagreen"),
        ("n_speed_up", "Speed Up", "orange"),
    ]

    bottom = np.zeros(len(df))
    for col, label, color in strategy_cols:
        if col in df.columns:
            values = df[col].fillna(0).values
            ax.bar(
                x,
                values,
                width=width,
                bottom=bottom,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += values

    ax.set_title("(a) Strategy Mix vs Carbon Price")
    ax.set_xlabel("ETS carbon price (€/tCO2)")
    ax.set_ylabel("Strategy count")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # -----------------------------------------------------------------
    # Panel (b): CO2 and ETS cost vs carbon price
    # -----------------------------------------------------------------
    ax = axes[1]
    lines: list = []
    labels: list[str] = []

    if "total_co2_t" in df.columns:
        line1 = ax.plot(
            df["carbon_price_eur"],
            df["total_co2_t"],
            color="black",
            marker="o",
            linewidth=2,
            label="CO2 (t)",
        )
        ax.set_ylabel("CO2 (tonnes)", color="black")
        lines += list(line1)
        labels += [line1[0].get_label()]

    if "total_ets_eur_at_price" in df.columns:
        ax2 = ax.twinx()
        line2 = ax2.plot(
            df["carbon_price_eur"],
            df["total_ets_eur_at_price"] / 1000.0,
            color="darkred",
            marker="s",
            linewidth=2,
            linestyle="--",
            label="ETS cost at price (€000s)",
        )
        ax2.set_ylabel("ETS cost (€000s)", color="darkred")
        lines += list(line2)
        labels += [line2[0].get_label()]
    elif "total_ets_eur" in df.columns:
        ax2 = ax.twinx()
        line2 = ax2.plot(
            df["carbon_price_eur"],
            df["total_ets_eur"] / 1000.0,
            color="darkred",
            marker="s",
            linewidth=2,
            linestyle="--",
            label="ETS cost (€000s)",
        )
        ax2.set_ylabel("ETS cost (€000s)", color="darkred")
        lines += list(line2)
        labels += [line2[0].get_label()]

    ax.set_title("(b) Emissions and ETS Cost vs Carbon Price")
    ax.set_xlabel("ETS carbon price (€/tCO2)")
    ax.grid(alpha=0.3)

    if lines:
        ax.legend(lines, labels, loc="upper left", fontsize=9)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 4. ETS PHASE-IN IMPACT
# =============================================================================

def plot_ets_phase_in_impact(
    years: list[int],
    ets_costs_eur: list[float],
    co2_tonnes: list[float],
    *,
    output_dir: str | Path = "results/figures/emissions",
    filename_stem: str = "ets_phase_in_impact",
    title: str = "EU ETS Phase-in Impact on Voyage Cost",
) -> dict[str, Path] | None:
    """
    Plot ETS cost and CO2 emissions against regulatory year.

    This figure is intended to show how the EU ETS phase-in schedule
    increases the financial burden of a fixed emissions profile.
    """
    if not years or not ets_costs_eur or not co2_tonnes:
        return None

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_ets = "darkred"
    color_co2 = "steelblue"

    ax1.bar(
        years,
        [c / 1000.0 for c in ets_costs_eur],
        color=color_ets,
        alpha=0.7,
        label="ETS cost (€000s)",
        width=0.4,
    )
    ax1.set_xlabel("Regulatory year")
    ax1.set_ylabel("ETS cost (€000s)", color=color_ets)
    ax1.tick_params(axis="y", labelcolor=color_ets)

    ax2 = ax1.twinx()
    ax2.plot(
        years,
        co2_tonnes,
        color=color_co2,
        marker="o",
        linewidth=2,
        label="CO2 (tonnes)",
    )
    ax2.set_ylabel("CO2 (tonnes)", color=color_co2)
    ax2.tick_params(axis="y", labelcolor=color_co2)

    phase_in = {2024: "40%", 2025: "70%", 2026: "100%"}
    for year, fraction in phase_in.items():
        if year in years:
            idx = years.index(year)
            ax1.annotate(
                f"Phase-in:\n{fraction}",
                xy=(year, ets_costs_eur[idx] / 1000.0),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color_ets,
            )

    lines1 = [mpatches.Patch(color=color_ets, label="ETS cost (€000s)")]
    lines2 = [plt.Line2D([0], [0], color=color_co2, marker="o", label="CO2 (t)")]
    ax1.legend(handles=lines1 + lines2, loc="upper left", fontsize=9)

    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 5. CII PROFILE ACROSS DELAY SCENARIOS
# =============================================================================

def plot_cii_profile(
    delay_df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/emissions",
    filename_stem: str = "cii_profile",
) -> dict[str, Path] | None:
    """
    Plot attained and required CII across delay scenarios.

    Points are coloured by reported CII rating, allowing the figure to
    show both the magnitude of attained CII and the categorical rating
    outcome under different disruption levels.
    """
    required_cols = {"initial_delay_h", "attained_cii", "required_cii"}
    if delay_df.empty or not required_cols.issubset(delay_df.columns):
        return None

    df = delay_df[delay_df["feasible"] == True].copy()
    df = df.sort_values("initial_delay_h")
    if df.empty:
        return None

    rating_colors = {
        "A": "seagreen",
        "B": "limegreen",
        "C": "gold",
        "D": "orange",
        "E": "tomato",
        "UNKNOWN": "gray",
    }

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        df["initial_delay_h"],
        df["required_cii"],
        color="black",
        linewidth=2,
        linestyle="--",
        label="Required CII",
        zorder=4,
    )

    for _, row in df.iterrows():
        rating = row.get("cii_rating", "UNKNOWN")
        color = rating_colors.get(rating, "gray")
        ax.scatter(
            row["initial_delay_h"],
            row["attained_cii"],
            color=color,
            s=100,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

    ax.plot(
        df["initial_delay_h"],
        df["attained_cii"],
        color="steelblue",
        linewidth=1.5,
        alpha=0.6,
        zorder=3,
    )

    legend_patches = [
        mpatches.Patch(color=color, label=f"Rating {rating}")
        for rating, color in rating_colors.items()
        if rating != "UNKNOWN"
    ]
    legend_patches.append(
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            label="Required CII",
        )
    )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9, ncol=2)

    ax.set_title(
        "CII Profile Across Disruption Scenarios",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Initial delay (hours)")
    ax.set_ylabel("CII [gCO2 / (DWT·nm)]")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved