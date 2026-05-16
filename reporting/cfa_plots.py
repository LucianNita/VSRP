# =============================================================================
# Plotting helpers for Cost Function Approximation (CFA) experiments.
#
# Purpose
# -------
# This module visualizes:
# - theta evolution during training
# - realized policy comparison
# - average realized performance
# - tail-risk distributions
# - regulatory compliance and ETS exposure
#
# Important interpretation
# ------------------------
# CFA policies should be compared using realized metrics rather than raw
# optimization objective values, because different policies solve
# differently tightened optimization instances.
# =============================================================================

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from reporting.export import save_figure


# =============================================================================
# 1. THETA EVOLUTION
# =============================================================================

def plot_theta_evolution(
    theta_history: list[dict[int, float]],
    *,
    ports: list[str],
    output_dir: str | Path = "results/figures/cfa",
    filename_stem: str = "theta_evolution",
    title: str = "Theta Evolution During CFA Training",
) -> dict[str, Path] | None:
    """
    Plot theta trajectories by port across training episodes.

    Only ports with nonzero theta trajectories are shown explicitly.
    If all theta values remain zero, a flat reference line is plotted.
    """
    if not theta_history:
        return None

    episodes = list(range(len(theta_history)))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ports)))

    plotted_any = False
    for p, color in zip(range(len(ports)), colors):
        values = [theta.get(p, 0.0) for theta in theta_history]
        if max(values) > 1e-9:
            ax.plot(
                episodes,
                values,
                label=ports[p],
                color=color,
                linewidth=2,
            )
            plotted_any = True

    if not plotted_any:
        ax.plot(
            episodes,
            [0.0] * len(episodes),
            color="gray",
            linewidth=1.5,
            label="All zero",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Theta tightening (hours)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 2. POLICY COMPARISON
# =============================================================================

def plot_cfa_policy_comparison(
    baseline_df: pd.DataFrame,
    additive_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    spsa_df: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "results/figures/cfa",
    filename_stem: str = "cfa_policy_comparison",
) -> dict[str, Path] | None:
    """
    Create a 1x3 figure comparing CFA policies on realized metrics.

    Panels
    ------
    (a) cumulative realized misses
    (b) realized service cost by episode
    (c) realized ETS cost by episode

    Important
    ---------
    Optimization objective values are intentionally not shown because
    they are not directly comparable across policies: each policy solves
    a differently tightened optimization instance.
    """
    dfs = {
        "Baseline": (baseline_df, "tomato", "-"),
        "Additive": (additive_df, "steelblue", "-"),
        "Decay": (decay_df, "seagreen", "--"),
    }
    if spsa_df is not None and not spsa_df.empty:
        dfs["SPSA"] = (spsa_df, "purple", ":")

    if any(df.empty for df in [baseline_df, additive_df, decay_df]):
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "CFA Policy Comparison (Realized Metrics Only)",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # (a) Cumulative realized misses
    # -----------------------------------------------------------------
    ax = axes[0]
    for label, (df, color, ls) in dfs.items():
        if "realized_total_missed" in df.columns:
            ax.plot(
                df["episode"],
                df["realized_total_missed"].cumsum(),
                color=color,
                linewidth=2,
                linestyle=ls,
                label=label,
            )
    ax.set_title("(a) Cumulative Realized Misses")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative missed containers")
    ax.grid(alpha=0.3)
    ax.legend()

    # -----------------------------------------------------------------
    # (b) Realized service cost
    # -----------------------------------------------------------------
    ax = axes[1]
    for label, (df, color, ls) in dfs.items():
        if "realized_service_cost_usd" in df.columns:
            ax.plot(
                df["episode"],
                df["realized_service_cost_usd"] / 1000.0,
                color=color,
                linewidth=1.8,
                linestyle=ls,
                label=label,
            )
    ax.set_title("(b) Realized Service Cost")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Service cost ($000s)")
    ax.grid(alpha=0.3)
    ax.legend()

    # -----------------------------------------------------------------
    # (c) Realized ETS cost
    # -----------------------------------------------------------------
    ax = axes[2]
    has_ets = False
    for label, (df, color, ls) in dfs.items():
        if "realized_ets_cost_eur" in df.columns:
            ax.plot(
                df["episode"],
                df["realized_ets_cost_eur"],
                color=color,
                linewidth=1.8,
                linestyle=ls,
                label=label,
            )
            has_ets = True

    if has_ets:
        ax.set_title("(c) Realized ETS Cost")
        ax.set_ylabel("ETS cost (EUR)")
    else:
        # Fallback when ETS is unavailable in the result rows.
        for label, (df, color, ls) in dfs.items():
            if "realized_delay_cost_usd" in df.columns:
                ax.plot(
                    df["episode"],
                    df["realized_delay_cost_usd"] / 1000.0,
                    color=color,
                    linewidth=1.8,
                    linestyle=ls,
                    label=label,
                )
        ax.set_title("(c) Realized Delay Cost")
        ax.set_ylabel("Delay cost ($000s)")

    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.text(
        0.5,
        -0.02,
        "Note: Optimization objectives are intentionally omitted because they are not "
        "directly comparable across CFA policies. Use realized metrics for fair comparison.",
        ha="center",
        fontsize=8,
        color="gray",
        style="italic",
    )

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 3. SUMMARY BAR CHART
# =============================================================================

def plot_cfa_summary_bars(
    baseline_df: pd.DataFrame,
    additive_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    spsa_df: pd.DataFrame | None = None,
    *,
    output_dir: str | Path = "results/figures/cfa",
    filename_stem: str = "cfa_summary_bars",
) -> dict[str, Path] | None:
    """
    Create bar charts comparing average realized policy performance.

    Panels
    ------
    (a) average realized misses
    (b) average realized service cost
    (c) average realized ETS cost
    """
    policy_data = [
        ("Baseline", baseline_df, "tomato"),
        ("Additive", additive_df, "steelblue"),
        ("Decay", decay_df, "seagreen"),
    ]
    if spsa_df is not None and not spsa_df.empty:
        policy_data.append(("SPSA", spsa_df, "purple"))

    if any(df.empty for _, df, _ in policy_data[:3]):
        return None

    labels = [p[0] for p in policy_data]
    colors = [p[2] for p in policy_data]

    def _safe_mean(df: pd.DataFrame, col: str) -> float:
        if col in df.columns:
            return float(df[col].mean())
        return 0.0

    avg_missed = [
        _safe_mean(df, "realized_total_missed")
        for _, df, _ in policy_data
    ]
    avg_service_cost = [
        _safe_mean(df, "realized_service_cost_usd") / 1000.0
        for _, df, _ in policy_data
    ]
    avg_ets_cost = [
        _safe_mean(df, "realized_ets_cost_eur")
        for _, df, _ in policy_data
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5 * len(labels), 5))
    fig.suptitle(
        "CFA Average Realized Policy Performance",
        fontsize=13,
        fontweight="bold",
    )

    def _bar_panel(ax, values, ylabel, title):
        bars = ax.bar(
            labels,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    _bar_panel(
        axes[0],
        avg_missed,
        "Missed containers",
        "(a) Avg Realized Misses",
    )
    _bar_panel(
        axes[1],
        avg_service_cost,
        "Service cost ($000s)",
        "(b) Avg Realized Service Cost",
    )
    _bar_panel(
        axes[2],
        avg_ets_cost,
        "ETS cost (EUR)",
        "(c) Avg Realized ETS Cost",
    )

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 4. TAIL RISK PLOT
# =============================================================================

def plot_cfa_tail_risk(
    baseline_df: pd.DataFrame,
    additive_df: pd.DataFrame,
    decay_df: pd.DataFrame,
    spsa_df: pd.DataFrame | None = None,
    *,
    cost_col: str = "realized_service_cost_usd",
    output_dir: str | Path = "results/figures/cfa",
    filename_stem: str = "cfa_tail_risk",
) -> dict[str, Path] | None:
    """
    Create box plots of realized service cost by policy with p90/p95 annotations.

    This plot emphasizes distributional performance and tail exposure
    rather than only average outcomes.
    """
    policy_data = [
        ("Baseline", baseline_df, "tomato"),
        ("Additive", additive_df, "steelblue"),
        ("Decay", decay_df, "seagreen"),
    ]
    if spsa_df is not None and not spsa_df.empty:
        policy_data.append(("SPSA", spsa_df, "purple"))

    if any(df.empty for _, df, _ in policy_data[:3]):
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    data_arrays = []
    labels = []
    colors = []

    for label, df, color in policy_data:
        if cost_col in df.columns:
            data_arrays.append(
                (df[cost_col].dropna() / 1000.0).values
            )
            labels.append(label)
            colors.append(color)

    if not data_arrays:
        return None

    bp = ax.boxplot(
        data_arrays,
        labels=labels,
        patch_artist=True,
        notch=False,
        showfliers=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, (label, arr) in enumerate(zip(labels, data_arrays), start=1):
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))

        ax.annotate(
            f"p90={p90:.1f}",
            xy=(i, p90),
            xytext=(i + 0.25, p90),
            fontsize=8,
            color="darkred",
            arrowprops=dict(arrowstyle="-", color="darkred", lw=0.8),
        )
        ax.annotate(
            f"p95={p95:.1f}",
            xy=(i, p95),
            xytext=(i + 0.25, p95 + (p95 - p90) * 0.3),
            fontsize=8,
            color="black",
            arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
        )

    ax.set_title(
        "Tail Risk: Realized Service Cost Distribution by Policy",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Realized service cost ($000s)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved


# =============================================================================
# 5. REGULATORY COMPLIANCE AND ETS
# =============================================================================

def plot_regulatory_compliance_and_ets(
    policy_dfs: dict[str, pd.DataFrame],
    *,
    output_dir: str | Path = "results/figures/cfa",
    filename_stem: str = "ets_compliance_rate",
) -> dict[str, Path] | None:
    """
    Create a two-panel figure for FuelEU compliance rate and average ETS cost.

    Notes
    -----
    Under the current single-fuel VLSFO assumption, FuelEU compliance is
    structurally expected to be 0% for post-2025 limits because VLSFO
    exceeds the applicable GHG-intensity threshold. The compliance panel
    is still useful because it:
    - documents regulatory exposure explicitly
    - provides a reusable structure for future multi-fuel extensions
    - complements the ETS-cost panel
    """
    if not policy_dfs:
        return None

    labels = list(policy_dfs.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fueleu_rates = []
    avg_ets_costs = []

    for label in labels:
        df = policy_dfs[label]
        if "realized_fueleu_compliant" in df.columns:
            fueleu_rates.append(
                float(df["realized_fueleu_compliant"].mean()) * 100.0
            )
        else:
            fueleu_rates.append(0.0)

        if "realized_ets_cost_eur" in df.columns:
            avg_ets_costs.append(float(df["realized_ets_cost_eur"].mean()))
        else:
            avg_ets_costs.append(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Regulatory Compliance and ETS Exposure by Policy",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: FuelEU compliance rate
    ax = axes[0]
    bars = ax.bar(
        labels,
        fueleu_rates,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, val in zip(bars, fueleu_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("(a) FuelEU Compliance Rate")
    ax.set_ylabel("% episodes compliant")
    ax.set_ylim(0, 110)
    ax.axhline(
        y=0,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label="VLSFO always non-compliant\n(91.16 > 89.34 gCO2eq/MJ)",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Average ETS cost
    ax = axes[1]
    bars = ax.bar(
        labels,
        avg_ets_costs,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, val in zip(bars, avg_ets_costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"€{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("(b) Average Realized ETS Cost")
    ax.set_ylabel("ETS cost (EUR)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved