from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from reporting.export import save_figure


def plot_stochastic_ets_exposure(
    df: pd.DataFrame,
    *,
    output_dir: str | Path = "results/figures/stochastic_ets",
    filename_stem: str = "stochastic_ets_exposure",
) -> dict[str, Path] | None:
    """
    Plot stochastic ETS exposure from a scenario-level results table.

    Panels
    ------
    (a) realized carbon price vs realized ETS cost
    (b) realized ETS cost distribution

    Parameters
    ----------
    df : pd.DataFrame
        Scenario-level ETS exposure table produced by the stochastic
        ETS experiment.
    output_dir : str | Path, default="results/figures/stochastic_ets"
        Output directory for the figure.
    filename_stem : str, default="stochastic_ets_exposure"
        Base filename for saved figure outputs.
    """
    if df.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Stochastic ETS Exposure",
        fontsize=13,
        fontweight="bold",
    )

    # -----------------------------------------------------------------
    # (a) Carbon price vs ETS cost
    # -----------------------------------------------------------------
    ax = axes[0]
    ax.scatter(
        df["realized_carbon_price_eur"],
        df["realized_ets_cost_eur"],
        color="steelblue",
        edgecolors="black",
        linewidths=0.5,
        alpha=0.8,
    )
    ax.set_title("(a) Carbon Price vs ETS Cost")
    ax.set_xlabel("Realized carbon price (€/tCO2)")
    ax.set_ylabel("Realized ETS cost (EUR)")
    ax.grid(alpha=0.3)

    # -----------------------------------------------------------------
    # (b) ETS cost histogram
    # -----------------------------------------------------------------
    ax = axes[1]
    ax.hist(
        df["realized_ets_cost_eur"].dropna(),
        bins=15,
        color="tomato",
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_title("(b) ETS Cost Distribution")
    ax.set_xlabel("Realized ETS cost (EUR)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    saved = save_figure(
        fig,
        output_dir=output_dir,
        filename_stem=filename_stem,
    )
    plt.close(fig)
    return saved