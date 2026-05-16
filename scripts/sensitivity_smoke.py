# scripts/sensitivity_smoke.py
# =============================================================================
# Sensitivity analysis smoke test.
#
# =============================================================================

from __future__ import annotations

from experiments.sensitivity import (
    run_alpha_sweep,
    run_delay_sweep,
    run_ets_price_sweep,
)
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from reporting.sensitivity_plots import (
    plot_alpha_sweep,
    plot_delay_sweep,
    plot_alpha_pareto,
)
from reporting.emissions_plots import (
    plot_green_recovery_tradeoff,
    plot_emissions_vs_delay_penalty,
    plot_strategy_mix_under_ets,
    plot_cii_profile,
)
from reporting.export import save_dataframe


def main() -> None:
    solver = XpressSolver()
    options = SolveOptions(time_limit_s=60, mip_gap=0.01)

    # -----------------------------------------------------------------
    # Alpha sweep
    # -----------------------------------------------------------------
    print("Running alpha sweep...")
    alpha_df = run_alpha_sweep(
        solver,
        n_containers=5,
        seed=42,
        initial_delay_h=48.0,
        include_fueleu_penalty=False,
        solve_options=options,
    )

    # -----------------------------------------------------------------
    # Delay sweep
    # -----------------------------------------------------------------
    print("Running delay sweep...")
    delay_df = run_delay_sweep(
        solver,
        n_containers=5,
        seed=42,
        alpha=0.5,
        include_fueleu_penalty=False,
        solve_options=options,
    )

    # -----------------------------------------------------------------
    # ETS price sweep
    # -----------------------------------------------------------------
    print("Running ETS price sweep...")
    ets_df = run_ets_price_sweep(
        solver,
        n_containers=5,
        seed=42,
        initial_delay_h=48.0,
        alpha=0.5,
        carbon_prices_eur=[25.0, 40.0, 55.0, 65.0, 80.0, 100.0, 130.0],
        solve_options=options,
    )

    # -----------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------
    print("\nALPHA SWEEP (selected columns)")
    display_cols = [
        "alpha", "feasible", "objective_value",
        "operational_cost_usd", "service_cost_usd",
        "total_co2_t", "total_ets_eur", "cii_rating",
        "n_delayed", "n_skipped",
    ]
    available = [c for c in display_cols if c in alpha_df.columns]
    print(alpha_df[available].to_string(index=False))

    print("\nDELAY SWEEP (selected columns)")
    display_cols_delay = [
        "initial_delay_h", "feasible", "objective_value",
        "total_co2_t", "total_ets_eur", "cii_rating",
        "n_delayed", "n_skipped",
    ]
    available_delay = [c for c in display_cols_delay if c in delay_df.columns]
    print(delay_df[available_delay].to_string(index=False))

    print("\nETS PRICE SWEEP (selected columns)")
    display_cols_ets = [
        "carbon_price_eur", "feasible", "objective_value",
        "total_co2_t", "total_ets_eur_at_price",
        "n_skipped", "n_expedited",
    ]
    available_ets = [c for c in display_cols_ets if c in ets_df.columns]
    print(ets_df[available_ets].to_string(index=False))

    # -----------------------------------------------------------------
    # Save tables
    # -----------------------------------------------------------------
    save_dataframe(
        alpha_df,
        output_dir="results/tables/sensitivity_smoke",
        filename_stem="alpha_sweep",
        index=False,
    )
    save_dataframe(
        delay_df,
        output_dir="results/tables/sensitivity_smoke",
        filename_stem="delay_sweep",
        index=False,
    )
    save_dataframe(
        ets_df,
        output_dir="results/tables/sensitivity_smoke",
        filename_stem="ets_sweep",
        index=False,
    )

    # -----------------------------------------------------------------
    # Standard sensitivity plots
    # -----------------------------------------------------------------
    print("\nGenerating sensitivity plots...")

    plot_alpha_sweep(
        alpha_df,
        output_dir="results/figures/sensitivity_smoke",
        filename_stem="alpha_sensitivity",
        exclude_alpha_1=True,
    )
    plot_delay_sweep(
        delay_df,
        output_dir="results/figures/sensitivity_smoke",
        filename_stem="delay_sensitivity",
    )
    plot_alpha_pareto(
        alpha_df,
        output_dir="results/figures/sensitivity_smoke",
        filename_stem="alpha_pareto",
        exclude_alpha_1=True,
    )

    # -----------------------------------------------------------------
    # Emissions / green recovery plots
    # -----------------------------------------------------------------
    print("Generating emissions plots...")

    plot_green_recovery_tradeoff(
        alpha_df,
        output_dir="results/figures/sensitivity_smoke",
        filename_stem="green_recovery_tradeoff",
        exclude_alpha_1=True,
    )

    plot_emissions_vs_delay_penalty(
        delay_df,
        output_dir="results/figures/sensitivity_smoke",
        filename_stem="emissions_vs_delay_penalty",
    )

    if not ets_df.empty:
        plot_strategy_mix_under_ets(
            ets_df,
            output_dir="results/figures/sensitivity_smoke",
            filename_stem="strategy_mix_ets",
        )

    plot_cii_profile(
        delay_df,
        output_dir="results/figures/sensitivity_smoke",
        filename_stem="cii_profile",
    )

    print("\nAll outputs saved to results/")
    print("Sensitivity smoke test complete.")


if __name__ == "__main__":
    main()