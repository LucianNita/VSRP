from __future__ import annotations

from core.simulation import UncertaintyConfig
from experiments.stochastic_ets import (
    evaluate_stochastic_ets_exposure,
    summarize_stochastic_ets,
)
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from reporting.export import save_dataframe


def main() -> None:
    solver = XpressSolver()
    options = SolveOptions(time_limit_s=60, mip_gap=0.01)

    uncertainty = UncertaintyConfig(
        carbon_price_mean_eur=65.0,
        carbon_price_std_eur=15.0,
        carbon_price_min_eur=30.0,
        carbon_price_max_eur=130.0,
    )

    print("Running stochastic ETS exposure smoke...")
    df = evaluate_stochastic_ets_exposure(
        solver,
        n_containers=5,
        seed=42,
        initial_delay_h=48.0,
        alpha=0.5,
        include_fueleu_penalty=False,
        solve_options=options,
        uncertainty_config=uncertainty,
        n_scenarios=50,
        scenario_seed=123,
    )

    display_cols = [
        "scenario_idx",
        "realized_carbon_price_eur",
        "realized_ets_cost_eur",
        "objective_value",
        "n_skipped",
        "n_swapped",
        "n_delayed",
        "route_signature",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].head(10).to_string(index=False))

    summary = summarize_stochastic_ets(df)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    save_dataframe(
        df,
        output_dir="results/tables/stochastic_ets_smoke",
        filename_stem="stochastic_ets_scenarios",
        index=False,
    )

    print("\nSaved results to results/tables/stochastic_ets_smoke/")
    print("Stochastic ETS smoke complete.")


if __name__ == "__main__":
    main()