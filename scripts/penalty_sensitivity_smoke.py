from __future__ import annotations

from experiments.penalty_sensitivity import run_penalty_sweep_multi_seed
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from reporting.export import save_dataframe


def main() -> None:
    solver = XpressSolver()
    options = SolveOptions(time_limit_s=60, mip_gap=0.01)

    seeds = [7, 21, 42, 99]
    penalty_values = [0, 250, 500, 750, 1000, 1500, 2500, 5000]

    print("Running multi-seed swap penalty sweep...")
    df = run_penalty_sweep_multi_seed(
        solver,
        penalty_name="swap_usd",
        penalty_values=penalty_values,
        seeds=seeds,
        n_containers=5,
        initial_delay_h=48.0,
        alpha=0.5,
        include_fueleu_penalty=False,
        solve_options=options,
    )

    display_cols = [
        "seed",
        "penalty_value",
        "feasible",
        "objective_value",
        "n_delayed",
        "n_skipped",
        "n_swapped",
        "n_omission",
        "n_swap_strategy",
        "route_changed_vs_first",
        "strategy_changed_vs_first",
        "route_signature",
        "strategy_signature",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    print("\nSummary by seed:")
    summary = (
        df.groupby("seed").agg(
            n_rows=("penalty_value", "count"),
            unique_routes=("route_signature", "nunique"),
            unique_strategies=("strategy_signature", "nunique"),
            any_route_change=("route_changed_vs_first", "max"),
            any_strategy_change=("strategy_changed_vs_first", "max"),
        ).reset_index()
    )
    print(summary.to_string(index=False))

    save_dataframe(
        df,
        output_dir="results/tables/penalty_sensitivity_smoke",
        filename_stem="swap_penalty_multi_seed",
        index=False,
    )
    save_dataframe(
        summary,
        output_dir="results/tables/penalty_sensitivity_smoke",
        filename_stem="swap_penalty_multi_seed_summary",
        index=False,
    )

    print("\nSaved results to results/tables/penalty_sensitivity_smoke/")
    print("Penalty sensitivity multi-seed smoke complete.")


if __name__ == "__main__":
    main()