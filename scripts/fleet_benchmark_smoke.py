from __future__ import annotations

from experiments.fleet_benchmark import run_fleet_benchmark
from model.base import SolveOptions
from model.cbc_solver import CBCSolver
from model.highs_solver import HighsSolver
from model.xpress_solver import XpressSolver
from reporting.benchmark_reporting import print_benchmark_summary
from reporting.export import save_dataframe
from reporting.fleet_plots import (
    plot_fleet_benchmark_summary,
    plot_fleet_scenario_comparison,
)


def main() -> None:
    print("=" * 80)
    print("FLEET BENCHMARK SMOKE TEST")
    print("=" * 80)

    solvers = [XpressSolver(), HighsSolver(), CBCSolver()]

    print("\nSolver availability:")
    for s in solvers:
        status = "available" if s.is_available else "NOT AVAILABLE"
        print(f"  {s.solver_name:<10} : {status}")

    options = SolveOptions(time_limit_s=120, mip_gap=0.001)

    print("\nRunning fleet benchmark (3 scenarios x 3 solvers)...")
    raw_df, summary_df = run_fleet_benchmark(
        solvers=solvers,
        solve_options=options,
    )

    # ------------------------------------------------------------------
    # Raw results
    # ------------------------------------------------------------------
    display_cols = [
        "solver_name", "scenario", "n_vessels",
        "feasible", "optimal",
        "fleet_objective_value",
        "avg_runtime_s", "total_runtime_s",
        "total_delayed", "total_misconnected",
        "total_skipped", "total_swapped",
        "total_co2_t", "total_ets_eur",
        "all_routes_valid", "all_strategies_consistent",
    ]
    available = [c for c in display_cols if c in raw_df.columns]
    print("\nRAW FLEET BENCHMARK RESULTS:")
    print(raw_df[available].to_string(index=False))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nFLEET BENCHMARK SUMMARY:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # ------------------------------------------------------------------
    # Stability metrics
    # ------------------------------------------------------------------
    stability_cols = [
        "solver_name",
        "pct_feasible", "pct_optimal",
        "avg_fleet_objective", "obj_std",
        "avg_per_vessel_runtime_s", "runtime_std",
        "avg_total_delayed", "avg_total_skipped",
        "route_valid_rate", "strategy_consistent_rate",
    ]
    stability_available = [
        c for c in stability_cols if c in summary_df.columns
    ]
    if stability_available:
        print("\nSTABILITY METRICS:")
        print(summary_df[stability_available].to_string(index=False))

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------
    save_dataframe(
        raw_df,
        output_dir="results/tables/fleet_benchmark_smoke",
        filename_stem="fleet_benchmark_raw",
        index=False,
    )
    save_dataframe(
        summary_df,
        output_dir="results/tables/fleet_benchmark_smoke",
        filename_stem="fleet_benchmark_summary",
        index=False,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating fleet benchmark plots...")

    plot_fleet_benchmark_summary(
        summary_df,
        output_dir="results/figures/fleet_benchmark_smoke",
        filename_stem="fleet_benchmark_summary",
    )

    plot_fleet_scenario_comparison(
        raw_df,
        output_dir="results/figures/fleet_benchmark_smoke",
        filename_stem="fleet_scenario_comparison",
    )

    print("\nSaved to:")
    print("  results/tables/fleet_benchmark_smoke/")
    print("  results/figures/fleet_benchmark_smoke/")
    print("=" * 80)
    print("FLEET BENCHMARK SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()