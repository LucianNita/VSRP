# scripts/benchmark_smoke.py
# =============================================================================
# Benchmark smoke test.
#
# =============================================================================

from __future__ import annotations

from experiments.benchmark import run_benchmark
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from model.highs_solver import HighsSolver
from model.cbc_solver import CBCSolver
from reporting.benchmark_reporting import (
    print_benchmark_summary,
    save_benchmark_outputs,
)
from reporting.benchmark_plots import plot_benchmark_summary


def main() -> None:
    print("=" * 72)
    print("VSRP BENCHMARK SMOKE TEST")
    print("=" * 72)

    print("\nAvailability check:")
    for solver in [XpressSolver(), HighsSolver(), CBCSolver()]:
        print(
            f"  {solver.solver_name:<10} : "
            f"{'available' if solver.is_available else 'NOT AVAILABLE'}"
        )

    raw_df, summary_df = run_benchmark(
        solvers=[XpressSolver(), HighsSolver(), CBCSolver()],
        n_instances=2,
        n_containers=5,
        initial_delay_h=48.0,
        alpha=0.5,
        allow_swap=True,
        max_skip=1,
        include_fueleu_penalty=True,
        solve_options=SolveOptions(time_limit_s=60, mip_gap=0.01),
    )

    print("\nRAW RESULTS (selected columns)")
    display_cols = [
        "solver_name", "instance_id", "feasible", "optimal",
        "objective_value", "runtime_s", "mip_gap", "best_bound",
        "time_to_first_feasible_s", "node_count",
        "n_delayed", "n_skipped", "n_swapped",
        "total_co2_t", "total_ets_eur",
        "route_valid", "strategy_consistent",
        "container_valid", "skipped_ports_valid",
        "max_constraint_violation",
    ]
    available = [c for c in display_cols if c in raw_df.columns]
    print(raw_df[available].to_string(index=False))

    print_benchmark_summary(summary_df)

    # Print stability metrics specifically
    stability_cols = [
        "solver_name",
        "pct_feasible", "pct_optimal",
        "avg_objective", "obj_std",
        "avg_runtime_s", "runtime_std",
        "avg_mip_gap", "gap_std",
        "avg_time_to_first_feasible_s",
    ]
    available_stability = [
        c for c in stability_cols if c in summary_df.columns
    ]
    if available_stability:
        print("\nSTABILITY METRICS")
        print("=" * 72)
        print(summary_df[available_stability].to_string(index=False))

    saved = save_benchmark_outputs(
        raw_df,
        summary_df,
        output_dir="results/tables/benchmark_smoke",
    )

    print("\nSaved files:")
    for group, paths in saved.items():
        for ext, path in paths.items():
            print(f"  {group}.{ext}: {path}")

    fig_paths = plot_benchmark_summary(
        summary_df,
        output_dir="results/figures/benchmark_smoke",
        filename_stem="benchmark_summary",
    )

    if fig_paths:
        print("\nSaved figures:")
        for ext, path in fig_paths.items():
            print(f"  benchmark_plot.{ext}: {path}")

    print("\n" + "=" * 72)
    print("BENCHMARK SMOKE TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()