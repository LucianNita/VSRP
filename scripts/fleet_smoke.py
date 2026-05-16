from __future__ import annotations

import pandas as pd

from core.fleet_costs import compute_fleet_cost_breakdown
from experiments.fleet_benchmark import FLEET_SCENARIOS, build_canonical_fleet
from model.base import SolveOptions
from model.fleet_solver import FleetSolver
from model.xpress_solver import XpressSolver
from reporting.export import save_dataframe
from reporting.fleet_plots import (
    plot_fleet_per_vessel_breakdown,
    plot_fleet_scenario_comparison,
)


def _per_vessel_rows(
    scenario_name: str,
    fleet,
    fleet_solution,
) -> list[dict]:
    """
    Build a list of per-vessel result dictionaries from a fleet solution.
    """
    costs = compute_fleet_cost_breakdown(fleet, fleet_solution)
    rows = []

    for v_idx, (instance, solution) in enumerate(
        zip(fleet.vessel_instances, fleet_solution.vessel_solutions)
    ):
        stats = solution.solver_stats
        emissions = solution.emissions
        vessel_costs = costs.vessel_breakdowns[v_idx]

        rows.append({
            "scenario": scenario_name,
            "vessel_idx": v_idx + 1,
            "vessel_id": instance.metadata.get("vessel_id", f"V{v_idx + 1}"),
            "instance_id": instance.instance_id,
            "initial_delay_h": instance.initial_delay_h,
            "n_containers": instance.n_containers,
            "feasible": solution.feasible,
            "optimal": solution.optimal,
            "objective_value": solution.objective_value,
            "runtime_s": stats.runtime_s if stats else None,
            "mip_gap": stats.mip_gap if stats else None,
            "best_bound": stats.best_bound if stats else None,
            "n_delayed": solution.n_delayed,
            "n_misconnected": solution.n_misconnected,
            "n_skipped": solution.n_skipped,
            "n_swapped": solution.n_swapped,
            "fuel_cost_usd": vessel_costs.fuel_cost_usd,
            "port_call_cost_usd": vessel_costs.port_call_cost_usd,
            "strategy_penalty_usd": vessel_costs.strategy_penalty_usd,
            "port_penalty_cost_usd": vessel_costs.port_penalty_cost_usd,
            "operational_cost_usd": vessel_costs.operational_cost_usd,
            "service_cost_usd": vessel_costs.service_cost_usd,
            "weighted_objective_usd": vessel_costs.weighted_objective_usd,
            "total_fuel_t": (
                emissions.total_fuel_t if emissions else None
            ),
            "total_co2_t": (
                emissions.total_co2_t if emissions else None
            ),
            "total_ets_eur": (
                emissions.total_ets_eur if emissions else None
            ),
            "total_fueleu_penalty_usd": (
                emissions.total_fueleu_penalty_usd if emissions else None
            ),
            "cii_rating": (
                emissions.cii_rating if emissions else None
            ),
            "fueleu_compliant": (
                emissions.fueleu_compliant if emissions else None
            ),
            "route_valid": (
                solution.validation.route_valid
                if solution.validation else None
            ),
            "strategy_consistent": (
                solution.validation.strategy_consistent
                if solution.validation else None
            ),
            "timeline_monotone": (
                solution.validation.timeline_monotone
                if solution.validation else None
            ),
            "container_valid": (
                solution.validation.container_valid
                if solution.validation else None
            ),
            "skipped_ports_valid": (
                solution.validation.skipped_ports_valid
                if solution.validation else None
            ),
            "max_constraint_violation": (
                solution.validation.max_constraint_violation
                if solution.validation else None
            ),
            "objective_recompute_gap": solution.metadata.get(
                "objective_recompute_abs_gap"
            ),
            "fleet_objective_value": fleet_solution.fleet_objective_value,
        })

    return rows


def _fleet_summary_row(
    scenario_name: str,
    fleet,
    fleet_solution,
    per_vessel_rows: list[dict],
) -> dict:
    """
    Build a single fleet-level summary row from a solved fleet scenario.
    """
    total_co2 = sum(
        r["total_co2_t"]
        for r in per_vessel_rows
        if r["total_co2_t"] is not None
    )
    total_ets = sum(
        r["total_ets_eur"]
        for r in per_vessel_rows
        if r["total_ets_eur"] is not None
    )

    return {
        "scenario": scenario_name,
        "description": FLEET_SCENARIOS[scenario_name]["description"],
        "n_vessels": fleet.n_vessels,
        "total_containers": fleet.total_containers,
        "feasible": fleet_solution.feasible,
        "optimal": fleet_solution.optimal,
        "fleet_objective_value": fleet_solution.fleet_objective_value,
        "avg_runtime_s": fleet_solution.avg_runtime_s,
        "total_runtime_s": fleet_solution.total_runtime_s,
        "total_delayed": fleet_solution.total_delayed,
        "total_misconnected": fleet_solution.total_misconnected,
        "total_skipped": fleet_solution.total_skipped,
        "total_swapped": fleet_solution.total_swapped,
        "total_co2_t": total_co2 if total_co2 else None,
        "total_ets_eur": total_ets if total_ets else None,
        "all_feasible": all(r["feasible"] for r in per_vessel_rows),
        "all_route_valid": all(
            r["route_valid"] for r in per_vessel_rows
            if r["route_valid"] is not None
        ),
        "all_strategy_consistent": all(
            r["strategy_consistent"] for r in per_vessel_rows
            if r["strategy_consistent"] is not None
        ),
    }


def _print_strategy_warnings(
    scenario_name: str,
    fleet,
    fleet_solution,
) -> bool:
    """
    Print strategy consistency warnings for vessels that fail validation.

    Returns True if any warnings were printed.
    """
    any_warnings = False

    for v_idx, (instance, solution) in enumerate(
        zip(fleet.vessel_instances, fleet_solution.vessel_solutions)
    ):
        if (
            solution.validation
            and not solution.validation.strategy_consistent
        ):
            any_warnings = True
            print(
                f"\n  [{scenario_name}] Vessel {v_idx + 1} "
                f"(delay={instance.initial_delay_h}h) "
                f"— strategy warnings:"
            )
            for w in solution.validation.strategy_warnings:
                print(f"    - {w}")

    return any_warnings


def main() -> None:
    print("=" * 80)
    print("FLEET DISRUPTION SCENARIO SMOKE TEST")
    print("=" * 80)

    solver = XpressSolver()
    fleet_solver = FleetSolver(vessel_solver=solver)
    options = SolveOptions(time_limit_s=120, mip_gap=0.001)

    all_vessel_rows: list[dict] = []
    fleet_summary_rows: list[dict] = []
    any_warnings = False

    for scenario_name in FLEET_SCENARIOS:
        print(f"\nRunning {scenario_name}...")
        fleet = build_canonical_fleet(scenario_name)
        fleet_solution = fleet_solver.solve(fleet, options=options)

        vessel_rows = _per_vessel_rows(scenario_name, fleet, fleet_solution)
        all_vessel_rows.extend(vessel_rows)

        fleet_summary_rows.append(
            _fleet_summary_row(
                scenario_name, fleet, fleet_solution, vessel_rows
            )
        )

        if _print_strategy_warnings(scenario_name, fleet, fleet_solution):
            any_warnings = True

    if not any_warnings:
        print("\n  All vessels passed strategy consistency checks.")

    per_vessel_df = pd.DataFrame(all_vessel_rows)
    fleet_df = pd.DataFrame(fleet_summary_rows)

    # ------------------------------------------------------------------
    # Per-vessel results
    # ------------------------------------------------------------------
    display_cols = [
        "scenario", "vessel_idx", "initial_delay_h",
        "feasible", "objective_value", "runtime_s",
        "n_delayed", "n_misconnected", "n_skipped", "n_swapped",
        "total_co2_t", "cii_rating",
        "route_valid", "strategy_consistent",
        "objective_recompute_gap",
    ]
    available = [c for c in display_cols if c in per_vessel_df.columns]
    print("\nPer-Vessel Results:")
    print(per_vessel_df[available].to_string(index=False))

    # ------------------------------------------------------------------
    # Fleet summary
    # ------------------------------------------------------------------
    fleet_display_cols = [
        "scenario", "n_vessels", "total_containers",
        "feasible", "fleet_objective_value",
        "total_delayed", "total_misconnected",
        "total_skipped", "total_swapped",
        "total_co2_t", "total_ets_eur",
        "avg_runtime_s", "total_runtime_s",
        "all_route_valid", "all_strategy_consistent",
    ]
    fleet_available = [
        c for c in fleet_display_cols if c in fleet_df.columns
    ]
    print("\nFleet-Level Summary:")
    print(fleet_df[fleet_available].to_string(index=False))

    # ------------------------------------------------------------------
    # Cost breakdown summary
    # ------------------------------------------------------------------
    cost_cols = [
        "scenario", "vessel_idx",
        "fuel_cost_usd", "port_call_cost_usd",
        "strategy_penalty_usd", "port_penalty_cost_usd",
        "operational_cost_usd", "service_cost_usd",
        "weighted_objective_usd",
    ]
    cost_available = [c for c in cost_cols if c in per_vessel_df.columns]
    print("\nCost Breakdown per Vessel:")
    print(per_vessel_df[cost_available].to_string(index=False))

    # ------------------------------------------------------------------
    # Save tables
    # ------------------------------------------------------------------
    save_dataframe(
        per_vessel_df,
        output_dir="results/tables/fleet_smoke",
        filename_stem="fleet_per_vessel",
        index=False,
    )
    save_dataframe(
        fleet_df,
        output_dir="results/tables/fleet_smoke",
        filename_stem="fleet_summary",
        index=False,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating fleet plots...")

    fleet_plot_df = fleet_df.copy()
    fleet_plot_df["solver_name"] = solver.solver_name

    plot_fleet_scenario_comparison(
        fleet_plot_df,
        output_dir="results/figures/fleet_smoke",
        filename_stem="fleet_scenario_comparison",
    )

    # Per-vessel breakdown plot uses the per-vessel DataFrame.
    plot_fleet_per_vessel_breakdown(
        per_vessel_df,
        output_dir="results/figures/fleet_smoke",
        filename_stem="fleet_per_vessel_breakdown",
    )

    print("\nSaved to:")
    print("  results/tables/fleet_smoke/")
    print("  results/figures/fleet_smoke/")
    print("=" * 80)
    print("FLEET SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()