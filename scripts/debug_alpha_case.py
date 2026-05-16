# scripts/debug_alpha_case.py
# =============================================================================
# Debug one alpha-sweep scenario in detail.
#
# Useful for investigating:
#   - route reconstruction
#   - skipped / swapped ports
#   - timeline issues
#   - validation warnings
# =============================================================================

from __future__ import annotations

from experiments.sensitivity import run_alpha_sweep
from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import SolveOptions
from model.xpress_solver import XpressSolver


def main() -> None:
    target_alpha = 0.7
    seed = 42
    n_containers = 5

    print("=" * 88)
    print(f"DEBUG ALPHA CASE — alpha={target_alpha}")
    print("=" * 88)

    # -----------------------------------------------------------------
    # 1. Regenerate the fixed container set used in the sensitivity sweep
    # -----------------------------------------------------------------
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n_containers,
        seed=seed,
    )

    print("\nContainers:")
    for c in containers:
        print(
            f"  {c.id}: "
            f"{BASE_PORTS[c.origin_idx]} -> {BASE_PORTS[c.destination_idx]}  "
            f"promise={c.promised_arrival_h:.1f}h"
        )

    # -----------------------------------------------------------------
    # 2. Build the matching instance
    # -----------------------------------------------------------------
    instance = build_base_instance(
        containers=containers,
        instance_id=f"debug_alpha_{target_alpha:.1f}",
        initial_delay_h=48.0,
        alpha=target_alpha,
        allow_swap=True,
        max_skip=1,
        metadata={"seed": seed, "alpha": target_alpha},
    )

    solver = XpressSolver()
    options = SolveOptions(time_limit_s=30, mip_gap=0.01)

    solution = solver.solve(instance, options=options)

    # -----------------------------------------------------------------
    # 3. Solver summary
    # -----------------------------------------------------------------
    stats = solution.solver_stats

    print("\nSolver summary:")
    print(f"  solver        : {stats.solver_name if stats else 'N/A'}")
    print(f"  status        : {stats.status if stats else 'N/A'}")
    print(f"  feasible      : {solution.feasible}")
    print(f"  optimal       : {solution.optimal}")
    print(
        f"  runtime_s     : {stats.runtime_s:.6f}"
        if stats and stats.runtime_s is not None
        else "  runtime_s     : N/A"
    )
    print(
        f"  mip_gap       : {stats.mip_gap:.6f}"
        if stats and stats.mip_gap is not None
        else "  mip_gap       : N/A"
    )
    print(
        f"  best_bound    : {stats.best_bound:,.6f}"
        if stats and stats.best_bound is not None
        else "  best_bound    : N/A"
    )
    print(
        f"  objective     : {solution.objective_value:,.6f}"
        if solution.objective_value is not None
        else "  objective     : None"
    )

    # -----------------------------------------------------------------
    # 4. Route
    # -----------------------------------------------------------------
    print("\nRoute:")
    if not solution.route_legs:
        print("  <empty>")
    else:
        for i, leg in enumerate(solution.route_legs, start=1):
            swap_tag = " [SWAP]" if leg.is_swap else ""
            print(
                f"  {i:>2d}. "
                f"{instance.port_name(leg.from_port_idx):<5} -> "
                f"{instance.port_name(leg.to_port_idx):<5}  "
                f"speed={leg.speed_label:<6}  "
                f"duration={leg.duration_label}{swap_tag}"
            )

    print("\nSkipped ports:")
    print(" ", [instance.port_name(p) for p in solution.skipped_port_indices])

    print("\nSwapped ports:")
    print(" ", [instance.port_name(p) for p in solution.swapped_port_indices])

    # -----------------------------------------------------------------
    # 5. Timeline
    # -----------------------------------------------------------------
    print("\nTimeline:")
    if not solution.timeline:
        print("  <empty>")
    else:
        print(
            f"  {'#':>2}  {'Port':<5}  {'Planned':>10}  {'Actual':>10}  "
            f"{'Delay':>10}  {'Departure':>10}  {'Status'}"
        )
        print("  " + "-" * 72)
        for i, t in enumerate(solution.timeline, start=1):
            print(
                f"  {i:>2d}  "
                f"{instance.port_name(t.port_idx):<5}  "
                f"{t.planned_arrival_h:>10.2f}  "
                f"{t.actual_arrival_h:>10.2f}  "
                f"{t.delay_h:>10.2f}  "
                f"{t.departure_h:>10.2f}  "
                f"{t.status}"
            )

    # -----------------------------------------------------------------
    # 6. Validation
    # -----------------------------------------------------------------
    print("\nValidation:")
    if solution.validation is None:
        print("  No validation attached")
    else:
        v = solution.validation
        print(f"  overall_valid         : {v.overall_valid}")
        print(f"  route_valid           : {v.route_valid}")
        print(f"  strategy_consistent   : {v.strategy_consistent}")
        print(f"  timeline_monotone     : {v.timeline_monotone}")
        print(f"  max_constraint_violation : {v.max_constraint_violation}")
        print(f"  n_violated_constraints   : {v.n_violated_constraints}")

        if v.route_errors:
            print("\n  Route errors:")
            for msg in v.route_errors:
                print(f"    - {msg}")

        if v.strategy_warnings:
            print("\n  Strategy warnings:")
            for msg in v.strategy_warnings:
                print(f"    - {msg}")

        if v.timeline_warnings:
            print("\n  Timeline warnings:")
            for msg in v.timeline_warnings:
                print(f"    - {msg}")

    # -----------------------------------------------------------------
    # 7. Costs
    # -----------------------------------------------------------------
    print("\nDiagnostics:")
    print(
        f"  objective_recompute_abs_gap : "
        f"{solution.metadata.get('objective_recompute_abs_gap')}"
    )

    cost_breakdown = solution.metadata.get("cost_breakdown")
    if cost_breakdown:
        print("  cost_breakdown:")
        for k, v in cost_breakdown.items():
            print(f"    - {k}: {v:,.6f}")

    # -----------------------------------------------------------------
    # 8. Emissions
    # -----------------------------------------------------------------
    print("\nEmissions:")
    if solution.emissions is None:
        print("  No emissions summary attached")
    else:
        print(f"  total_fuel_t     : {solution.emissions.total_fuel_t:,.6f}")
        print(f"  total_co2_t      : {solution.emissions.total_co2_t:,.6f}")
        print(f"  total_ets_eur    : {solution.emissions.total_ets_eur:,.6f}")
        print(
            f"  avg_ghg_gco2eq_per_mj : "
            f"{solution.emissions.avg_ghg_gco2eq_per_mj:,.6f}"
        )
        print(f"  fueleu_compliant : {solution.emissions.fueleu_compliant}")

    print("\n" + "=" * 88)
    print("DEBUG COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()