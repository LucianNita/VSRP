# scripts/smoke_test.py
# =============================================================================
# Minimal end-to-end smoke test for the refactored VSRP codebase.
#
# =============================================================================

from __future__ import annotations

from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import SolveOptions
from model.xpress_solver import XpressSolver


def main() -> None:
    print("=" * 72)
    print("VSRP REFACTORED CODEBASE — SMOKE TEST")
    print("=" * 72)

    # -----------------------------------------------------------------
    # 1. Generate containers (notebook-compatible penalties)
    # -----------------------------------------------------------------
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=5,
        seed=42,
        promised_slack_h=24.0,
        notebook_compatible=True,
    )

    print(f"\nGenerated {len(containers)} containers:")
    for c in containers:
        trans_str = (
            f" via {BASE_PORTS[c.transshipment_port_indices[0]]}"
            if c.transshipment_port_indices else ""
        )
        print(
            f"  {c.id}: {BASE_PORTS[c.origin_idx]} -> "
            f"{BASE_PORTS[c.destination_idx]}{trans_str}  "
            f"promise={c.promised_arrival_h:.1f}h  "
            f"delay_pen={c.penalty_delay:,.0f}  "
            f"misconnect_pen={c.penalty_misconnect:,.0f}"
        )
        if c.connecting_service_deadline_h is not None:
            print(
                f"    connecting_deadline={c.connecting_service_deadline_h:.1f}h"
            )

    # -----------------------------------------------------------------
    # 2. Build canonical instance
    # -----------------------------------------------------------------
    instance = build_base_instance(
        containers=containers,
        instance_id="smoke_test_001",
        initial_delay_h=48.0,
        alpha=0.5,
        allow_swap=True,
        swap_ordering_vars_enabled=True,
        max_skip=1,
        fuel_price_usd_per_tonne=600.0,
        include_fueleu_penalty=False,
    )

    print(f"\nInstance built:")
    print(f"  instance_id              : {instance.instance_id}")
    print(f"  ports                    : {instance.n_ports}")
    print(f"  containers               : {instance.n_containers}")
    print(f"  initial delay            : {instance.initial_delay_h:.1f}h")
    print(f"  alpha                    : {instance.alpha:.2f}")
    print(f"  allow_swap               : {instance.allow_swap}")
    print(f"  swap_ordering_vars       : {instance.swap_ordering_vars_enabled}")
    print(f"  max_skip                 : {instance.max_skip}")
    print(f"  fuel_price_usd_per_tonne : {instance.fuel_price_usd_per_tonne:.0f}")
    print(f"  include_fueleu_penalty   : {instance.include_fueleu_penalty}")

    # -----------------------------------------------------------------
    # 3. Solve with Xpress
    # -----------------------------------------------------------------
    solver = XpressSolver()
    options = SolveOptions(
        time_limit_s=60,
        mip_gap=0.01,
        log_to_console=False,
        random_seed=42,
    )

    print("\nSolving with Xpress...")
    solution = solver.solve(instance, options=options)

    # -----------------------------------------------------------------
    # 4. Solver summary
    # -----------------------------------------------------------------
    stats = solution.solver_stats
    print("\nSolver summary:")
    print(f"  solver                   : {stats.solver_name if stats else 'N/A'}")
    print(f"  status                   : {stats.status if stats else 'N/A'}")
    print(f"  feasible                 : {solution.feasible}")
    print(f"  optimal                  : {solution.optimal}")

    if stats and stats.runtime_s is not None:
        print(f"  runtime_s                : {stats.runtime_s:.4f}")
    else:
        print(f"  runtime_s                : N/A")

    if solution.objective_value is not None:
        print(f"  objective                : {solution.objective_value:,.4f}")
    else:
        print(f"  objective                : None")

    if stats and stats.mip_gap is not None:
        print(f"  mip_gap                  : {stats.mip_gap:.6f}")
    else:
        print(f"  mip_gap                  : N/A")

    if stats and stats.best_bound is not None:
        print(f"  best_bound               : {stats.best_bound:,.4f}")
    else:
        print(f"  best_bound               : N/A")

    if stats and stats.node_count is not None:
        print(f"  node_count               : {stats.node_count}")
    else:
        print(f"  node_count               : N/A")

    if stats and stats.time_to_first_feasible_s is not None:
        print(
            f"  time_to_first_feasible   : "
            f"{stats.time_to_first_feasible_s:.4f}s"
        )
    else:
        print(f"  time_to_first_feasible   : N/A")
    
    if stats and stats.message:
        print(f"  error_message            : {stats.message}")

    # -----------------------------------------------------------------
    # 5. Route
    # -----------------------------------------------------------------
    print("\nRoute:")
    if not solution.route_legs:
        print("  <empty>")
    else:
        for leg in solution.route_legs:
            swap_tag = ""
            if leg.is_swap:
                swap_tag = f" [SWAP group={leg.swap_group_id}]"
            print(
                f"  {instance.port_name(leg.from_port_idx):<5} -> "
                f"{instance.port_name(leg.to_port_idx):<5}  "
                f"{leg.speed_label:<10}  {leg.duration_label}{swap_tag}"
            )

    # -----------------------------------------------------------------
    # 6. Strategy summary
    # -----------------------------------------------------------------
    print("\nStrategies:")
    if not solution.strategy_decisions:
        print("  <none reported>")
    else:
        for s in solution.strategy_decisions:
            print(
                f"  {instance.port_name(s.port_idx):<5} : {s.strategy}"
            )

    print("\nSkipped ports:")
    if solution.skipped_port_indices:
        print(
            " ",
            [instance.port_name(p) for p in solution.skipped_port_indices],
        )
    else:
        print("  []")

    print("\nSwapped ports:")
    if solution.swapped_port_indices:
        print(
            " ",
            [instance.port_name(p) for p in solution.swapped_port_indices],
        )
    else:
        print("  []")

    # -----------------------------------------------------------------
    # 7. Container outcomes
    # -----------------------------------------------------------------
    print("\nContainer outcomes:")
    if not solution.container_outcomes:
        print("  <none>")
    else:
        for c_id, outcome in solution.container_outcomes.items():
            flags = []
            if outcome.delayed:
                flags.append("DELAYED")
            if outcome.misconnected:
                flags.append("MISCONNECTED")
            flag_text = ", ".join(flags) if flags else "OK"

            trans_str = ""
            if outcome.transshipment_port_indices:
                trans_str = (
                    f" via "
                    f"{[instance.port_name(p) for p in outcome.transshipment_port_indices]}"
                )

            print(
                f"  {c_id}: "
                f"{instance.port_name(outcome.origin_idx)} -> "
                f"{instance.port_name(outcome.destination_idx)}"
                f"{trans_str}  [{flag_text}]"
            )

    # -----------------------------------------------------------------
    # 8. Validation summary
    # -----------------------------------------------------------------
    print("\nValidation:")
    if solution.validation is None:
        print("  No validation result attached")
    else:
        v = solution.validation
        print(f"  overall_valid            : {v.overall_valid}")
        print(f"  route_valid              : {v.route_valid}")
        print(f"  strategy_consistent      : {v.strategy_consistent}")
        print(f"  timeline_monotone        : {v.timeline_monotone}")
        print(f"  container_valid          : {v.container_valid}")
        print(f"  skipped_ports_valid      : {v.skipped_ports_valid}")
        print(
            f"  max_constraint_violation : "
            f"{v.max_constraint_violation}"
        )
        print(
            f"  n_violated_constraints   : "
            f"{v.n_violated_constraints}"
        )

        if v.route_errors:
            print("\n  Route / structure issues:")
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

        if v.container_warnings:
            print("\n  Container warnings:")
            for msg in v.container_warnings:
                print(f"    - {msg}")

        if v.skipped_port_warnings:
            print("\n  Skipped port warnings:")
            for msg in v.skipped_port_warnings:
                print(f"    - {msg}")

    # -----------------------------------------------------------------
    # 9. Emissions summary
    # -----------------------------------------------------------------
    print("\nEmissions:")
    if solution.emissions is None:
        print("  No emissions summary attached")
    else:
        em = solution.emissions
        print(
            f"  total_fuel_t             : "
            f"{em.total_fuel_t:.4f}" if em.total_fuel_t is not None
            else "  total_fuel_t             : N/A"
        )
        print(
            f"  total_co2_t              : "
            f"{em.total_co2_t:.4f}" if em.total_co2_t is not None
            else "  total_co2_t              : N/A"
        )
        print(
            f"  total_ets_eur            : "
            f"{em.total_ets_eur:.2f}" if em.total_ets_eur is not None
            else "  total_ets_eur            : N/A"
        )
        print(
            f"  total_ets_usd            : "
            f"{em.total_ets_usd:.2f}" if em.total_ets_usd is not None
            else "  total_ets_usd            : N/A"
        )
        print(
            f"  total_fueleu_penalty_usd : "
            f"{em.total_fueleu_penalty_usd:.2f}"
            if em.total_fueleu_penalty_usd is not None
            else "  total_fueleu_penalty_usd : N/A"
        )
        print(
            f"  avg_ghg_gco2eq_per_mj    : "
            f"{em.avg_ghg_gco2eq_per_mj:.4f}"
            if em.avg_ghg_gco2eq_per_mj is not None
            else "  avg_ghg_gco2eq_per_mj    : N/A"
        )
        print(
            f"  fueleu_compliant         : {em.fueleu_compliant}"
        )
        print(
            f"  fueleu_limit             : "
            f"{em.fueleu_limit_gco2eq_per_mj:.4f}"
            if em.fueleu_limit_gco2eq_per_mj is not None
            else "  fueleu_limit             : N/A"
        )
        print(f"  cii_rating               : {em.cii_rating}")
        print(
            f"  attained_cii             : "
            f"{em.attained_cii:.6f}" if em.attained_cii is not None
            else "  attained_cii             : N/A"
        )
        print(
            f"  required_cii             : "
            f"{em.required_cii:.6f}" if em.required_cii is not None
            else "  required_cii             : N/A"
        )

    # -----------------------------------------------------------------
    # 10. Cost breakdown
    # -----------------------------------------------------------------
    print("\nCost breakdown:")
    cost_breakdown = solution.metadata.get("cost_breakdown")
    if cost_breakdown:
        for k, v in cost_breakdown.items():
            print(f"  {k:<35} : {v:,.4f}")
    else:
        print("  N/A")

    obj_gap = solution.metadata.get("objective_recompute_abs_gap")
    if obj_gap is not None:
        print(f"\n  objective_recompute_abs_gap : {obj_gap:.6f}")

    print("\n" + "=" * 72)
    print("SMOKE TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()