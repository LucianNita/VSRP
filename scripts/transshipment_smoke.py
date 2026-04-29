from __future__ import annotations

from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from model.highs_solver import HighsSolver
from model.cbc_solver import CBCSolver


def run_solver(name, solver, instance, options):
    print(f"\n--- {name} ---")
    if not solver.is_available:
        print("Not available")
        return

    sol = solver.solve(instance, options=options)
    stats = sol.solver_stats

    print(f"status       : {stats.status if stats else 'N/A'}")
    print(f"feasible     : {sol.feasible}")
    print(f"optimal      : {sol.optimal}")
    print(f"objective    : {sol.objective_value}")
    if sol.solver_stats and sol.solver_stats.message:
        print(f"error_message: {sol.solver_stats.message}")
    print(f"n_delayed    : {sol.n_delayed}")
    print(f"n_misconn    : {sol.n_misconnected}")

    if sol.validation is not None:
        v = sol.validation
        print(f"overall_valid          : {v.overall_valid}")
        print(f"route_valid            : {v.route_valid}")
        print(f"strategy_consistent    : {v.strategy_consistent}")
        print(f"timeline_monotone      : {v.timeline_monotone}")
        print(f"container_valid        : {v.container_valid}")
        print(f"skipped_ports_valid    : {v.skipped_ports_valid}")
        print(f"max_constraint_violation: {v.max_constraint_violation}")
        print(f"n_violated_constraints : {v.n_violated_constraints}")

        if v.route_errors:
            print("route_errors:")
            for msg in v.route_errors:
                print("  -", msg)

        if v.strategy_warnings:
            print("strategy_warnings:")
            for msg in v.strategy_warnings:
                print("  -", msg)

        if v.timeline_warnings:
            print("timeline_warnings:")
            for msg in v.timeline_warnings:
                print("  -", msg)

        if v.container_warnings:
            print("container_warnings:")
            for msg in v.container_warnings:
                print("  -", msg)

        if v.skipped_port_warnings:
            print("skipped_port_warnings:")
            for msg in v.skipped_port_warnings:
                print("  -", msg)

    if sol.metadata.get("objective_recompute_abs_gap") is not None:
        print(
            "recompute_gap:",
            sol.metadata.get("objective_recompute_abs_gap")
        )

    if sol.container_outcomes:
        print("container outcomes:")
        for cid, out in sol.container_outcomes.items():
            print(
                f"  {cid}: delayed={out.delayed}, "
                f"misconnected={out.misconnected}, "
                f"trans={out.transshipment_port_indices}"
            )
    print("route:")
    for leg in sol.route_legs:
        print(
            leg.from_port_idx,
            "->",
            leg.to_port_idx,
            leg.speed_label,
            leg.duration_label,
            "swap" if leg.is_swap else ""
        )

    print("timeline:")
    for t in sol.timeline:
        print(
            t.port_idx,
            t.planned_arrival_h,
            t.actual_arrival_h,
            t.delay_h,
            t.departure_h,
            t.status
        )


def main():
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=6,
        seed=123,
        notebook_compatible=True,
        transshipment_probability=0.5,
    )

    print("Generated containers:")
    for c in containers:
        print(
            f"{c.id}: {BASE_PORTS[c.origin_idx]} -> {BASE_PORTS[c.destination_idx]} "
            f"trans={c.transshipment_port_indices} "
            f"deadline={c.connecting_service_deadline_h}"
        )

    instance = build_base_instance(
        containers=containers,
        instance_id="transshipment_smoke",
        initial_delay_h=48.0,
        alpha=0.5,
        allow_swap=True,
        swap_ordering_vars_enabled=True,
        max_skip=1,
        include_fueleu_penalty=False,
    )

    options = SolveOptions(time_limit_s=60, mip_gap=0.01, log_to_console=False)

    run_solver("Xpress", XpressSolver(), instance, options)
    run_solver("HiGHS", HighsSolver(), instance, options)
    run_solver("CBC", CBCSolver(), instance, options)


if __name__ == "__main__":
    main()