# =============================================================================
# Sensitivity analysis for the refactored VSRP codebase.
#
# Purpose
# -------
# This module runs deterministic parameter sweeps on fixed generated
# container sets and converts the resulting solutions into tidy rows for
# comparison, export, and plotting.
#
# Main sweeps
# -----------
# - alpha sweep: trade-off between operational and service cost
# - delay sweep: effect of larger initial disruption delays
# - ETS price sweep: post-solve carbon-price exposure analysis
#
# Architectural role
# ------------------
# This file sits between:
# - the optimization layer (`model/`)
# - the reporting layer (`reporting/`)
#
# It provides structured comparative-static experiment outputs.
# =============================================================================

from __future__ import annotations

import pandas as pd

from core.costs import compute_cost_breakdown
from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import BaseSolver, SolveOptions


DEFAULT_ALPHA_RANGE: list[float] = [round(x * 0.1, 1) for x in range(11)]
DEFAULT_DELAY_RANGE_H: list[int] = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]
DEFAULT_ETS_CARBON_PRICES_EUR: list[float] = [
    25.0, 40.0, 55.0, 65.0, 80.0, 100.0, 130.0, 160.0
]


def run_alpha_sweep(
    solver: BaseSolver,
    *,
    n_containers: int = 5,
    seed: int = 42,
    initial_delay_h: float = 48.0,
    alpha_values: list[float] | None = None,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
) -> pd.DataFrame:
    """
    Run a sweep over objective weight values on a fixed generated instance.

    The same container set is reused across all alpha values so that the
    sweep isolates the effect of the objective trade-off parameter rather
    than changing the demand realization.

    Returns
    -------
    pd.DataFrame
        One row per alpha value.
    """
    alpha_values = alpha_values or DEFAULT_ALPHA_RANGE
    solve_options = solve_options or SolveOptions()

    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n_containers,
        seed=seed,
    )

    rows: list[dict] = []

    for alpha in alpha_values:
        instance = build_base_instance(
            containers=containers,
            instance_id=f"alpha_{alpha:.1f}",
            initial_delay_h=initial_delay_h,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            include_fueleu_penalty=include_fueleu_penalty,
            metadata={"seed": seed, "alpha": alpha},
        )

        solution = solver.solve(instance, options=solve_options)
        row = _solution_to_sensitivity_row(
            solution=solution,
            instance=instance,
            sweep_type="alpha",
            sweep_value=alpha,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def run_delay_sweep(
    solver: BaseSolver,
    *,
    n_containers: int = 5,
    seed: int = 42,
    alpha: float = 0.5,
    delay_values_h: list[int] | None = None,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
) -> pd.DataFrame:
    """
    Run a sweep over initial disruption delay values.

    The same generated container set is reused across all delay values so
    that the experiment isolates the effect of disruption severity.

    Returns
    -------
    pd.DataFrame
        One row per initial-delay value.
    """
    delay_values_h = delay_values_h or DEFAULT_DELAY_RANGE_H
    solve_options = solve_options or SolveOptions()

    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n_containers,
        seed=seed,
    )

    rows: list[dict] = []

    for delay_h in delay_values_h:
        instance = build_base_instance(
            containers=containers,
            instance_id=f"delay_{delay_h:03d}",
            initial_delay_h=float(delay_h),
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            include_fueleu_penalty=include_fueleu_penalty,
            metadata={"seed": seed, "delay_h": delay_h},
        )

        solution = solver.solve(instance, options=solve_options)
        row = _solution_to_sensitivity_row(
            solution=solution,
            instance=instance,
            sweep_type="delay",
            sweep_value=float(delay_h),
        )
        rows.append(row)

    return pd.DataFrame(rows)


def run_ets_price_sweep(
    solver: BaseSolver,
    *,
    n_containers: int = 5,
    seed: int = 42,
    initial_delay_h: float = 48.0,
    alpha: float = 0.5,
    carbon_prices_eur: list[float] | None = None,
    allow_swap: bool = True,
    max_skip: int = 1,
    solve_options: SolveOptions | None = None,
) -> pd.DataFrame:
    """
    Run an ETS exposure sweep over post-solve carbon prices on a fixed
    solved instance.

    Important interpretation
    ------------------------
    The current formulation does not directly internalize ETS carbon
    price into the optimization objective. The route is solved under the
    existing operational/service objective, and ETS cost is then
    recomputed at alternative carbon prices.

    This experiment therefore measures carbon-price exposure of the
    selected recovery plan rather than a fully endogenous carbon-price
    response.

    Returns
    -------
    pd.DataFrame
        One row per ETS carbon-price scenario.
    """
    carbon_prices_eur = carbon_prices_eur or DEFAULT_ETS_CARBON_PRICES_EUR
    solve_options = solve_options or SolveOptions()

    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n_containers,
        seed=seed,
    )

    rows: list[dict] = []

    for carbon_price in carbon_prices_eur:
        instance = build_base_instance(
            containers=containers,
            instance_id=f"ets_{carbon_price:.0f}",
            initial_delay_h=initial_delay_h,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            include_fueleu_penalty=True,
            metadata={
                "seed": seed,
                "carbon_price_eur": carbon_price,
            },
        )

        solution = solver.solve(instance, options=solve_options)

        # Recompute ETS cost at the post-solve carbon price while
        # keeping the selected route fixed.
        from core.emissions import (
            compute_solution_leg_emissions,
            ets_cost_eur as _ets_cost_eur,
        )

        leg_records = compute_solution_leg_emissions(
            instance=instance,
            solution=solution,
            year=2026,
        )
        total_co2_t = sum(r.co2_tonnes for r in leg_records)
        total_ets_eur_at_price = _ets_cost_eur(
            co2_tonnes=total_co2_t,
            year=2026,
            carbon_price_eur_per_tco2=carbon_price,
        )

        strategies = solution.strategy_decisions
        n_expedited = sum(
            1 for s in strategies if s.strategy == "EXPEDITED_PORT"
        )
        n_speed_up = sum(
            1 for s in strategies if s.strategy == "SPEED_UP"
        )

        row = _solution_to_sensitivity_row(
            solution=solution,
            instance=instance,
            sweep_type="ets_price",
            sweep_value=carbon_price,
        )
        row["carbon_price_eur"] = carbon_price
        row["total_ets_eur_at_price"] = total_ets_eur_at_price
        row["n_expedited"] = n_expedited
        row["n_speed_up"] = n_speed_up
        rows.append(row)

    return pd.DataFrame(rows)


def _solution_to_sensitivity_row(
    *,
    solution,
    instance,
    sweep_type: str,
    sweep_value: float,
) -> dict:
    """
    Convert one solved instance into a flat sensitivity-analysis row.

    The output row combines:
    - solve status and core optimization metrics
    - cost decomposition
    - service KPIs
    - emissions and regulatory metrics
    - selected validation outputs
    - experiment metadata such as seed and sweep value

    This flat structure is convenient for:
    - export to CSV/Excel
    - summary tables
    - plotting modules
    """
    stats = solution.solver_stats
    validation = solution.validation
    emissions = solution.emissions
    costs = compute_cost_breakdown(instance, solution)

    return {
        "instance_id": solution.instance_id,
        "sweep_type": sweep_type,
        "sweep_value": sweep_value,
        "alpha": instance.alpha,
        "initial_delay_h": instance.initial_delay_h,
        "solver_name": stats.solver_name if stats else None,
        "status": stats.status if stats else None,
        "feasible": solution.feasible,
        "optimal": solution.optimal,
        "runtime_s": stats.runtime_s if stats else None,
        "mip_gap": stats.mip_gap if stats else None,
        "best_bound": stats.best_bound if stats else None,
        "objective_value": solution.objective_value,
        "fuel_cost_usd": costs.fuel_cost_usd,
        "port_call_cost_usd": costs.port_call_cost_usd,
        "strategy_penalty_usd": costs.strategy_penalty_usd,
        "port_penalty_cost_usd": costs.port_penalty_cost_usd,
        "fueleu_penalty_usd": costs.fueleu_penalty_usd,
        "delay_cost_usd": costs.delay_cost_usd,
        "misconnection_cost_usd": costs.misconnection_cost_usd,
        "operational_cost_usd": costs.operational_cost_usd,
        "service_cost_usd": costs.service_cost_usd,
        "recomputed_objective_usd": costs.weighted_objective_usd,
        "n_delayed": solution.n_delayed,
        "n_misconnected": solution.n_misconnected,
        "n_skipped": solution.n_skipped,
        "n_swapped": solution.n_swapped,
        "total_fuel_t": emissions.total_fuel_t if emissions else None,
        "total_co2_t": emissions.total_co2_t if emissions else None,
        "total_ets_eur": emissions.total_ets_eur if emissions else None,
        "total_ets_usd": emissions.total_ets_usd if emissions else None,
        "total_fueleu_penalty_usd": (
            emissions.total_fueleu_penalty_usd if emissions else None
        ),
        "avg_ghg_gco2eq_per_mj": (
            emissions.avg_ghg_gco2eq_per_mj if emissions else None
        ),
        "fueleu_compliant": emissions.fueleu_compliant if emissions else None,
        "cii_rating": emissions.cii_rating if emissions else None,
        "attained_cii": emissions.attained_cii if emissions else None,
        "required_cii": emissions.required_cii if emissions else None,
        "route_valid": validation.route_valid if validation else None,
        "strategy_consistent": (
            validation.strategy_consistent if validation else None
        ),
        "timeline_monotone": (
            validation.timeline_monotone if validation else None
        ),
        "objective_recompute_abs_gap": solution.metadata.get(
            "objective_recompute_abs_gap"
        ),
        "seed": instance.metadata.get("seed"),
    }