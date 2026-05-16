# =============================================================================
# Penalty sensitivity experiments for the refactored VSRP codebase.
#
# Purpose
# -------
# This module studies how route structure, strategy mix, costs, and
# emissions respond to changes in tactical recovery penalty parameters.
# =============================================================================
from __future__ import annotations

import pandas as pd

from core.costs import compute_cost_breakdown
from core.entities import PenaltyProfile
from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import BaseSolver, SolveOptions


# Valid operational penalty fields that may be swept in this experiment.
VALID_PENALTY_NAMES = {
    "speed_up_usd",
    "expedited_port_usd",
    "omission_usd",
    "swap_usd",
}


def run_penalty_sweep(
    solver: BaseSolver,
    *,
    penalty_name: str,
    penalty_values: list[float],
    n_containers: int = 5,
    seed: int = 42,
    initial_delay_h: float = 48.0,
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
) -> pd.DataFrame:
    """
    Sweep one operational penalty parameter while holding the others fixed.

    Supported penalty names
    -----------------------
    - `speed_up_usd`
    - `expedited_port_usd`
    - `omission_usd`
    - `swap_usd`

    Notes
    -----
    For a fixed seed, the same generated container set is reused across
    all penalty values. This ensures the sweep isolates the effect of
    the selected penalty parameter rather than changing the demand set.

    Returns
    -------
    pd.DataFrame
        One row per penalty value.
    """
    if penalty_name not in VALID_PENALTY_NAMES:
        raise ValueError(
            f"Unknown penalty_name={penalty_name!r}. "
            f"Choose from {sorted(VALID_PENALTY_NAMES)}"
        )

    solve_options = solve_options or SolveOptions()

    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n_containers,
        seed=seed,
    )

    # Build one base instance so the default penalty profile can be reused
    # and modified one field at a time.
    base_instance = build_base_instance(
        containers=containers,
        instance_id="penalty_base",
        initial_delay_h=initial_delay_h,
        alpha=alpha,
        allow_swap=allow_swap,
        max_skip=max_skip,
        include_fueleu_penalty=include_fueleu_penalty,
        metadata={"seed": seed},
    )

    rows: list[dict] = []

    for penalty_value in penalty_values:
        base_penalties = base_instance.penalties
        penalty_kwargs = {
            "speed_up_usd": base_penalties.speed_up_usd,
            "expedited_port_usd": base_penalties.expedited_port_usd,
            "omission_usd": base_penalties.omission_usd,
            "swap_usd": base_penalties.swap_usd,
        }
        penalty_kwargs[penalty_name] = float(penalty_value)

        penalties = PenaltyProfile(**penalty_kwargs)

        instance = build_base_instance(
            containers=containers,
            instance_id=f"{penalty_name}_{penalty_value:.0f}",
            initial_delay_h=initial_delay_h,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            penalties=penalties,
            include_fueleu_penalty=include_fueleu_penalty,
            metadata={
                "seed": seed,
                "penalty_name": penalty_name,
                "penalty_value": penalty_value,
            },
        )

        solution = solver.solve(instance, options=solve_options)
        rows.append(
            _solution_to_penalty_row(
                solution=solution,
                instance=instance,
                penalty_name=penalty_name,
                penalty_value=float(penalty_value),
            )
        )

    return pd.DataFrame(rows)


def _solution_to_penalty_row(
    *,
    solution,
    instance,
    penalty_name: str,
    penalty_value: float,
) -> dict:
    """
    Convert one solved penalty-sweep instance into a flat row.

    The output includes:
    - optimization status and objective information
    - cost decomposition
    - service and emissions KPIs
    - validation metrics
    - route and strategy signatures for structural-change detection
    """
    stats = solution.solver_stats
    validation = solution.validation
    emissions = solution.emissions
    costs = compute_cost_breakdown(instance, solution)

    strategies = solution.strategy_decisions
    n_speed_up = sum(1 for s in strategies if s.strategy == "SPEED_UP")
    n_expedited = sum(1 for s in strategies if s.strategy == "EXPEDITED_PORT")
    n_omission = sum(1 for s in strategies if s.strategy == "PORT_OMISSION")
    n_swap_strategy = sum(1 for s in strategies if s.strategy == "PORT_SWAP")

    route_signature = (
        "->".join(
            [str(solution.route_legs[0].from_port_idx)] +
            [str(leg.to_port_idx) for leg in solution.route_legs]
        )
        if solution.route_legs
        else ""
    )

    return {
        "instance_id": solution.instance_id,
        "penalty_name": penalty_name,
        "penalty_value": penalty_value,
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
        "n_speed_up": n_speed_up,
        "n_expedited": n_expedited,
        "n_omission": n_omission,
        "n_swap_strategy": n_swap_strategy,
        "total_fuel_t": emissions.total_fuel_t if emissions else None,
        "total_co2_t": emissions.total_co2_t if emissions else None,
        "total_ets_eur": emissions.total_ets_eur if emissions else None,
        "total_fueleu_penalty_usd": (
            emissions.total_fueleu_penalty_usd if emissions else None
        ),
        "cii_rating": emissions.cii_rating if emissions else None,
        "route_valid": validation.route_valid if validation else None,
        "strategy_consistent": (
            validation.strategy_consistent if validation else None
        ),
        "timeline_monotone": (
            validation.timeline_monotone if validation else None
        ),
        "container_valid": validation.container_valid if validation else None,
        "skipped_ports_valid": (
            validation.skipped_ports_valid if validation else None
        ),
        "max_constraint_violation": (
            validation.max_constraint_violation if validation else None
        ),
        "objective_recompute_abs_gap": solution.metadata.get(
            "objective_recompute_abs_gap"
        ),
        "seed": instance.metadata.get("seed"),
        "route_signature": route_signature,
        "strategy_signature": "|".join(
            sorted(
                f"{s.port_idx}:{s.strategy}"
                for s in solution.strategy_decisions
            )
        ),
    }


def run_penalty_sweep_multi_seed(
    solver: BaseSolver,
    *,
    penalty_name: str,
    penalty_values: list[float],
    seeds: list[int],
    n_containers: int = 5,
    initial_delay_h: float = 48.0,
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
) -> pd.DataFrame:
    """
    Run the same penalty sweep across multiple random seeds.

    Notes
    -----
    - The same penalty grid is used for every seed.
    - Each seed generates a different container set.
    - Route and strategy change diagnostics are computed relative to the
      first penalty value within each seed.

    Returns
    -------
    pd.DataFrame
        Concatenated sweep results with one row per `(seed, penalty value)`.
    """
    all_dfs: list[pd.DataFrame] = []

    for seed in seeds:
        df_seed = run_penalty_sweep(
            solver,
            penalty_name=penalty_name,
            penalty_values=penalty_values,
            n_containers=n_containers,
            seed=seed,
            initial_delay_h=initial_delay_h,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            include_fueleu_penalty=include_fueleu_penalty,
            solve_options=solve_options,
        ).copy()

        if not df_seed.empty:
            base_route = df_seed["route_signature"].iloc[0]
            base_strategy = df_seed["strategy_signature"].iloc[0]

            df_seed["route_changed_vs_first"] = (
                df_seed["route_signature"] != base_route
            )
            df_seed["strategy_changed_vs_first"] = (
                df_seed["strategy_signature"] != base_strategy
            )

        all_dfs.append(df_seed)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def summarize_penalty_sweep_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize structural variation within each seed's penalty sweep.

    The summary reports:
    - number of rows in the seed sweep
    - number of distinct route signatures
    - number of distinct strategy signatures
    - whether any route change occurred
    - whether any strategy change occurred
    - whether the seed was completely flat across the tested penalty range
    """
    if df.empty:
        return pd.DataFrame()

    if "seed" not in df.columns:
        return pd.DataFrame()

    summary = (
        df.groupby("seed").agg(
            n_rows=("penalty_value", "count"),
            unique_routes=("route_signature", "nunique"),
            unique_strategies=("strategy_signature", "nunique"),
            any_route_change=("route_changed_vs_first", "max"),
            any_strategy_change=("strategy_changed_vs_first", "max"),
        ).reset_index()
    )

    summary["flat_seed"] = (
        (summary["unique_routes"] == 1)
        & (summary["unique_strategies"] == 1)
    )

    return summary