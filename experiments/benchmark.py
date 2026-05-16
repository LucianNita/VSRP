# =============================================================================
# Solver benchmarking for the refactored VSRP codebase.
#
# Purpose
# -------
# This module runs multiple solver backends on common randomly generated
# instances and aggregates the results into benchmark-ready tables.
#
# Main responsibilities
# ---------------------
# - convert canonical solver outputs into flat benchmark records
# - run multiple solvers on multiple instances
# - compute solver-level summary statistics
# - expose outputs in DataFrame form for reporting and plotting
#
# Architectural role
# ------------------
# This module sits between:
# - the optimization layer (`model/`)
# - the reporting layer (`reporting/`)
#
# It turns one-off solver runs into comparable benchmark evidence.
# =============================================================================

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from core.entities import BenchmarkRecord
from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import BaseSolver, SolveOptions


# =============================================================================
# 1. RECORD BUILDING
# =============================================================================

def solution_to_benchmark_record(
    solver: BaseSolver,
    solution,
) -> BenchmarkRecord:
    """
    Convert a canonical solver output into a flat benchmark record.

    Notes
    -----
    Some solver backends currently return only partial canonical solutions.
    In those cases, route-level, validation, and emissions KPIs are left
    as `None` so that the benchmark table remains structurally consistent
    while reflecting backend capability differences honestly.
    """
    stats = solution.solver_stats
    validation = solution.validation
    emissions = solution.emissions
    metadata = dict(solution.metadata)

    partial_backend = metadata.get("partial_solution_backend", False)

    return BenchmarkRecord(
        solver_name=solver.solver_name,
        instance_id=solution.instance_id,
        available=solver.is_available,
        feasible=solution.feasible,
        optimal=solution.optimal,
        objective_value=solution.objective_value,
        runtime_s=stats.runtime_s if stats else None,
        mip_gap=stats.mip_gap if stats else None,
        best_bound=stats.best_bound if stats else None,
        time_to_first_feasible_s=(
            stats.time_to_first_feasible_s if stats else None
        ),
        node_count=stats.node_count if stats else None,
        iteration_count=stats.iteration_count if stats else None,
        n_delayed=None if partial_backend else solution.n_delayed,
        n_misconnected=None if partial_backend else solution.n_misconnected,
        n_skipped=None if partial_backend else solution.n_skipped,
        n_swapped=None if partial_backend else solution.n_swapped,
        total_co2_t=None if partial_backend else (
            emissions.total_co2_t if emissions else None
        ),
        total_ets_eur=None if partial_backend else (
            emissions.total_ets_eur if emissions else None
        ),
        route_valid=None if partial_backend else (
            validation.route_valid if validation else None
        ),
        strategy_consistent=None if partial_backend else (
            validation.strategy_consistent if validation else None
        ),
        timeline_monotone=None if partial_backend else (
            validation.timeline_monotone if validation else None
        ),
        container_valid=None if partial_backend else (
            validation.container_valid if validation else None
        ),
        skipped_ports_valid=None if partial_backend else (
            validation.skipped_ports_valid if validation else None
        ),
        max_constraint_violation=None if partial_backend else (
            validation.max_constraint_violation if validation else None
        ),
        n_violated_constraints=None if partial_backend else (
            validation.n_violated_constraints if validation else None
        ),
        status=stats.status if stats else None,
        raw_status_code=stats.raw_status_code if stats else None,
        error=stats.message if stats else None,
        metadata=metadata,
    )


# =============================================================================
# 2. BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    solvers: list[BaseSolver],
    *,
    n_instances: int = 3,
    n_containers: int = 5,
    initial_delay_h: float = 48.0,
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one or more solver backends on multiple random instances.

    Workflow
    --------
    For each benchmark instance:
    1. generate a reproducible container set
    2. build a canonical base instance
    3. run each requested solver
    4. convert each result into a `BenchmarkRecord`
    5. aggregate all rows into raw and summary DataFrames

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        `(raw_df, summary_df)` where:
        - `raw_df` contains one row per solver-instance run
        - `summary_df` contains solver-level aggregate metrics
    """
    solve_options = solve_options or SolveOptions()
    all_records: list[BenchmarkRecord] = []

    for inst_idx in range(n_instances):
        seed = 42 + inst_idx * 7

        containers = generate_containers(
            ports=BASE_PORTS,
            distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
            n=n_containers,
            seed=seed,
        )

        instance = build_base_instance(
            containers=containers,
            instance_id=f"benchmark_inst_{inst_idx + 1:03d}",
            initial_delay_h=initial_delay_h,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            include_fueleu_penalty=include_fueleu_penalty,
            metadata={"seed": seed},
        )

        for solver in solvers:
            if not solver.is_available:
                all_records.append(
                    BenchmarkRecord(
                        solver_name=solver.solver_name,
                        instance_id=instance.instance_id,
                        available=False,
                        feasible=False,
                        optimal=False,
                        status="UNAVAILABLE",
                        error="Solver backend not available",
                        metadata={"seed": seed},
                    )
                )
                continue

            solution = solver.solve(instance, options=solve_options)
            record = solution_to_benchmark_record(solver, solution)
            record.metadata = {**record.metadata, "seed": seed}
            all_records.append(record)

    raw_df = pd.DataFrame(
        [_benchmark_record_to_row(r) for r in all_records]
    )
    summary_df = summarize_benchmark(raw_df)

    return raw_df, summary_df


# =============================================================================
# 3. SUMMARY AGGREGATION
# =============================================================================

def summarize_benchmark(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw benchmark rows into solver-level summary metrics.

    Summary categories
    ------------------
    - coverage metrics:
      number of runs, available runs, feasible runs, optimal runs
    - rate metrics:
      percentage feasible, percentage optimal
    - core optimization metrics:
      objective, runtime, gap, best bound, node count, iteration count
    - stability metrics:
      objective standard deviation, runtime standard deviation, gap standard deviation
    - route / service / emissions KPIs:
      when available from full canonical solution backends
    - validation rates:
      when available from full canonical solution backends

    Notes
    -----
    The function is intentionally defensive against missing columns
    because partial-solution backends do not populate every KPI field.
    """
    if raw_df.empty:
        return pd.DataFrame()

    counts = (
        raw_df.groupby("solver_name", dropna=False).agg(
            n_runs=("instance_id", "count"),
            n_available=("available", "sum"),
            n_feasible=("feasible", "sum"),
            n_optimal=("optimal", "sum"),
        ).reset_index()
    )

    counts["pct_feasible"] = counts["n_feasible"] / counts["n_runs"]
    counts["pct_optimal"] = counts["n_optimal"] / counts["n_runs"]

    feasible_df = raw_df[raw_df["feasible"] == True].copy()

    if feasible_df.empty:
        return counts

    def _safe_agg(col: str) -> str | None:
        """
        Return the column name if it exists in the feasible subset,
        otherwise return None.
        """
        return col if col in feasible_df.columns else None

    agg_spec: dict[str, tuple] = {}

    # Core optimization metrics
    for col, alias in [
        ("objective_value", "avg_objective"),
        ("runtime_s", "avg_runtime_s"),
        ("mip_gap", "avg_mip_gap"),
        ("best_bound", "avg_best_bound"),
        ("node_count", "avg_node_count"),
        ("iteration_count", "avg_iteration_count"),
        ("time_to_first_feasible_s", "avg_time_to_first_feasible_s"),
    ]:
        if _safe_agg(col):
            agg_spec[alias] = (col, "mean")

    # Stability metrics
    for col, alias in [
        ("objective_value", "obj_std"),
        ("mip_gap", "gap_std"),
        ("runtime_s", "runtime_std"),
    ]:
        if _safe_agg(col):
            agg_spec[alias] = (col, "std")

    # Service / emissions / diagnostic metrics
    for col, alias in [
        ("n_delayed", "avg_n_delayed"),
        ("n_misconnected", "avg_n_misconnected"),
        ("n_skipped", "avg_n_skipped"),
        ("n_swapped", "avg_n_swapped"),
        ("total_co2_t", "avg_total_co2_t"),
        ("total_ets_eur", "avg_total_ets_eur"),
        ("fueleu_penalty_usd", "avg_fueleu_penalty_usd"),
        ("operational_cost_usd", "avg_operational_cost_usd"),
        ("service_cost_usd", "avg_service_cost_usd"),
        ("objective_recompute_abs_gap", "avg_objective_recompute_abs_gap"),
    ]:
        if _safe_agg(col):
            agg_spec[alias] = (col, "mean")

    # Validation rates
    for col, alias in [
        ("route_valid", "route_valid_rate"),
        ("strategy_consistent", "strategy_consistent_rate"),
        ("timeline_monotone", "timeline_monotone_rate"),
        ("container_valid", "container_valid_rate"),
        ("skipped_ports_valid", "skipped_ports_valid_rate"),
    ]:
        if _safe_agg(col):
            agg_spec[alias] = (col, "mean")

    if not agg_spec:
        return counts

    feasible_summary = (
        feasible_df.groupby("solver_name", dropna=False).agg(**agg_spec).reset_index()
    )

    summary = counts.merge(feasible_summary, on="solver_name", how="left")
    return summary


# =============================================================================
# 4. INTERNAL HELPERS
# =============================================================================

def _benchmark_record_to_row(record: BenchmarkRecord) -> dict:
    """
    Flatten a `BenchmarkRecord` into a plain dictionary row.

    This helper also promotes selected metadata and cost-breakdown fields
    into top-level columns so that downstream reporting modules do not
    need to unpack nested metadata manually.
    """
    row = asdict(record)

    metadata = row.get("metadata") or {}
    cost_breakdown = metadata.get("cost_breakdown") or {}

    row["seed"] = metadata.get("seed")
    row["objective_recompute_abs_gap"] = metadata.get(
        "objective_recompute_abs_gap"
    )

    row["partial_solution_backend"] = metadata.get("partial_solution_backend")
    row["export_based_backend"] = metadata.get("export_based_backend")
    row["objective_extracted"] = metadata.get("objective_extracted")
    row["best_bound_extracted"] = metadata.get("best_bound_extracted")
    row["mip_gap_extracted"] = metadata.get("mip_gap_extracted")

    row["fuel_cost_usd"] = cost_breakdown.get("fuel_cost_usd")
    row["port_call_cost_usd"] = cost_breakdown.get("port_call_cost_usd")
    row["strategy_penalty_usd"] = cost_breakdown.get("strategy_penalty_usd")
    row["port_penalty_cost_usd"] = cost_breakdown.get("port_penalty_cost_usd")
    row["fueleu_penalty_usd"] = cost_breakdown.get("fueleu_penalty_usd")
    row["delay_cost_usd"] = cost_breakdown.get("delay_cost_usd")
    row["misconnection_cost_usd"] = cost_breakdown.get("misconnection_cost_usd")
    row["operational_cost_usd"] = cost_breakdown.get("operational_cost_usd")
    row["service_cost_usd"] = cost_breakdown.get("service_cost_usd")
    row["recomputed_objective_usd"] = cost_breakdown.get(
        "weighted_objective_usd"
    )

    row.pop("metadata", None)
    return row