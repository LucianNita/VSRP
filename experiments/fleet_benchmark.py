from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from core.entities import FleetBenchmarkRecord
from core.fleet_costs import compute_fleet_cost_breakdown
from data.base_instance import BASE_DISTANCE_MATRIX_NM, BASE_PORTS
from data.fleet_instance import build_fleet_from_delays
from data.instance_generator import generate_containers
from model.base import BaseSolver, SolveOptions
from model.fleet_solver import FleetSolver


# =============================================================================
# CANONICAL FLEET SCENARIOS
# =============================================================================

FLEET_SCENARIOS: dict[str, dict] = {
    "Case1_Delayed": {
        "vessel_delays_h": [36.0, 24.0, 48.0],
        "containers_per_vessel": [3, 4, 3],
        "seeds": [42, 43, 44],
        "port_penalties_usd": {},
        "description": "Multiple delayed vessels, no port penalties",
    },
    "Case2_PortClosure": {
        "vessel_delays_h": [24.0, 36.0, 12.0],
        "containers_per_vessel": [3, 4, 3],
        "seeds": [42, 43, 44],
        "port_penalties_usd": {BASE_PORTS.index("LGB"): 240_000.0},
        "description": "LGB port closure modelled as $240,000 visit penalty",
    },
    "Case3_Congestion": {
        "vessel_delays_h": [12.0, 24.0, 6.0],
        "containers_per_vessel": [3, 4, 3],
        "seeds": [42, 43, 44],
        "port_penalties_usd": {BASE_PORTS.index("DHB"): 120_000.0},
        "description": "DHB congestion modelled as $120,000 visit penalty",
    },
}


# =============================================================================
# FLEET INSTANCE FACTORY
# =============================================================================

def build_canonical_fleet(
    scenario_name: str,
    *,
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
):
    """
    Build one of the three canonical fleet disruption scenarios.

    Parameters
    ----------
    scenario_name : str
        One of the keys in FLEET_SCENARIOS.
    alpha : float, default=0.5
        Objective trade-off weight applied to all vessels.
    allow_swap : bool, default=True
        Whether port swapping is available for all vessels.
    max_skip : int, default=1
        Maximum consecutive port skips for all vessels.
    include_fueleu_penalty : bool, default=False
        Whether the FuelEU penalty proxy is included for all vessels.

    Returns
    -------
    FleetInstance
        Ready-to-solve fleet instance for the requested scenario.

    Raises
    ------
    ValueError
        If scenario_name is not a recognised canonical scenario.
    """
    if scenario_name not in FLEET_SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario_name!r}. "
            f"Choose from: {sorted(FLEET_SCENARIOS)}"
        )

    cfg = FLEET_SCENARIOS[scenario_name]

    containers_per_vessel = [
        generate_containers(
            ports=BASE_PORTS,
            distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
            n=n_cont,
            seed=seed,
        )
        for n_cont, seed in zip(
            cfg["containers_per_vessel"],
            cfg["seeds"],
        )
    ]

    return build_fleet_from_delays(
        vessel_delays_h=cfg["vessel_delays_h"],
        containers_per_vessel=containers_per_vessel,
        fleet_id=scenario_name,
        alpha=alpha,
        allow_swap=allow_swap,
        max_skip=max_skip,
        port_penalties_usd=cfg["port_penalties_usd"],
        include_fueleu_penalty=include_fueleu_penalty,
        metadata={
            "scenario": scenario_name,
            "description": cfg["description"],
        },
    )


# =============================================================================
# RECORD BUILDING
# =============================================================================

def fleet_solution_to_benchmark_record(
    solver: BaseSolver,
    fleet,
    fleet_solution,
    scenario: str,
) -> FleetBenchmarkRecord:
    """
    Convert a FleetSolution into a flat FleetBenchmarkRecord.

    Fields that are only available from the Xpress full-extraction
    backend are left as None for partial-solution backends.

    Parameters
    ----------
    solver : BaseSolver
        The solver backend that produced the solution.
    fleet : FleetInstance
        The fleet instance that was solved.
    fleet_solution : FleetSolution
        The aggregated fleet solution.
    scenario : str
        Scenario name for the record.

    Returns
    -------
    FleetBenchmarkRecord
        Flat record suitable for benchmark table construction.
    """
    costs = compute_fleet_cost_breakdown(fleet, fleet_solution)

    # Check if any vessel used a partial-solution backend
    is_partial = any(
        s.metadata.get("partial_solution_backend", False)
        for s in fleet_solution.vessel_solutions
    )

    total_co2 = (
        sum(
            s.emissions.total_co2_t
            for s in fleet_solution.vessel_solutions
            if s.emissions and s.emissions.total_co2_t is not None
        )
        if not is_partial else None
    )
    total_ets = (
        sum(
            s.emissions.total_ets_eur
            for s in fleet_solution.vessel_solutions
            if s.emissions and s.emissions.total_ets_eur is not None
        )
        if not is_partial else None
    )

    validations = [
        s.validation
        for s in fleet_solution.vessel_solutions
        if s.validation is not None
    ]
    all_routes_valid = (
        all(v.route_valid for v in validations)
        if validations and not is_partial else None
    )
    all_strat_consistent = (
        all(v.strategy_consistent for v in validations)
        if validations and not is_partial else None
    )

    runtimes = [
        s.solver_stats.runtime_s
        for s in fleet_solution.vessel_solutions
        if s.solver_stats and s.solver_stats.runtime_s is not None
    ]

    return FleetBenchmarkRecord(
        solver_name=solver.solver_name,
        fleet_id=fleet.fleet_id,
        n_vessels=fleet.n_vessels,
        scenario=scenario,
        feasible=fleet_solution.feasible,
        optimal=fleet_solution.optimal,
        fleet_objective_value=fleet_solution.fleet_objective_value,
        avg_runtime_s=(
            sum(runtimes) / len(runtimes) if runtimes else None
        ),
        total_runtime_s=sum(runtimes) if runtimes else None,
        total_delayed=(
            fleet_solution.total_delayed if not is_partial else None
        ),
        total_misconnected=(
            fleet_solution.total_misconnected if not is_partial else None
        ),
        total_skipped=(
            fleet_solution.total_skipped if not is_partial else None
        ),
        total_swapped=(
            fleet_solution.total_swapped if not is_partial else None
        ),
        total_co2_t=total_co2,
        total_ets_eur=total_ets,
        all_routes_valid=all_routes_valid,
        all_strategies_consistent=all_strat_consistent,
        metadata={
            "scenario_description": FLEET_SCENARIOS[scenario]["description"],
            "fleet_objective_recompute": costs.fleet_objective_usd,
            "partial_solution_backend": is_partial,
        },
    )


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_fleet_benchmark(
    solvers: list[BaseSolver],
    *,
    scenarios: list[str] | None = None,
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all canonical fleet scenarios across all solver backends.

    For each (solver, scenario) pair:
    1. Build the canonical fleet instance
    2. Solve using FleetSolver wrapping the provided backend
    3. Convert the result to a FleetBenchmarkRecord
    4. Aggregate into raw and summary DataFrames

    Parameters
    ----------
    solvers : list[BaseSolver]
        Solver backends to benchmark.
    scenarios : list[str] | None, default=None
        Subset of FLEET_SCENARIOS to run. Defaults to all three.
    alpha : float, default=0.5
        Objective trade-off weight for all fleet instances.
    allow_swap : bool, default=True
        Whether port swapping is available.
    max_skip : int, default=1
        Maximum consecutive port skips.
    include_fueleu_penalty : bool, default=False
        Whether the FuelEU penalty proxy is included.
    solve_options : SolveOptions | None, default=None
        Solve controls applied to every per-vessel sub-problem.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (raw_df, summary_df) where raw_df has one row per
        (solver, scenario) and summary_df has solver-level aggregates.
    """
    scenarios = scenarios or list(FLEET_SCENARIOS.keys())
    solve_options = solve_options or SolveOptions()
    records: list[FleetBenchmarkRecord] = []

    for scenario_name in scenarios:
        fleet = build_canonical_fleet(
            scenario_name,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            include_fueleu_penalty=include_fueleu_penalty,
        )

        for solver in solvers:
            if not solver.is_available:
                records.append(
                    FleetBenchmarkRecord(
                        solver_name=solver.solver_name,
                        fleet_id=fleet.fleet_id,
                        n_vessels=fleet.n_vessels,
                        scenario=scenario_name,
                        feasible=False,
                        optimal=False,
                        status="UNAVAILABLE",
                        error="Solver backend not available",
                    )
                )
                continue

            fleet_solver = FleetSolver(vessel_solver=solver)
            fleet_solution = fleet_solver.solve(fleet, options=solve_options)

            record = fleet_solution_to_benchmark_record(
                solver, fleet, fleet_solution, scenario_name
            )
            records.append(record)

    raw_df = pd.DataFrame([_record_to_row(r) for r in records])
    summary_df = summarize_fleet_benchmark(raw_df)

    return raw_df, summary_df


# =============================================================================
# SUMMARY AGGREGATION
# =============================================================================

def summarize_fleet_benchmark(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw fleet benchmark rows into solver-level summary metrics.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of run_fleet_benchmark() raw DataFrame.

    Returns
    -------
    pd.DataFrame
        Solver-level summary with feasibility rates, average objectives,
        runtimes, service KPIs, and validation rates.
    """
    if raw_df.empty:
        return pd.DataFrame()

    feasible_df = raw_df[raw_df["feasible"] == True].copy()

    counts = (
        raw_df.groupby("solver_name").agg(
            n_runs=("scenario", "count"),
            n_feasible=("feasible", "sum"),
            n_optimal=("optimal", "sum"),
        ).reset_index()
    )
    counts["pct_feasible"] = counts["n_feasible"] / counts["n_runs"]
    counts["pct_optimal"] = counts["n_optimal"] / counts["n_runs"]

    if feasible_df.empty:
        return counts

    agg_spec: dict[str, tuple] = {}

    for col, alias in [
        ("fleet_objective_value", "avg_fleet_objective"),
        ("avg_runtime_s", "avg_per_vessel_runtime_s"),
        ("total_runtime_s", "avg_total_runtime_s"),
        ("total_delayed", "avg_total_delayed"),
        ("total_misconnected", "avg_total_misconnected"),
        ("total_skipped", "avg_total_skipped"),
        ("total_swapped", "avg_total_swapped"),
        ("total_co2_t", "avg_total_co2_t"),
        ("total_ets_eur", "avg_total_ets_eur"),
    ]:
        if col in feasible_df.columns:
            agg_spec[alias] = (col, "mean")

    for col, alias in [
        ("fleet_objective_value", "obj_std"),
        ("avg_runtime_s", "runtime_std"),
    ]:
        if col in feasible_df.columns:
            agg_spec[alias] = (col, "std")

    for col, alias in [
        ("all_routes_valid", "route_valid_rate"),
        ("all_strategies_consistent", "strategy_consistent_rate"),
    ]:
        if col in feasible_df.columns:
            agg_spec[alias] = (col, "mean")

    if not agg_spec:
        return counts

    feasible_summary = (
        feasible_df.groupby("solver_name").agg(**agg_spec).reset_index()
    )

    return counts.merge(feasible_summary, on="solver_name", how="left")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _record_to_row(record: FleetBenchmarkRecord) -> dict:
    """
    Flatten a FleetBenchmarkRecord into a plain dictionary row.
    """
    row = {
        "solver_name": record.solver_name,
        "fleet_id": record.fleet_id,
        "scenario": record.scenario,
        "n_vessels": record.n_vessels,
        "feasible": record.feasible,
        "optimal": record.optimal,
        "fleet_objective_value": record.fleet_objective_value,
        "avg_runtime_s": record.avg_runtime_s,
        "total_runtime_s": record.total_runtime_s,
        "total_delayed": record.total_delayed,
        "total_misconnected": record.total_misconnected,
        "total_skipped": record.total_skipped,
        "total_swapped": record.total_swapped,
        "total_co2_t": record.total_co2_t,
        "total_ets_eur": record.total_ets_eur,
        "all_routes_valid": record.all_routes_valid,
        "all_strategies_consistent": record.all_strategies_consistent,
        "status": record.status,
        "error": record.error,
    }

    metadata = record.metadata or {}
    row["fleet_objective_recompute"] = metadata.get(
        "fleet_objective_recompute"
    )
    row["scenario_description"] = metadata.get("scenario_description")

    return row