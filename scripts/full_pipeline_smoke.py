from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.fleet_costs import compute_fleet_cost_breakdown
from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.fleet_instance import build_fleet_from_delays
from data.instance_generator import generate_containers
from experiments.benchmark import run_benchmark
from experiments.cfa import (
    compute_tail_risk_summary,
    initialize_theta,
    test_cfa_policy,
    train_cfa,
)
from experiments.fleet_benchmark import (
    FLEET_SCENARIOS,
    build_canonical_fleet,
    run_fleet_benchmark,
)
from experiments.fleet_cfa import (
    compute_fleet_tail_risk_summary,
    initialize_fleet_theta,
    test_fleet_cfa_policy,
    train_fleet_cfa,
)
from experiments.sensitivity import run_alpha_sweep, run_delay_sweep
from model.base import SolveOptions
from model.cbc_solver import CBCSolver
from model.fleet_solver import FleetSolver
from model.highs_solver import HighsSolver
from model.xpress_solver import XpressSolver
from reporting.export import save_dataframe


# =============================================================================
# CHECK REGISTRY
# =============================================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str = ""
    warning: str = ""


checks: list[CheckResult] = []


def run_check(name: str, fn) -> CheckResult:
    """
    Run one named check function and record pass/fail.
    """
    try:
        warning = fn()
        result = CheckResult(
            name=name,
            passed=True,
            warning=warning or "",
        )
    except Exception as exc:
        result = CheckResult(
            name=name,
            passed=False,
            message=f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()

    checks.append(result)
    status = "PASS" if result.passed else "FAIL"
    suffix = f"  ⚠ {result.warning}" if result.warning else ""
    print(f"  [{status}] {name}{suffix}")
    return result


# =============================================================================
# SHARED FIXTURES
# =============================================================================

SOLVER = XpressSolver()
OPTIONS = SolveOptions(time_limit_s=60, mip_gap=0.01, log_to_console=False)
OUTPUT_DIR = Path("results/tables/full_pipeline_smoke")
FIGURE_DIR = Path("results/figures/full_pipeline_smoke")


def _make_base_instance(seed: int = 42, n: int = 5, delay: float = 48.0):
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n,
        seed=seed,
    )
    return build_base_instance(
        containers=containers,
        instance_id=f"pipeline_seed{seed}",
        initial_delay_h=delay,
        alpha=0.5,
        allow_swap=True,
        max_skip=1,
    )


# =============================================================================
# CHECKS
# =============================================================================

def check_single_vessel_solve():
    instance = _make_base_instance()
    solution = SOLVER.solve(instance, options=OPTIONS)

    assert solution.feasible, "Solution not feasible"
    assert solution.objective_value is not None, "No objective value"
    assert solution.validation is not None, "No validation attached"
    assert solution.validation.overall_valid, (
        f"Validation failed: {solution.validation.route_errors}"
    )
    assert solution.emissions is not None, "No emissions attached"

    gap = solution.metadata.get("objective_recompute_abs_gap", None)
    assert gap is not None, "No recompute gap in metadata"
    assert gap < 1.0, f"Recompute gap too large: {gap}"

    if not solution.optimal:
        return "Xpress returned feasible but not optimal (community license)"


def check_transshipment_solve():
    from data.instance_generator import generate_containers as gc
    containers = gc(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=6,
        seed=123,
        notebook_compatible=True,
        transshipment_probability=0.5,
    )
    instance = build_base_instance(
        containers=containers,
        instance_id="pipeline_transshipment",
        initial_delay_h=48.0,
        alpha=0.5,
    )
    solution = SOLVER.solve(instance, options=OPTIONS)

    assert solution.feasible, "Transshipment solution not feasible"
    assert solution.validation is not None, "No validation"
    assert solution.validation.route_valid, "Route not valid"

    gap = solution.metadata.get("objective_recompute_abs_gap", 0.0)
    assert gap < 1.0, f"Recompute gap too large: {gap}"


def check_all_solvers_agree():
    instance = _make_base_instance(seed=99)
    solvers = [XpressSolver(), HighsSolver(), CBCSolver()]
    objectives = {}

    for solver in solvers:
        if not solver.is_available:
            continue
        sol = solver.solve(instance, options=OPTIONS)
        if sol.feasible and sol.objective_value is not None:
            objectives[solver.solver_name] = sol.objective_value

    if len(objectives) < 2:
        return "Only one solver available — cross-solver check skipped"

    values = list(objectives.values())
    max_diff = max(values) - min(values)
    assert max_diff < 1.0, (
        f"Solver objective disagreement: {objectives}, diff={max_diff:.4f}"
    )


def check_validation_layers():
    from core.validation import validate_solution
    instance = _make_base_instance()
    solution = SOLVER.solve(instance, options=OPTIONS)
    result = validate_solution(instance, solution)

    assert result.route_valid, f"Route invalid: {result.route_errors}"
    assert result.container_valid, (
        f"Container invalid: {result.container_warnings}"
    )
    assert result.skipped_ports_valid, (
        f"Skipped ports invalid: {result.skipped_port_warnings}"
    )
    assert result.max_constraint_violation is not None
    assert result.max_constraint_violation < 1e-6, (
        f"Constraint violation: {result.max_constraint_violation}"
    )


def check_emissions():
    from core.emissions import compute_solution_emissions_summary
    instance = _make_base_instance()
    solution = SOLVER.solve(instance, options=OPTIONS)
    summary = compute_solution_emissions_summary(
        instance=instance,
        solution=solution,
        year=2026,
    )

    assert summary.total_fuel_t is not None and summary.total_fuel_t > 0
    assert summary.total_co2_t is not None and summary.total_co2_t > 0
    assert summary.total_ets_eur is not None and summary.total_ets_eur > 0
    assert summary.cii_rating in {"A", "B", "C", "D", "E"}
    assert summary.fueleu_compliant is not None


def check_cost_recomputation():
    from core.costs import compute_cost_breakdown
    instance = _make_base_instance()
    solution = SOLVER.solve(instance, options=OPTIONS)
    breakdown = compute_cost_breakdown(instance, solution)

    assert breakdown.fuel_cost_usd > 0
    assert breakdown.operational_cost_usd > 0
    assert breakdown.weighted_objective_usd > 0

    gap = abs(
        breakdown.weighted_objective_usd - solution.objective_value
    )
    assert gap < 1.0, f"Cost recompute gap: {gap:.6f}"


def check_alpha_sweep():
    df = run_alpha_sweep(
        SOLVER,
        n_containers=5,
        seed=42,
        initial_delay_h=48.0,
        alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0],
        solve_options=OPTIONS,
    )

    assert not df.empty, "Alpha sweep returned empty DataFrame"
    feasible = df[df["feasible"] == True]
    assert len(feasible) >= 4, (
        f"Too few feasible alpha sweep solutions: {len(feasible)}"
    )

    save_dataframe(
        df,
        output_dir=OUTPUT_DIR,
        filename_stem="alpha_sweep",
        index=False,
    )

    obj_at_0 = df[df["alpha"] == 0.0]["objective_value"].values
    obj_at_1 = df[df["alpha"] == 1.0]["objective_value"].values
    if len(obj_at_0) and len(obj_at_1):
        if obj_at_0[0] is not None and obj_at_1[0] is not None:
            if obj_at_0[0] < obj_at_1[0]:
                return (
                    "alpha=0 objective < alpha=1 objective "
                    "(operationally cheaper at alpha=0 is expected)"
                )


def check_delay_sweep():
    df = run_delay_sweep(
        SOLVER,
        n_containers=5,
        seed=42,
        alpha=0.5,
        delay_values_h=[0, 24, 48, 72, 96],
        solve_options=OPTIONS,
    )

    assert not df.empty, "Delay sweep returned empty DataFrame"
    feasible = df[df["feasible"] == True]
    assert len(feasible) >= 4, (
        f"Too few feasible delay sweep solutions: {len(feasible)}"
    )

    save_dataframe(
        df,
        output_dir=OUTPUT_DIR,
        filename_stem="delay_sweep",
        index=False,
    )


def check_benchmark():
    raw_df, summary_df = run_benchmark(
        solvers=[XpressSolver(), HighsSolver(), CBCSolver()],
        n_instances=2,
        n_containers=5,
        initial_delay_h=48.0,
        alpha=0.5,
        solve_options=OPTIONS,
    )

    assert not raw_df.empty, "Benchmark raw DataFrame is empty"
    assert not summary_df.empty, "Benchmark summary DataFrame is empty"

    xpress_rows = raw_df[raw_df["solver_name"] == "Xpress"]
    assert xpress_rows["feasible"].all(), "Xpress not feasible on all instances"

    save_dataframe(
        raw_df,
        output_dir=OUTPUT_DIR,
        filename_stem="benchmark_raw",
        index=False,
    )
    save_dataframe(
        summary_df,
        output_dir=OUTPUT_DIR,
        filename_stem="benchmark_summary",
        index=False,
    )


def check_fleet_solve():
    fleet = build_canonical_fleet("Case1_Delayed")
    fleet_solver = FleetSolver(vessel_solver=SOLVER)
    fleet_solution = fleet_solver.solve(fleet, options=OPTIONS)

    assert fleet_solution.feasible, "Fleet solution not feasible"
    assert fleet_solution.fleet_objective_value is not None
    assert fleet_solution.n_vessels == 3

    costs = compute_fleet_cost_breakdown(fleet, fleet_solution)
    assert costs.fleet_objective_usd > 0

    gap = abs(
        costs.fleet_objective_usd - fleet_solution.fleet_objective_value
    )
    assert gap < 3.0, f"Fleet recompute gap too large: {gap:.4f}"

    for v_idx, solution in enumerate(fleet_solution.vessel_solutions):
        assert solution.feasible, f"Vessel {v_idx + 1} not feasible"
        if solution.validation:
            assert solution.validation.route_valid, (
                f"Vessel {v_idx + 1} route invalid"
            )
            assert solution.validation.strategy_consistent, (
                f"Vessel {v_idx + 1} strategy inconsistent: "
                f"{solution.validation.strategy_warnings}"
            )


def check_fleet_all_scenarios():
    fleet_solver = FleetSolver(vessel_solver=SOLVER)
    warnings = []

    for scenario_name in FLEET_SCENARIOS:
        fleet = build_canonical_fleet(scenario_name)
        fleet_solution = fleet_solver.solve(fleet, options=OPTIONS)

        assert fleet_solution.feasible, (
            f"{scenario_name}: fleet not feasible"
        )

        for v_idx, solution in enumerate(fleet_solution.vessel_solutions):
            assert solution.feasible, (
                f"{scenario_name} Vessel {v_idx + 1}: not feasible"
            )
            if solution.validation:
                assert solution.validation.route_valid, (
                    f"{scenario_name} Vessel {v_idx + 1}: route invalid"
                )
                if not solution.validation.strategy_consistent:
                    warnings.append(
                        f"{scenario_name} V{v_idx + 1}: "
                        f"strategy_consistent=False"
                    )

    if warnings:
        return "; ".join(warnings)


def check_fleet_benchmark():
    raw_df, summary_df = run_fleet_benchmark(
        solvers=[XpressSolver(), HighsSolver(), CBCSolver()],
        scenarios=["Case1_Delayed"],
        solve_options=OPTIONS,
    )

    assert not raw_df.empty, "Fleet benchmark raw DataFrame is empty"

    xpress_rows = raw_df[raw_df["solver_name"] == "Xpress"]
    assert xpress_rows["feasible"].all(), (
        "Xpress not feasible on fleet benchmark"
    )

    highs_rows = raw_df[raw_df["solver_name"] == "HiGHS"]
    cbc_rows = raw_df[raw_df["solver_name"] == "CBC"]

    if not highs_rows.empty and not xpress_rows.empty:
        xpress_obj = xpress_rows["fleet_objective_value"].values[0]
        highs_obj = highs_rows["fleet_objective_value"].values[0]
        if xpress_obj is not None and highs_obj is not None:
            diff = abs(xpress_obj - highs_obj)
            if diff > 1000.0:
                return (
                    f"Xpress vs HiGHS fleet objective diff={diff:.0f} "
                    f"(may be community license gap)"
                )

    save_dataframe(
        raw_df,
        output_dir=OUTPUT_DIR,
        filename_stem="fleet_benchmark_raw",
        index=False,
    )


def check_single_vessel_cfa():
    instance = _make_base_instance(n=5, delay=48.0)

    baseline_theta = initialize_theta(instance)
    train_result = train_cfa(
        instance,
        SOLVER,
        n_episodes=5,
        solve_options=OPTIONS,
        seed=42,
        update_policy="additive",
        step_size=1.0,
    )

    assert not train_result.results_df.empty, "CFA training returned empty"
    assert len(train_result.theta_history) == 6, (
        "Theta history length mismatch"
    )

    trained_theta = train_result.theta_history[-1]
    test_result = test_cfa_policy(
        instance,
        SOLVER,
        trained_theta,
        n_episodes=5,
        solve_options=OPTIONS,
        seed=123,
    )

    assert not test_result.results_df.empty, "CFA test returned empty"

    tail = compute_tail_risk_summary(test_result.results_df)
    assert "mean" in tail, "Tail risk summary missing mean"
    assert tail["mean"] >= 0, "Negative mean service cost"

    save_dataframe(
        train_result.results_df,
        output_dir=OUTPUT_DIR,
        filename_stem="cfa_train",
        index=False,
    )
    save_dataframe(
        test_result.results_df,
        output_dir=OUTPUT_DIR,
        filename_stem="cfa_test",
        index=False,
    )


def check_fleet_cfa():
    base_fleet = build_canonical_fleet("Case1_Delayed")

    theta = initialize_fleet_theta(base_fleet)
    assert len(theta) == base_fleet.n_ports, "Theta length mismatch"

    train_result = train_fleet_cfa(
        base_fleet,
        SOLVER,
        n_episodes=5,
        solve_options=OPTIONS,
        seed=42,
        update_policy="additive",
        step_size=1.0,
    )

    assert not train_result.results_df.empty, (
        "Fleet CFA training returned empty"
    )
    assert not train_result.per_vessel_df.empty, (
        "Fleet CFA per-vessel DataFrame empty"
    )
    assert len(train_result.theta_history) == 6, (
        "Fleet theta history length mismatch"
    )

    trained_theta = train_result.theta_history[-1]
    test_result = test_fleet_cfa_policy(
        base_fleet,
        SOLVER,
        trained_theta,
        n_episodes=5,
        solve_options=OPTIONS,
        seed=123,
    )

    assert not test_result.results_df.empty, "Fleet CFA test returned empty"

    tail = compute_fleet_tail_risk_summary(test_result.results_df)
    assert "mean" in tail, "Fleet tail risk summary missing mean"
    assert tail["mean"] >= 0, "Negative mean fleet service cost"

    save_dataframe(
        train_result.results_df,
        output_dir=OUTPUT_DIR,
        filename_stem="fleet_cfa_train",
        index=False,
    )
    save_dataframe(
        test_result.results_df,
        output_dir=OUTPUT_DIR,
        filename_stem="fleet_cfa_test",
        index=False,
    )


def check_fueleu_penalty():
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=5,
        seed=42,
    )
    instance = build_base_instance(
        containers=containers,
        instance_id="pipeline_fueleu",
        initial_delay_h=48.0,
        alpha=0.5,
        include_fueleu_penalty=True,
    )
    solution = SOLVER.solve(instance, options=OPTIONS)

    assert solution.feasible, "FuelEU instance not feasible"
    assert solution.emissions is not None, "No emissions on FuelEU instance"
    assert solution.emissions.fueleu_compliant is not None

    gap = solution.metadata.get("objective_recompute_abs_gap", 0.0)
    assert gap < 1.0, f"FuelEU recompute gap: {gap}"


def check_port_penalty():
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=5,
        seed=42,
    )
    lgb_idx = BASE_PORTS.index("LGB")
    instance = build_base_instance(
        containers=containers,
        instance_id="pipeline_port_penalty",
        initial_delay_h=48.0,
        alpha=0.5,
        port_penalties_usd={lgb_idx: 240_000.0},
    )
    solution = SOLVER.solve(instance, options=OPTIONS)

    assert solution.feasible, "Port penalty instance not feasible"
    assert solution.validation is not None
    assert solution.validation.route_valid, "Route invalid with port penalty"
    assert solution.validation.strategy_consistent, (
        f"Strategy inconsistent with port penalty: "
        f"{solution.validation.strategy_warnings}"
    )

    gap = solution.metadata.get("objective_recompute_abs_gap", 0.0)
    assert gap < 1.0, f"Port penalty recompute gap: {gap}"


def check_output_files():
    expected_files = [
        OUTPUT_DIR / "alpha_sweep.csv",
        OUTPUT_DIR / "delay_sweep.csv",
        OUTPUT_DIR / "benchmark_raw.csv",
        OUTPUT_DIR / "benchmark_summary.csv",
        OUTPUT_DIR / "fleet_benchmark_raw.csv",
        OUTPUT_DIR / "cfa_train.csv",
        OUTPUT_DIR / "cfa_test.csv",
        OUTPUT_DIR / "fleet_cfa_train.csv",
        OUTPUT_DIR / "fleet_cfa_test.csv",
    ]
    missing = [str(f) for f in expected_files if not f.exists()]
    assert not missing, f"Missing output files: {missing}"


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 88)
    print("FULL PIPELINE INTEGRATION SMOKE TEST")
    print("=" * 88)
    print(f"\nOutput directory : {OUTPUT_DIR}")
    print(f"Solver           : {SOLVER.solver_name}")
    print(f"Time limit       : {OPTIONS.time_limit_s}s per vessel")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("CORE SOLVER CHECKS")
    print("-" * 50)
    run_check("Single-vessel solve + validation", check_single_vessel_solve)
    run_check("Transshipment misconnection logic", check_transshipment_solve)
    run_check("Cross-solver objective agreement", check_all_solvers_agree)
    run_check("Validation layer (all checks)", check_validation_layers)
    run_check("Emissions computation", check_emissions)
    run_check("Cost recomputation gap", check_cost_recomputation)
    run_check("FuelEU penalty instance", check_fueleu_penalty)
    run_check("Port penalty instance", check_port_penalty)

    print()
    print("EXPERIMENT CHECKS")
    print("-" * 50)
    run_check("Alpha sweep (5 values)", check_alpha_sweep)
    run_check("Delay sweep (5 values)", check_delay_sweep)
    run_check("Multi-solver benchmark (2 instances)", check_benchmark)

    print()
    print("FLEET CHECKS")
    print("-" * 50)
    run_check("Fleet solve — Case1_Delayed", check_fleet_solve)
    run_check("Fleet solve — all 3 scenarios", check_fleet_all_scenarios)
    run_check("Fleet benchmark — all solvers", check_fleet_benchmark)

    print()
    print("CFA CHECKS")
    print("-" * 50)
    run_check("Single-vessel CFA (5 episodes)", check_single_vessel_cfa)
    run_check("Fleet CFA (5 episodes)", check_fleet_cfa)

    print()
    print("OUTPUT FILE CHECKS")
    print("-" * 50)
    run_check("All expected output files exist", check_output_files)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    n_total = len(checks)
    n_passed = sum(1 for c in checks if c.passed)
    n_failed = n_total - n_passed
    n_warnings = sum(1 for c in checks if c.passed and c.warning)

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(f"  Total checks : {n_total}")
    print(f"  Passed       : {n_passed}")
    print(f"  Failed       : {n_failed}")
    print(f"  Warnings     : {n_warnings}")
    print()

    if n_failed > 0:
        print("FAILED CHECKS:")
        for c in checks:
            if not c.passed:
                print(f"  ✗ {c.name}")
                print(f"    {c.message}")
        print()

    if n_warnings > 0:
        print("WARNINGS:")
        for c in checks:
            if c.passed and c.warning:
                print(f"  ⚠ {c.name}")
                print(f"    {c.warning}")
        print()

    if n_failed == 0:
        print("ALL CHECKS PASSED")
    else:
        print(f"{n_failed} CHECK(S) FAILED")

    print("=" * 88)


if __name__ == "__main__":
    main()