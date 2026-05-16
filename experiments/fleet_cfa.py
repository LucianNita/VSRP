from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.emissions import (
    DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ,
    get_fueleu_limit,
)
from core.entities import FleetInstance, FleetSolution
from core.simulation import (
    FleetUncertaintyScenario,
    RealizedPerformance,
    UncertaintyConfig,
    sample_estimated_delay,
    sample_fleet_uncertainty_scenario,
    simulate_fleet_realized_performance,
)
from experiments.cfa import (
    DEFAULT_THETA_MAX_H,
    apply_theta_to_instance,
    update_theta_additive,
    update_theta_decay,
    update_theta_spsa,
    update_theta_spsa_fleet,
)
from model.base import BaseSolver, SolveOptions
from model.fleet_solver import FleetSolver


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_UNCERTAINTY_CONFIG = UncertaintyConfig()


# =============================================================================
# RESULT CONTAINER
# =============================================================================

@dataclass(slots=True)
class FleetCFAPolicyResult:
    """
    Container for fleet CFA training or testing results.

    Attributes
    ----------
    results_df : pd.DataFrame
        Episode-level result rows with fleet-aggregated metrics.
    theta_history : list[dict[int, float]]
        History of the shared theta vector across episodes.
    per_vessel_df : pd.DataFrame
        Per-vessel realized performance rows for detailed inspection.
    """
    results_df: pd.DataFrame
    theta_history: list[dict[int, float]]
    per_vessel_df: pd.DataFrame


# =============================================================================
# FLEET THETA HELPERS
# =============================================================================

def initialize_fleet_theta(fleet: FleetInstance) -> dict[int, float]:
    """
    Initialize the shared fleet theta vector to zero.

    The theta vector is indexed by port and shared across all vessels.
    Each entry represents the promised-arrival tightening in hours
    applied at that destination port for all vessels.

    Parameters
    ----------
    fleet : FleetInstance
        Used to determine the valid port index range.

    Returns
    -------
    dict[int, float]
        Zero-initialized port-indexed tightening vector.
    """
    return {p: 0.0 for p in range(fleet.n_ports)}


def apply_fleet_theta(
    fleet: FleetInstance,
    theta: dict[int, float],
) -> FleetInstance:
    """
    Return a deep-copied fleet with promised arrivals tightened by theta.

    The same theta vector is applied to every vessel in the fleet.
    Only the optimization instances are tightened. Realized simulation
    is always evaluated against the original untightened fleet.

    Parameters
    ----------
    fleet : FleetInstance
        Original fleet instance.
    theta : dict[int, float]
        Shared port-indexed tightening vector in hours.

    Returns
    -------
    FleetInstance
        Deep-copied fleet with tightened per-vessel instances.
    """
    tightened_instances = [
        apply_theta_to_instance(vessel_instance, theta)
        for vessel_instance in fleet.vessel_instances
    ]

    tightened_fleet = deepcopy(fleet)
    tightened_fleet.vessel_instances = tightened_instances
    return tightened_fleet


# =============================================================================
# FLEET MISSED-BY-PORT AGGREGATION
# =============================================================================

def aggregate_fleet_missed_by_port(
    vessel_performances: list[RealizedPerformance],
) -> dict[int, int]:
    """
    Aggregate missed-container counts by port across all vessels.

    The aggregated signal is used to update the shared theta vector.
    Ports where multiple vessels experience misses receive a stronger
    update signal, which is the key advantage of shared-theta learning
    over independent per-vessel CFA.

    Parameters
    ----------
    vessel_performances : list[RealizedPerformance]
        One RealizedPerformance per vessel.

    Returns
    -------
    dict[int, int]
        Total missed containers per port index across all vessels.
    """
    aggregated: dict[int, int] = {}

    for perf in vessel_performances:
        for port_idx, count in perf.missed_by_port.items():
            aggregated[port_idx] = aggregated.get(port_idx, 0) + count

    return aggregated


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_fleet_cfa(
    base_fleet: FleetInstance,
    solver: BaseSolver,
    *,
    n_episodes: int = 20,
    solve_options: SolveOptions | None = None,
    seed: int = 42,
    update_policy: str = "additive",
    step_size: float = 1.0,
    step_up: float | None = None,
    step_down: float = 0.25,
    perturbation_size: float = 2.0,
    uncertainty_config: UncertaintyConfig | None = None,
    theta_max_h: float = DEFAULT_THETA_MAX_H,
) -> FleetCFAPolicyResult:
    """
    Train a shared fleet CFA theta vector over repeated simulated episodes.

    Per-episode workflow
    --------------------
    1. Sample one estimated delay per vessel for optimization
    2. Sample one realized fleet uncertainty scenario for simulation
    3. Build a tightened fleet instance using the current shared theta
    4. Solve all vessel sub-problems using FleetSolver
    5. Simulate realized performance for all vessels
    6. Aggregate missed-by-port counts across all vessels
    7. Update the shared theta using the chosen policy rule

    The key difference from single-vessel CFA is step 6: the update
    signal is the fleet-wide aggregate of misses, not a single vessel's
    misses. This allows vessels to learn from each other's experience
    at shared ports.

    Parameters
    ----------
    base_fleet : FleetInstance
        Original fleet instance used as the simulation ground truth.
    solver : BaseSolver
        Vessel-level solver backend used inside FleetSolver.
    n_episodes : int, default=20
        Number of training episodes.
    solve_options : SolveOptions | None, default=None
        Solve controls applied to every per-vessel sub-problem.
    seed : int, default=42
        Random seed for reproducibility.
    update_policy : str, default="additive"
        One of: "additive", "decay", "spsa".
    step_size : float, default=1.0
        Step size for the update rule.
    step_up : float | None, default=None
        Upward step size for the decay rule. Defaults to step_size.
    step_down : float, default=0.25
        Downward decay step for the decay rule.
    perturbation_size : float, default=2.0
        Perturbation magnitude for the SPSA rule.
    uncertainty_config : UncertaintyConfig | None, default=None
        Distribution parameters for uncertainty sampling.
    theta_max_h : float, default=DEFAULT_THETA_MAX_H
        Maximum allowed theta value in hours.

    Returns
    -------
    FleetCFAPolicyResult
        Training results including episode-level and per-vessel rows,
        and the full theta history.
    """
    solve_options = solve_options or SolveOptions()
    config = uncertainty_config or DEFAULT_UNCERTAINTY_CONFIG
    rng = np.random.default_rng(seed)
    fleet_solver = FleetSolver(vessel_solver=solver)

    theta = initialize_fleet_theta(base_fleet)
    theta_history: list[dict[int, float]] = [theta.copy()]
    episode_rows: list[dict] = []
    vessel_rows: list[dict] = []

    fueleu_limit = get_fueleu_limit(2026)
    fueleu_compliant = (
        DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ <= fueleu_limit
    )

    for episode in range(1, n_episodes + 1):

        # Sample per-vessel estimated delays for optimization
        est_delays_h = [
            sample_estimated_delay(rng, config)
            for _ in range(base_fleet.n_vessels)
        ]

        # Sample one realized fleet scenario for simulation
        fleet_scenario = sample_fleet_uncertainty_scenario(
            rng, base_fleet, config
        )

        # Build optimization fleet with estimated delays and tightened theta
        opt_fleet = deepcopy(base_fleet)
        for v_idx, vessel_instance in enumerate(opt_fleet.vessel_instances):
            vessel_instance.initial_delay_h = est_delays_h[v_idx]
        opt_fleet = apply_fleet_theta(opt_fleet, theta)

        # Solve all vessels
        fleet_solution = fleet_solver.solve(opt_fleet, options=solve_options)

        # Simulate realized performance on the original base fleet
        vessel_performances = simulate_fleet_realized_performance(
            base_fleet,
            fleet_solution,
            fleet_scenario,
        )

        # Aggregate fleet-wide missed-by-port signal
        fleet_missed_by_port = aggregate_fleet_missed_by_port(
            vessel_performances
        )

        # Update shared theta
        if update_policy == "additive":
            theta = update_theta_additive(
                theta,
                fleet_missed_by_port,
                step_size=step_size,
                theta_max_h=theta_max_h,
            )
        elif update_policy == "decay":
            theta = update_theta_decay(
                theta,
                fleet_missed_by_port,
                step_size=step_size,
                step_up=step_up,
                step_down=step_down,
                theta_max_h=theta_max_h,
            )
        elif update_policy == "spsa":
            theta = update_theta_spsa_fleet(
                theta,
                base_fleet=base_fleet,
                fleet_solver=fleet_solver,
                rng=rng,
                fleet_scenario=fleet_scenario,
                step_size=step_size,
                perturbation_size=perturbation_size,
                theta_max_h=theta_max_h,
                solve_options=solve_options,
            )
        else:
            raise ValueError(
                f"Unknown update_policy: {update_policy!r}. "
                f"Choose from: 'additive', 'decay', 'spsa'"
            )

        theta_history.append(theta.copy())

        # Fleet-level episode row
        total_missed = sum(p.total_missed for p in vessel_performances)
        total_service_cost = sum(
            p.realized_service_cost_usd for p in vessel_performances
        )
        total_delay_cost = sum(
            p.realized_delay_cost_usd for p in vessel_performances
        )
        total_misconnection_cost = sum(
            p.realized_misconnection_cost_usd for p in vessel_performances
        )
        total_fuel_cost = sum(
            p.realized_fuel_cost_usd for p in vessel_performances
        )
        total_ets_cost = sum(
            p.realized_ets_cost_eur for p in vessel_performances
        )

        episode_rows.append({
            "episode": episode,
            "phase": "train",
            "policy": update_policy,
            "objective_comparable": False,
            "n_vessels": base_fleet.n_vessels,
            "fleet_objective_value": fleet_solution.fleet_objective_value,
            "fleet_feasible": fleet_solution.feasible,
            "realized_total_missed": total_missed,
            "realized_service_cost_usd": total_service_cost,
            "realized_delay_cost_usd": total_delay_cost,
            "realized_misconnection_cost_usd": total_misconnection_cost,
            "realized_fuel_cost_usd": total_fuel_cost,
            "realized_ets_cost_eur": total_ets_cost,
            "realized_carbon_price_eur": (
                fleet_scenario.realized_carbon_price_eur
            ),
            "realized_fuel_price_usd": (
                fleet_scenario.realized_fuel_price_usd
            ),
            "realized_fueleu_compliant": fueleu_compliant,
            "theta_mean_h": float(np.mean(list(theta.values()))),
            "theta_max_h": float(
                max(theta.values()) if theta else 0.0
            ),
        })

        # Per-vessel rows for this episode
        for v_idx, (perf, vessel_solution) in enumerate(
            zip(vessel_performances, fleet_solution.vessel_solutions)
        ):
            vessel_rows.append({
                "episode": episode,
                "phase": "train",
                "policy": update_policy,
                "vessel_idx": v_idx + 1,
                "vessel_id": base_fleet.vessel_instances[v_idx].metadata.get(
                    "vessel_id", f"V{v_idx + 1}"
                ),
                "estimated_delay_h": est_delays_h[v_idx],
                "realized_delay_h": (
                    fleet_scenario.vessel_scenarios[v_idx]
                    .realized_initial_delay_h
                ),
                "realized_total_missed": perf.total_missed,
                "realized_service_cost_usd": perf.realized_service_cost_usd,
                "realized_ets_cost_eur": perf.realized_ets_cost_eur,
                "vessel_objective_value": vessel_solution.objective_value,
                "vessel_feasible": vessel_solution.feasible,
                "n_skipped": vessel_solution.n_skipped,
                "n_swapped": vessel_solution.n_swapped,
            })

    return FleetCFAPolicyResult(
        results_df=pd.DataFrame(episode_rows),
        theta_history=theta_history,
        per_vessel_df=pd.DataFrame(vessel_rows),
    )


# =============================================================================
# TESTING LOOP
# =============================================================================

def test_fleet_cfa_policy(
    base_fleet: FleetInstance,
    solver: BaseSolver,
    theta: dict[int, float],
    *,
    n_episodes: int = 50,
    solve_options: SolveOptions | None = None,
    seed: int = 123,
    policy_name: str = "trained_fleet_policy",
    uncertainty_config: UncertaintyConfig | None = None,
) -> FleetCFAPolicyResult:
    """
    Evaluate a fixed shared theta policy over repeated fleet test episodes.

    The theta vector is held fixed throughout. No parameter updates occur.
    Realized performance is evaluated on the original base fleet.

    Parameters
    ----------
    base_fleet : FleetInstance
        Original fleet instance used as the simulation ground truth.
    solver : BaseSolver
        Vessel-level solver backend.
    theta : dict[int, float]
        Fixed shared theta vector to evaluate.
    n_episodes : int, default=50
        Number of test episodes.
    solve_options : SolveOptions | None, default=None
        Solve controls applied to every per-vessel sub-problem.
    seed : int, default=123
        Random seed for reproducibility.
    policy_name : str, default="trained_fleet_policy"
        Label attached to all result rows.
    uncertainty_config : UncertaintyConfig | None, default=None
        Distribution parameters for uncertainty sampling.

    Returns
    -------
    FleetCFAPolicyResult
        Test results including episode-level and per-vessel rows.
        theta_history contains only the fixed theta (no updates).
    """
    solve_options = solve_options or SolveOptions()
    config = uncertainty_config or DEFAULT_UNCERTAINTY_CONFIG
    rng = np.random.default_rng(seed)
    fleet_solver = FleetSolver(vessel_solver=solver)

    theta_history: list[dict[int, float]] = [theta.copy()]
    episode_rows: list[dict] = []
    vessel_rows: list[dict] = []

    fueleu_limit = get_fueleu_limit(2026)
    fueleu_compliant = (
        DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ <= fueleu_limit
    )

    for episode in range(1, n_episodes + 1):

        est_delays_h = [
            sample_estimated_delay(rng, config)
            for _ in range(base_fleet.n_vessels)
        ]

        fleet_scenario = sample_fleet_uncertainty_scenario(
            rng, base_fleet, config
        )

        opt_fleet = deepcopy(base_fleet)
        for v_idx, vessel_instance in enumerate(opt_fleet.vessel_instances):
            vessel_instance.initial_delay_h = est_delays_h[v_idx]
        opt_fleet = apply_fleet_theta(opt_fleet, theta)

        fleet_solution = fleet_solver.solve(opt_fleet, options=solve_options)

        vessel_performances = simulate_fleet_realized_performance(
            base_fleet,
            fleet_solution,
            fleet_scenario,
        )

        total_missed = sum(p.total_missed for p in vessel_performances)
        total_service_cost = sum(
            p.realized_service_cost_usd for p in vessel_performances
        )
        total_delay_cost = sum(
            p.realized_delay_cost_usd for p in vessel_performances
        )
        total_misconnection_cost = sum(
            p.realized_misconnection_cost_usd for p in vessel_performances
        )
        total_fuel_cost = sum(
            p.realized_fuel_cost_usd for p in vessel_performances
        )
        total_ets_cost = sum(
            p.realized_ets_cost_eur for p in vessel_performances
        )

        episode_rows.append({
            "episode": episode,
            "phase": "test",
            "policy": policy_name,
            "objective_comparable": False,
            "n_vessels": base_fleet.n_vessels,
            "fleet_objective_value": fleet_solution.fleet_objective_value,
            "fleet_feasible": fleet_solution.feasible,
            "realized_total_missed": total_missed,
            "realized_service_cost_usd": total_service_cost,
            "realized_delay_cost_usd": total_delay_cost,
            "realized_misconnection_cost_usd": total_misconnection_cost,
            "realized_fuel_cost_usd": total_fuel_cost,
            "realized_ets_cost_eur": total_ets_cost,
            "realized_carbon_price_eur": (
                fleet_scenario.realized_carbon_price_eur
            ),
            "realized_fuel_price_usd": (
                fleet_scenario.realized_fuel_price_usd
            ),
            "realized_fueleu_compliant": fueleu_compliant,
            "theta_mean_h": float(np.mean(list(theta.values()))),
            "theta_max_h": float(
                max(theta.values()) if theta else 0.0
            ),
        })

        for v_idx, (perf, vessel_solution) in enumerate(
            zip(vessel_performances, fleet_solution.vessel_solutions)
        ):
            vessel_rows.append({
                "episode": episode,
                "phase": "test",
                "policy": policy_name,
                "vessel_idx": v_idx + 1,
                "vessel_id": base_fleet.vessel_instances[v_idx].metadata.get(
                    "vessel_id", f"V{v_idx + 1}"
                ),
                "estimated_delay_h": est_delays_h[v_idx],
                "realized_delay_h": (
                    fleet_scenario.vessel_scenarios[v_idx]
                    .realized_initial_delay_h
                ),
                "realized_total_missed": perf.total_missed,
                "realized_service_cost_usd": perf.realized_service_cost_usd,
                "realized_ets_cost_eur": perf.realized_ets_cost_eur,
                "vessel_objective_value": vessel_solution.objective_value,
                "vessel_feasible": vessel_solution.feasible,
                "n_skipped": vessel_solution.n_skipped,
                "n_swapped": vessel_solution.n_swapped,
            })

    return FleetCFAPolicyResult(
        results_df=pd.DataFrame(episode_rows),
        theta_history=theta_history,
        per_vessel_df=pd.DataFrame(vessel_rows),
    )


# =============================================================================
# TAIL RISK SUMMARY
# =============================================================================

def compute_fleet_tail_risk_summary(
    results_df: pd.DataFrame,
    *,
    cost_col: str = "realized_service_cost_usd",
    percentiles: list[float] | None = None,
) -> dict:
    """
    Compute tail-risk statistics for a fleet CFA results table.

    The cost column is the fleet-aggregated realized service cost
    across all vessels in each episode.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from train_fleet_cfa() or test_fleet_cfa_policy().
    cost_col : str, default="realized_service_cost_usd"
        Column on which tail-risk statistics are computed.
    percentiles : list[float] | None, default=None
        Percentiles to compute. Defaults to [50, 75, 90, 95, 99].

    Returns
    -------
    dict
        Tail-risk summary including mean, std, percentiles,
        FuelEU compliance rate, and average realized ETS cost.
    """
    percentiles = percentiles or [50, 75, 90, 95, 99]

    if results_df.empty or cost_col not in results_df.columns:
        return {}

    costs = results_df[cost_col].dropna()
    if costs.empty:
        return {}

    result: dict = {
        "mean": float(costs.mean()),
        "std": float(costs.std()),
        "n_episodes": int(len(costs)),
    }

    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(costs, p))

    p95 = np.percentile(costs, 95)
    tail = costs[costs >= p95]
    if len(tail) > 0:
        result["tail_mean_above_p95"] = float(tail.mean())

    if "realized_fueleu_compliant" in results_df.columns:
        result["fueleu_compliance_rate"] = float(
            results_df["realized_fueleu_compliant"].mean()
        )

    if "realized_ets_cost_eur" in results_df.columns:
        result["avg_realized_ets_cost_eur"] = float(
            results_df["realized_ets_cost_eur"].mean()
        )

    return result