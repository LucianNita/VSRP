# =============================================================================
# Cost Function Approximation (CFA) for the refactored VSRP codebase.
#
# Purpose
# -------
# This module implements a policy-learning framework in which a parameter
# vector `theta` modifies the optimization instance, the resulting plan is
# simulated under realized uncertainty, and `theta` is then updated based
# on observed performance.
#
# Core idea
# ---------
# The optimization model is solved on an estimated state, but performance
# is evaluated on a realized state. The parameter vector `theta` tightens
# promised arrivals by destination and therefore biases the optimization
# model toward more conservative recovery behavior.
#
# Implemented policy updates
# --------------------------
# - additive
# - decay
# - SPSA (Simultaneous Perturbation Stochastic Approximation)
#
# Architectural role
# ------------------
# This file is the main bridge between:
# - deterministic optimization (`model/`)
# - realized simulation under uncertainty (`core/simulation.py`)
# - policy evaluation and learning
# =============================================================================

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.emissions import (
    DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ,
    get_fueleu_limit,
)
from core.entities import Container, VSRPInstance
from core.simulation import (
    RealizedPerformance,
    UncertaintyConfig,
    UncertaintyScenario,
    sample_estimated_delay,
    sample_uncertainty_scenario,
    simulate_realized_performance,
)
from model.base import BaseSolver, SolveOptions


# =============================================================================
# 1. DEFAULT PARAMETERS
# =============================================================================

DEFAULT_UNCERTAINTY_CONFIG = UncertaintyConfig()
DEFAULT_THETA_MAX_H: float = 30.0


# =============================================================================
# 2. POLICY RESULT CONTAINERS
# =============================================================================

@dataclass(slots=True)
class CFAPolicyResult:
    """
    Container for CFA training or testing results.

    Attributes
    ----------
    results_df : pd.DataFrame
        Episode-level result rows.
    theta_history : list[dict[int, float]]
        History of theta vectors across training or testing.
    """
    results_df: pd.DataFrame
    theta_history: list[dict[int, float]]


# =============================================================================
# 3. THETA HELPERS
# =============================================================================

def initialize_theta(instance: VSRPInstance) -> dict[int, float]:
    """
    Initialize destination-indexed tightening parameters to zero.

    Each port index receives its own theta value, interpreted as an
    amount of promised-arrival tightening in hours.
    """
    return {p: 0.0 for p in range(instance.n_ports)}


def apply_theta_to_instance(
    instance: VSRPInstance,
    theta: dict[int, float],
) -> VSRPInstance:
    """
    Return a deep-copied instance with promised arrivals tightened by theta.

    Important modeling choice
    -------------------------
    - Only the optimization instance is tightened.
    - Realized simulation is always evaluated against the original
      untightened instance and original promised arrivals.

    This asymmetry is intentional and is the essence of the CFA method:
    the policy modifies the optimization problem in order to improve
    realized out-of-sample performance.
    """
    inst_copy = deepcopy(instance)

    tightened_containers: list[Container] = []
    for c in inst_copy.containers:
        tighten_h = theta.get(c.destination_idx, 0.0)
        tightened_containers.append(
            Container(
                id=c.id,
                origin_idx=c.origin_idx,
                destination_idx=c.destination_idx,
                promised_arrival_h=max(
                    0.0,
                    c.promised_arrival_h - tighten_h
                ),
                penalty_delay=c.penalty_delay,
                penalty_misconnect=c.penalty_misconnect,
                quantity_teu=c.quantity_teu,
                transshipment_port_indices=list(
                    c.transshipment_port_indices
                ),
                priority=c.priority,
                connecting_service_deadline_h=(
                    c.connecting_service_deadline_h
                ),
            )
        )

    inst_copy.containers = tightened_containers
    return inst_copy


# =============================================================================
# 4. UPDATE RULES
# =============================================================================

def update_theta_additive(
    theta: dict[int, float],
    missed_by_port: dict[int, int],
    *,
    step_size: float,
    theta_max_h: float = DEFAULT_THETA_MAX_H,
) -> dict[int, float]:
    """
    Apply an additive update rule.

    Rule
    ----
    $$
    \\theta_p \\leftarrow \\theta_p + \\text{step\\_size} \\cdot \\text{misses}_p
    $$

    The updated values are clipped to:
    $$
    [0, \\theta_{\\max}]
    $$
    """
    updated = theta.copy()
    for p in updated:
        updated[p] = min(
            theta_max_h,
            max(0.0, updated[p] + step_size * missed_by_port.get(p, 0)),
        )
    return updated


def update_theta_decay(
    theta: dict[int, float],
    missed_by_port: dict[int, int],
    *,
    step_size: float = 1.0,
    step_up: float | None = None,
    step_down: float = 0.25,
    theta_max_h: float = DEFAULT_THETA_MAX_H,
) -> dict[int, float]:
    """
    Apply a two-sided decay update rule.

    Behavior
    --------
    - increase theta where misses occur
    - decrease theta slightly where no misses occur

    Notes
    -----
    `step_up` overrides `step_size` when provided. Otherwise,
    `step_size` is used as the upward adjustment.
    """
    effective_step_up = step_up if step_up is not None else step_size

    updated = theta.copy()
    for p in updated:
        misses = missed_by_port.get(p, 0)
        if misses > 0:
            updated[p] = min(
                theta_max_h,
                updated[p] + effective_step_up * misses,
            )
        else:
            updated[p] = max(0.0, updated[p] - step_down)

    return updated


def update_theta_spsa(
    theta: dict[int, float],
    base_instance: VSRPInstance,
    solver: BaseSolver,
    rng: np.random.Generator,
    scenario: UncertaintyScenario,
    *,
    step_size: float,
    perturbation_size: float = 2.0,
    theta_max_h: float = DEFAULT_THETA_MAX_H,
    solve_options: SolveOptions | None = None,
) -> dict[int, float]:
    """
    Apply one SPSA update step.

    SPSA outline
    ------------
    1. draw a Rademacher perturbation vector
    2. evaluate realized service cost at:
       - $$\\theta + c\\delta$$
       - $$\\theta - c\\delta$$
    3. estimate a gradient from the two evaluations
    4. update theta in the descent direction
    5. clip to the feasible range

    Important
    ---------
    The objective used for the SPSA update is realized service cost, not
    the optimization objective value. This is intentional because
    different theta vectors solve different tightened instances, so raw
    optimization objectives are not directly comparable across policies.
    """
    solve_options = solve_options or SolveOptions()
    port_keys = sorted(theta.keys())
    n = len(port_keys)

    if n == 0:
        return theta.copy()

    delta = {
        p: float(rng.choice([-1.0, 1.0]))
        for p in port_keys
    }

    theta_plus = {
        p: np.clip(
            theta[p] + perturbation_size * delta[p],
            0.0,
            theta_max_h,
        )
        for p in port_keys
    }
    theta_minus = {
        p: np.clip(
            theta[p] - perturbation_size * delta[p],
            0.0,
            theta_max_h,
        )
        for p in port_keys
    }

    def _evaluate(theta_eval: dict[int, float]) -> float:
        """
        Evaluate realized service cost for one theta configuration.
        """
        opt_instance = deepcopy(base_instance)
        opt_instance.initial_delay_h = scenario.realized_initial_delay_h
        param_instance = apply_theta_to_instance(opt_instance, theta_eval)
        solution = solver.solve(param_instance, options=solve_options)

        if not solution.feasible:
            # Heavy penalty for infeasible perturbed policies.
            return 1e9

        realized = simulate_realized_performance(
            base_instance,
            solution,
            scenario,
        )
        return realized.realized_service_cost_usd

    cost_plus = _evaluate(theta_plus)
    cost_minus = _evaluate(theta_minus)

    gradient = {
        p: (cost_plus - cost_minus) / (
            2.0 * perturbation_size * delta[p]
        )
        for p in port_keys
    }

    updated = {
        p: float(np.clip(
            theta[p] - step_size * gradient[p],
            0.0,
            theta_max_h,
        ))
        for p in port_keys
    }

    return updated

def update_theta_spsa_fleet(
    theta: dict[int, float],
    base_fleet,
    fleet_solver,
    rng: np.random.Generator,
    fleet_scenario,
    *,
    step_size: float,
    perturbation_size: float = 2.0,
    theta_max_h: float = DEFAULT_THETA_MAX_H,
    solve_options: SolveOptions | None = None,
) -> dict[int, float]:
    """
    Apply one fleet-level SPSA update step.

    Fleet-level SPSA outline
    ------------------------
    1. Draw a shared Rademacher perturbation vector delta
    2. Evaluate total fleet realized service cost at:
       - theta + c * delta  (applied to all vessels simultaneously)
       - theta - c * delta  (applied to all vessels simultaneously)
    3. Estimate a gradient from the two fleet-level cost evaluations
    4. Update theta in the descent direction
    5. Clip to the feasible range

    The key difference from single-vessel SPSA is that both cost
    evaluations solve all vessel sub-problems and sum the realized
    service costs across the entire fleet. This produces a gradient
    that reflects the fleet-wide response to theta perturbations
    rather than a single vessel's response.

    Parameters
    ----------
    theta : dict[int, float]
        Current shared port-indexed tightening vector.
    base_fleet : FleetInstance
        Original fleet instance used as simulation ground truth.
    fleet_solver : FleetSolver
        Fleet solver used to solve all vessel sub-problems.
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    fleet_scenario : FleetUncertaintyScenario
        Realized fleet uncertainty scenario for this episode.
    step_size : float
        Gradient descent step size.
    perturbation_size : float, default=2.0
        Perturbation magnitude c in the SPSA formula.
    theta_max_h : float, default=DEFAULT_THETA_MAX_H
        Maximum allowed theta value in hours.
    solve_options : SolveOptions | None, default=None
        Solve controls applied to every per-vessel sub-problem.

    Returns
    -------
    dict[int, float]
        Updated theta vector after one fleet-level SPSA step.
    """
    from copy import deepcopy

    from core.simulation import simulate_fleet_realized_performance
    from experiments.fleet_cfa import apply_fleet_theta

    solve_options = solve_options or SolveOptions()
    port_keys = sorted(theta.keys())

    if not port_keys:
        return theta.copy()

    # Step 1: Draw shared Rademacher perturbation vector
    delta = {
        p: float(rng.choice([-1.0, 1.0]))
        for p in port_keys
    }

    theta_plus = {
        p: float(np.clip(
            theta[p] + perturbation_size * delta[p],
            0.0,
            theta_max_h,
        ))
        for p in port_keys
    }
    theta_minus = {
        p: float(np.clip(
            theta[p] - perturbation_size * delta[p],
            0.0,
            theta_max_h,
        ))
        for p in port_keys
    }

    def _evaluate_fleet(theta_eval: dict[int, float]) -> float:
        """
        Evaluate total fleet realized service cost for one theta vector.

        Solves all vessel sub-problems with the tightened theta and
        simulates realized performance on the original base fleet.
        Returns the sum of realized service costs across all vessels.
        Returns a large penalty value if any vessel is infeasible.
        """
        opt_fleet = deepcopy(base_fleet)
        for v_idx, vessel_instance in enumerate(opt_fleet.vessel_instances):
            vessel_instance.initial_delay_h = (
                fleet_scenario.vessel_scenarios[v_idx]
                .realized_initial_delay_h
            )
        opt_fleet = apply_fleet_theta(opt_fleet, theta_eval)

        fleet_solution = fleet_solver.solve(opt_fleet, options=solve_options)

        if not fleet_solution.feasible:
            return 1e9

        vessel_performances = simulate_fleet_realized_performance(
            base_fleet,
            fleet_solution,
            fleet_scenario,
        )

        return sum(
            p.realized_service_cost_usd for p in vessel_performances
        )

    # Step 2: Evaluate fleet cost at both perturbation points
    cost_plus = _evaluate_fleet(theta_plus)
    cost_minus = _evaluate_fleet(theta_minus)

    # Step 3: Estimate gradient
    gradient = {
        p: (cost_plus - cost_minus) / (
            2.0 * perturbation_size * delta[p]
        )
        for p in port_keys
    }

    # Step 4 and 5: Update and clip
    updated = {
        p: float(np.clip(
            theta[p] - step_size * gradient[p],
            0.0,
            theta_max_h,
        ))
        for p in port_keys
    }

    return updated


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train_cfa(
    base_instance: VSRPInstance,
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
) -> CFAPolicyResult:
    """
    Train CFA parameters over repeated simulated episodes.

    Per episode workflow
    --------------------
    1. sample an estimated delay for optimization
    2. sample a realized uncertainty scenario for simulation
    3. build an optimization instance using the estimated delay
    4. tighten promised arrivals using the current theta vector
    5. solve the tightened optimization instance
    6. simulate realized performance on the original instance
    7. update theta using the chosen policy rule

    Notes
    -----
    Episode rows are marked with `objective_comparable=False` because
    optimization objectives are not directly comparable across policies
    or theta values. The fair comparison metric is realized performance.
    """
    solve_options = solve_options or SolveOptions()
    config = uncertainty_config or DEFAULT_UNCERTAINTY_CONFIG
    rng = np.random.default_rng(seed)

    theta = initialize_theta(base_instance)
    theta_history: list[dict[int, float]] = [theta.copy()]
    rows: list[dict] = []

    for episode in range(1, n_episodes + 1):
        est_delay_h = sample_estimated_delay(rng, config)
        scenario = sample_uncertainty_scenario(rng, base_instance, config)

        opt_instance = deepcopy(base_instance)
        opt_instance.initial_delay_h = est_delay_h

        param_instance = apply_theta_to_instance(opt_instance, theta)
        solution = solver.solve(param_instance, options=solve_options)

        realized = simulate_realized_performance(
            base_instance,
            solution,
            scenario,
        )

        if update_policy == "additive":
            theta = update_theta_additive(
                theta,
                realized.missed_by_port,
                step_size=step_size,
                theta_max_h=theta_max_h,
            )
        elif update_policy == "decay":
            theta = update_theta_decay(
                theta,
                realized.missed_by_port,
                step_size=step_size,
                step_up=step_up,
                step_down=step_down,
                theta_max_h=theta_max_h,
            )
        elif update_policy == "spsa":
            theta = update_theta_spsa(
                theta,
                base_instance=base_instance,
                solver=solver,
                rng=rng,
                scenario=scenario,
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

        fueleu_limit = get_fueleu_limit(2026)
        fueleu_compliant = (
            DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ <= fueleu_limit
        )

        rows.append({
            "episode": episode,
            "phase": "train",
            "policy": update_policy,
            "objective_comparable": False,
            "estimated_delay_h": est_delay_h,
            "realized_delay_h": scenario.realized_initial_delay_h,
            "realized_carbon_price_eur": scenario.realized_carbon_price_eur,
            "realized_fuel_price_usd": scenario.realized_fuel_price_usd,
            "objective_value": solution.objective_value,
            "solver_feasible": solution.feasible,
            "solver_optimal": solution.optimal,
            "runtime_s": (
                solution.solver_stats.runtime_s
                if solution.solver_stats else None
            ),
            "mip_gap": (
                solution.solver_stats.mip_gap
                if solution.solver_stats else None
            ),
            "realized_total_missed": realized.total_missed,
            "realized_total_missed_containers": (
                realized.total_missed_containers
            ),
            "realized_service_cost_usd": realized.realized_service_cost_usd,
            "realized_delay_cost_usd": realized.realized_delay_cost_usd,
            "realized_misconnection_cost_usd": (
                realized.realized_misconnection_cost_usd
            ),
            "realized_fuel_cost_usd": realized.realized_fuel_cost_usd,
            "realized_ets_cost_eur": realized.realized_ets_cost_eur,
            "realized_fueleu_compliant": fueleu_compliant,
            "n_skipped": solution.n_skipped,
            "n_swapped": solution.n_swapped,
            "theta_mean_h": float(np.mean(list(theta.values()))),
            "theta_max_h": float(
                max(theta.values()) if theta else 0.0
            ),
        })

    return CFAPolicyResult(
        results_df=pd.DataFrame(rows),
        theta_history=theta_history,
    )


# =============================================================================
# 6. TESTING LOOP
# =============================================================================

def test_cfa_policy(
    base_instance: VSRPInstance,
    solver: BaseSolver,
    theta: dict[int, float],
    *,
    n_episodes: int = 50,
    solve_options: SolveOptions | None = None,
    seed: int = 123,
    policy_name: str = "trained_policy",
    uncertainty_config: UncertaintyConfig | None = None,
) -> CFAPolicyResult:
    """
    Evaluate a fixed theta policy over repeated simulated test episodes.

    Per episode workflow
    --------------------
    1. sample an estimated delay for optimization
    2. sample a realized uncertainty scenario
    3. solve the theta-tightened optimization instance
    4. simulate realized performance on the original instance
    5. record realized metrics without updating theta

    Notes
    -----
    Optimization objectives remain marked as not directly comparable
    across policies. Realized service cost is the primary policy metric.
    """
    solve_options = solve_options or SolveOptions()
    config = uncertainty_config or DEFAULT_UNCERTAINTY_CONFIG
    rng = np.random.default_rng(seed)

    theta_history: list[dict[int, float]] = [theta.copy()]
    rows: list[dict] = []

    for episode in range(1, n_episodes + 1):
        est_delay_h = sample_estimated_delay(rng, config)
        scenario = sample_uncertainty_scenario(rng, base_instance, config)

        opt_instance = deepcopy(base_instance)
        opt_instance.initial_delay_h = est_delay_h

        param_instance = apply_theta_to_instance(opt_instance, theta)
        solution = solver.solve(param_instance, options=solve_options)

        realized = simulate_realized_performance(
            base_instance,
            solution,
            scenario,
        )

        fueleu_limit = get_fueleu_limit(2026)
        fueleu_compliant = (
            DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ <= fueleu_limit
        )

        rows.append({
            "episode": episode,
            "phase": "test",
            "policy": policy_name,
            "objective_comparable": False,
            "estimated_delay_h": est_delay_h,
            "realized_delay_h": scenario.realized_initial_delay_h,
            "realized_carbon_price_eur": scenario.realized_carbon_price_eur,
            "realized_fuel_price_usd": scenario.realized_fuel_price_usd,
            "objective_value": solution.objective_value,
            "solver_feasible": solution.feasible,
            "solver_optimal": solution.optimal,
            "runtime_s": (
                solution.solver_stats.runtime_s
                if solution.solver_stats else None
            ),
            "mip_gap": (
                solution.solver_stats.mip_gap
                if solution.solver_stats else None
            ),
            "realized_total_missed": realized.total_missed,
            "realized_total_missed_containers": (
                realized.total_missed_containers
            ),
            "realized_service_cost_usd": realized.realized_service_cost_usd,
            "realized_delay_cost_usd": realized.realized_delay_cost_usd,
            "realized_misconnection_cost_usd": (
                realized.realized_misconnection_cost_usd
            ),
            "realized_fuel_cost_usd": realized.realized_fuel_cost_usd,
            "realized_ets_cost_eur": realized.realized_ets_cost_eur,
            "realized_fueleu_compliant": fueleu_compliant,
            "n_skipped": solution.n_skipped,
            "n_swapped": solution.n_swapped,
            "theta_mean_h": float(np.mean(list(theta.values()))),
            "theta_max_h": float(
                max(theta.values()) if theta else 0.0
            ),
        })

    return CFAPolicyResult(
        results_df=pd.DataFrame(rows),
        theta_history=theta_history,
    )


# =============================================================================
# 7. TAIL RISK SUMMARY
# =============================================================================

def compute_tail_risk_summary(
    results_df: pd.DataFrame,
    *,
    cost_col: str = "realized_service_cost_usd",
    percentiles: list[float] | None = None,
) -> dict:
    """
    Compute tail-risk statistics for a CFA results table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output from `train_cfa()` or `test_cfa_policy()`.
    cost_col : str, default="realized_service_cost_usd"
        Column on which tail-risk statistics are computed.
    percentiles : list[float] | None
        Percentiles to compute. Defaults to:
        [50, 75, 90, 95, 99]

    Returns
    -------
    dict
        Tail-risk summary including:
        - mean
        - standard deviation
        - selected percentiles
        - FuelEU compliance rate, when available
        - average realized ETS cost, when available
    """
    percentiles = percentiles or [50, 75, 90, 95, 99]

    if results_df.empty or cost_col not in results_df.columns:
        return {}

    costs = results_df[cost_col].dropna()

    result: dict = {
        "mean": float(costs.mean()),
        "std": float(costs.std()),
    }

    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(costs, p))

    if "realized_fueleu_compliant" in results_df.columns:
        result["fueleu_compliance_rate"] = float(
            results_df["realized_fueleu_compliant"].mean()
        )

    if "realized_ets_cost_eur" in results_df.columns:
        result["avg_realized_ets_cost_eur"] = float(
            results_df["realized_ets_cost_eur"].mean()
        )

    return result