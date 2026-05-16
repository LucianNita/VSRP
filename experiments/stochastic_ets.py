from __future__ import annotations

import numpy as np
import pandas as pd

from core.simulation import UncertaintyConfig, sample_uncertainty_scenario
from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from model.base import BaseSolver, SolveOptions


def evaluate_stochastic_ets_exposure(
    solver: BaseSolver,
    *,
    n_containers: int = 5,
    seed: int = 42,
    initial_delay_h: float = 48.0,
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    include_fueleu_penalty: bool = False,
    solve_options: SolveOptions | None = None,
    uncertainty_config: UncertaintyConfig | None = None,
    n_scenarios: int = 100,
    scenario_seed: int = 123,
) -> pd.DataFrame:
    """
    Solve one deterministic instance, then evaluate ETS exposure under
    stochastic carbon prices across many sampled scenarios.

    Workflow
    --------
    1. generate one container set
    2. build one deterministic optimization instance
    3. solve the route once
    4. sample many uncertainty scenarios
    5. recompute realized ETS cost for the fixed solved route

    Important interpretation
    ------------------------
    This experiment evaluates stochastic ETS exposure for a fixed solved
    recovery solution. It does not re-optimize the route for each carbon
    price scenario. The resulting distribution therefore reflects
    regulatory cost exposure under price uncertainty, not a fully
    endogenous stochastic control policy.

    Returns
    -------
    pd.DataFrame
        One row per sampled scenario.
    """
    solve_options = solve_options or SolveOptions()
    uncertainty_config = uncertainty_config or UncertaintyConfig()

    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=n_containers,
        seed=seed,
    )

    instance = build_base_instance(
        containers=containers,
        instance_id=f"stochastic_ets_seed_{seed}",
        initial_delay_h=initial_delay_h,
        alpha=alpha,
        allow_swap=allow_swap,
        max_skip=max_skip,
        include_fueleu_penalty=include_fueleu_penalty,
        metadata={"seed": seed},
    )

    solution = solver.solve(instance, options=solve_options)

    rng = np.random.default_rng(scenario_seed)
    rows: list[dict] = []

    # Use the shared uncertainty sampler so this experiment remains
    # consistent with the broader simulation framework, even though only
    # realized carbon price directly matters for the ETS exposure metric.
    from core.simulation import _compute_realized_ets_cost

    for scen_idx in range(1, n_scenarios + 1):
        scenario = sample_uncertainty_scenario(
            rng,
            instance,
            uncertainty_config,
        )

        realized_ets_cost_eur = _compute_realized_ets_cost(
            instance,
            solution,
            scenario,
        )

        rows.append({
            "instance_id": instance.instance_id,
            "solver_name": solver.solver_name,
            "base_seed": seed,
            "scenario_idx": scen_idx,
            "initial_delay_h": instance.initial_delay_h,
            "alpha": instance.alpha,
            "objective_value": solution.objective_value,
            "solution_feasible": solution.feasible,
            "solution_optimal": solution.optimal,
            "realized_carbon_price_eur": scenario.realized_carbon_price_eur,
            "realized_ets_cost_eur": realized_ets_cost_eur,
            "n_skipped": solution.n_skipped,
            "n_swapped": solution.n_swapped,
            "n_delayed": solution.n_delayed,
            "total_co2_t": (
                solution.emissions.total_co2_t
                if solution.emissions is not None
                else None
            ),
            "route_signature": "->".join(
                [str(solution.route_legs[0].from_port_idx)] +
                [str(leg.to_port_idx) for leg in solution.route_legs]
            ) if solution.route_legs else "",
        })

    return pd.DataFrame(rows)


def summarize_stochastic_ets(
    df: pd.DataFrame,
) -> dict:
    """
    Summarize a scenario-level ETS exposure DataFrame.

    Returned statistics
    -------------------
    - number of scenarios
    - mean / standard deviation of ETS cost
    - min / max ETS cost
    - median and upper-tail percentiles
    - mean / standard deviation of sampled carbon price
    - simple tail mean above the empirical p95 threshold

    Returns
    -------
    dict
        Summary of the realized ETS cost distribution.
    """
    if df.empty or "realized_ets_cost_eur" not in df.columns:
        return {}

    costs = df["realized_ets_cost_eur"].dropna()
    prices = df["realized_carbon_price_eur"].dropna()

    if costs.empty:
        return {}

    result = {
        "n_scenarios": int(len(costs)),
        "mean_ets_cost_eur": float(costs.mean()),
        "std_ets_cost_eur": float(costs.std()),
        "min_ets_cost_eur": float(costs.min()),
        "p50_ets_cost_eur": float(np.percentile(costs, 50)),
        "p90_ets_cost_eur": float(np.percentile(costs, 90)),
        "p95_ets_cost_eur": float(np.percentile(costs, 95)),
        "p99_ets_cost_eur": float(np.percentile(costs, 99)),
        "max_ets_cost_eur": float(costs.max()),
        "mean_carbon_price_eur": (
            float(prices.mean()) if not prices.empty else None
        ),
        "std_carbon_price_eur": (
            float(prices.std()) if not prices.empty else None
        ),
    }

    # Simple empirical tail-average statistic above the p95 threshold.
    p95 = np.percentile(costs, 95)
    tail = costs[costs >= p95]
    if len(tail) > 0:
        result["tail_mean_above_p95_eur"] = float(tail.mean())

    return result