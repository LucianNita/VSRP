# =============================================================================
# Solver-agnostic simulation helpers for realized VSRP performance.
#
# Purpose
# -------
# This module evaluates fixed optimization solutions under realized
# uncertainty. It is primarily used by:
# - the CFA policy framework
# - stochastic ETS exposure experiments
# - post-solve realized-cost analysis
#
# Main responsibilities
# ---------------------
# - define a full realized uncertainty scenario
# - sample uncertainty from configurable distributions
# - reconstruct route arrivals under realized conditions
# - evaluate missed containers and realized service cost
# - compute realized fuel and ETS cost under scenario prices
#
# Architectural role
# ------------------
# The optimization model is solved on an estimated state, but performance
# is evaluated on a realized state. This file implements that separation
# between planning and realization, which is central to the CFA workflow.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.entities import VSRPInstance, VSRPSolution


# =============================================================================
# 1. UNCERTAINTY SCENARIO
# =============================================================================

@dataclass(slots=True)
class UncertaintyScenario:
    """
    One realized uncertainty scenario for a simulation episode.

    This object bundles all exogenous uncertainty affecting realized
    performance after the optimization model has already selected a route.

    Attributes
    ----------
    realized_initial_delay_h : float
        Actual initial disruption delay at the voyage origin in hours.
    port_handling_multipliers : dict[int, float]
        Per-port multiplier applied to nominal port-call duration.
        Example: 1.0 = nominal handling, 1.5 = 50% slower handling.
    weather_speed_factors : dict[int, float]
        Per-leg effective speed factor keyed by `from_port_idx`.
        Example: 1.0 = no weather effect, 0.8 = 20% speed reduction.
    realized_carbon_price_eur : float
        Realized ETS carbon price used in post-solve ETS cost evaluation.
    realized_fuel_price_usd : float
        Realized fuel price used in post-solve fuel-cost evaluation.
    """
    realized_initial_delay_h: float

    port_handling_multipliers: dict[int, float] = field(
        default_factory=dict
    )
    weather_speed_factors: dict[int, float] = field(
        default_factory=dict
    )

    realized_carbon_price_eur: float = 65.0
    realized_fuel_price_usd: float = 600.0


@dataclass(slots=True)
class UncertaintyConfig:
    """
    Configuration for sampling uncertainty scenarios.

    The current implementation uses clipped normal draws as a simple,
    reproducible approximation to truncated normal sampling.

    Estimated vs realized delay
    ---------------------------
    The model distinguishes between:
    - estimated delay: used when building the optimization instance
    - realized delay: used when simulating realized execution

    This separation is essential for the CFA setup, where the policy is
    optimized on an estimate but judged on out-of-sample realization.
    """
    # Estimated delay used in optimization
    est_delay_mean_h: float = 48.0
    est_delay_std_h: float = 12.0

    # Realized delay used in simulation
    real_delay_mean_h: float = 55.0
    real_delay_std_h: float = 10.0
    delay_min_h: float = 20.0
    delay_max_h: float = 80.0

    # Port handling time uncertainty
    port_handling_mean: float = 1.0
    port_handling_std: float = 0.1
    handling_min: float = 0.8
    handling_max: float = 1.5

    # Weather / effective speed uncertainty
    weather_factor_mean: float = 1.0
    weather_factor_std: float = 0.05
    weather_min: float = 0.7
    weather_max: float = 1.0

    # ETS carbon price uncertainty
    carbon_price_mean_eur: float = 65.0
    carbon_price_std_eur: float = 15.0
    carbon_price_min_eur: float = 30.0
    carbon_price_max_eur: float = 130.0

    # Fuel price uncertainty
    fuel_price_mean_usd: float = 600.0
    fuel_price_std_usd: float = 80.0
    fuel_price_min_usd: float = 400.0
    fuel_price_max_usd: float = 900.0


# =============================================================================
# 2. SCENARIO SAMPLING
# =============================================================================

def _clip_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    """
    Draw one clipped normal sample.

    This is a lightweight approximation to truncated normal sampling and
    is sufficient for the current simulation and CFA experiments.
    """
    return float(np.clip(rng.normal(mean, std), low, high))


def sample_uncertainty_scenario(
    rng: np.random.Generator,
    instance: VSRPInstance,
    config: UncertaintyConfig,
) -> UncertaintyScenario:
    """
    Sample one full realized uncertainty scenario.

    Sampled components
    ------------------
    - realized initial delay
    - per-port handling multipliers
    - per-leg weather speed factors
    - realized ETS carbon price
    - realized fuel price

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    instance : VSRPInstance
        Used to determine valid port and leg indices.
    config : UncertaintyConfig
        Distribution parameters.

    Returns
    -------
    UncertaintyScenario
        One realized stochastic scenario.
    """
    realized_delay_h = _clip_normal(
        rng,
        config.real_delay_mean_h,
        config.real_delay_std_h,
        config.delay_min_h,
        config.delay_max_h,
    )

    # Sample port-handling multipliers for interior ports only.
    port_handling_multipliers: dict[int, float] = {}
    for p in range(1, instance.n_ports - 1):
        port_handling_multipliers[p] = _clip_normal(
            rng,
            config.port_handling_mean,
            config.port_handling_std,
            config.handling_min,
            config.handling_max,
        )

    # Sample weather speed factors keyed by leg origin.
    weather_speed_factors: dict[int, float] = {}
    for p in range(instance.n_ports - 1):
        weather_speed_factors[p] = _clip_normal(
            rng,
            config.weather_factor_mean,
            config.weather_factor_std,
            config.weather_min,
            config.weather_max,
        )

    realized_carbon_price_eur = _clip_normal(
        rng,
        config.carbon_price_mean_eur,
        config.carbon_price_std_eur,
        config.carbon_price_min_eur,
        config.carbon_price_max_eur,
    )

    realized_fuel_price_usd = _clip_normal(
        rng,
        config.fuel_price_mean_usd,
        config.fuel_price_std_usd,
        config.fuel_price_min_usd,
        config.fuel_price_max_usd,
    )

    return UncertaintyScenario(
        realized_initial_delay_h=realized_delay_h,
        port_handling_multipliers=port_handling_multipliers,
        weather_speed_factors=weather_speed_factors,
        realized_carbon_price_eur=realized_carbon_price_eur,
        realized_fuel_price_usd=realized_fuel_price_usd,
    )


def sample_estimated_delay(
    rng: np.random.Generator,
    config: UncertaintyConfig,
) -> float:
    """
    Sample the estimated initial delay used when constructing the
    optimization instance for one episode.
    """
    return _clip_normal(
        rng,
        config.est_delay_mean_h,
        config.est_delay_std_h,
        config.delay_min_h,
        config.delay_max_h,
    )


# =============================================================================
# 3. ROUTE TIMING UNDER UNCERTAINTY
# =============================================================================

def compute_actual_route_arrivals(
    instance: VSRPInstance,
    solution: VSRPSolution,
    scenario: UncertaintyScenario,
) -> dict[int, float]:
    """
    Compute realized arrival times at visited ports for a fixed route.

    Realized timing incorporates:
    1. realized initial delay
    2. weather-reduced effective sailing speed
    3. port-handling multipliers at visited ports

    Parameters
    ----------
    instance : VSRPInstance
        Original optimization instance.
    solution : VSRPSolution
        Fixed solved route to be evaluated.
    scenario : UncertaintyScenario
        Realized uncertainty for the episode.

    Returns
    -------
    dict[int, float]
        Mapping `port_idx -> realized arrival time`.
    """
    route = solution.route_legs
    if not route:
        return {}

    arrivals: dict[int, float] = {}

    origin_idx = route[0].from_port_idx
    arrivals[origin_idx] = scenario.realized_initial_delay_h

    # Departure from origin uses the standard port-call duration scaled
    # by any realized handling multiplier.
    origin_handling_mult = scenario.port_handling_multipliers.get(
        origin_idx,
        1.0,
    )
    nominal_duration = instance.port_call_profile.durations_h[0]
    current_time = (
        scenario.realized_initial_delay_h
        + nominal_duration * origin_handling_mult
    )

    for leg in route:
        weather_factor = scenario.weather_speed_factors.get(
            leg.from_port_idx,
            1.0,
        )
        effective_speed = max(
            leg.speed_knots * weather_factor,
            1.0,
        )

        distance_nm = instance.distance_matrix_nm[
            leg.from_port_idx
        ][leg.to_port_idx]
        travel_h = distance_nm / effective_speed

        arrival_h = current_time + travel_h
        arrivals[leg.to_port_idx] = arrival_h

        handling_mult = scenario.port_handling_multipliers.get(
            leg.to_port_idx,
            1.0,
        )
        port_duration = (
            instance.port_call_profile.durations_h[leg.duration_idx]
            * handling_mult
        )
        current_time = arrival_h + port_duration

    return arrivals


# =============================================================================
# 4. CONTAINER-LEVEL REALIZED OUTCOMES
# =============================================================================

def evaluate_realized_container_misses(
    instance: VSRPInstance,
    solution: VSRPSolution,
    scenario: UncertaintyScenario,
) -> dict:
    """
    Evaluate realized service misses against the original promised arrivals.

    A container is counted as missed if:
    - its destination is not reached in the realized route, or
    - realized arrival at its destination exceeds its promised arrival

    Important note
    --------------
    The function returns `total_missed_containers`, which is simply the
    count of missed containers. This is not a distinct "delayed count"
    separate from the missed-container concept.

    Returns
    -------
    dict
        Dictionary containing:
        - actual_arrivals_by_port
        - missed_by_port
        - missed_container_ids
        - total_missed
        - total_missed_containers
        - realized_delay_cost_usd
        - realized_misconnection_cost_usd
        - realized_service_cost_usd
        - realized_fuel_cost_usd
        - realized_ets_cost_eur
    """
    actual_arrivals = compute_actual_route_arrivals(
        instance,
        solution,
        scenario,
    )

    missed_by_port = {p: 0 for p in range(instance.n_ports)}
    missed_container_ids: list[str] = []

    realized_delay_cost_usd = 0.0
    realized_misconnection_cost_usd = 0.0

    for container in instance.containers:
        dest = container.destination_idx
        arrival_h = actual_arrivals.get(dest)

        if arrival_h is None:
            missed_by_port[dest] += 1
            missed_container_ids.append(container.id)
            realized_misconnection_cost_usd += container.penalty_misconnect
            continue

        if arrival_h > container.promised_arrival_h:
            missed_by_port[dest] += 1
            missed_container_ids.append(container.id)
            realized_delay_cost_usd += container.penalty_delay

    total_missed = len(missed_container_ids)

    realized_fuel_cost_usd = _compute_realized_fuel_cost(
        instance,
        solution,
        scenario,
    )
    realized_ets_cost_eur = _compute_realized_ets_cost(
        instance,
        solution,
        scenario,
    )

    return {
        "actual_arrivals_by_port": actual_arrivals,
        "missed_by_port": missed_by_port,
        "missed_container_ids": missed_container_ids,
        "total_missed": total_missed,
        "total_missed_containers": total_missed,
        "realized_delay_cost_usd": realized_delay_cost_usd,
        "realized_misconnection_cost_usd": realized_misconnection_cost_usd,
        "realized_service_cost_usd": (
            realized_delay_cost_usd + realized_misconnection_cost_usd
        ),
        "realized_fuel_cost_usd": realized_fuel_cost_usd,
        "realized_ets_cost_eur": realized_ets_cost_eur,
    }


# =============================================================================
# 5. HIGH-LEVEL SIMULATION WRAPPER
# =============================================================================

@dataclass(slots=True)
class RealizedPerformance:
    """
    Realized episode-level performance summary.

    This dataclass is the canonical simulation output returned by
    `simulate_realized_performance()`.
    """
    realized_delay_h: float

    actual_arrivals_by_port: dict[int, float]
    missed_by_port: dict[int, int]

    total_missed: int
    total_missed_containers: int

    realized_delay_cost_usd: float
    realized_misconnection_cost_usd: float
    realized_service_cost_usd: float

    realized_fuel_cost_usd: float
    realized_ets_cost_eur: float

    scenario: UncertaintyScenario | None = None


def simulate_realized_performance(
    instance: VSRPInstance,
    solution: VSRPSolution,
    scenario: UncertaintyScenario,
) -> RealizedPerformance:
    """
    Simulate one realized episode for a fixed solved solution.

    The route is held fixed; this function does not re-optimize.
    Instead, it evaluates how the chosen plan performs under one realized
    uncertainty scenario.

    Returns
    -------
    RealizedPerformance
        Structured realized-performance summary.
    """
    result = evaluate_realized_container_misses(
        instance,
        solution,
        scenario,
    )

    return RealizedPerformance(
        realized_delay_h=scenario.realized_initial_delay_h,
        actual_arrivals_by_port=result["actual_arrivals_by_port"],
        missed_by_port=result["missed_by_port"],
        total_missed=result["total_missed"],
        total_missed_containers=result["total_missed_containers"],
        realized_delay_cost_usd=result["realized_delay_cost_usd"],
        realized_misconnection_cost_usd=result[
            "realized_misconnection_cost_usd"
        ],
        realized_service_cost_usd=result["realized_service_cost_usd"],
        realized_fuel_cost_usd=result["realized_fuel_cost_usd"],
        realized_ets_cost_eur=result["realized_ets_cost_eur"],
        scenario=scenario,
    )


# =============================================================================
# 6. INTERNAL HELPERS
# =============================================================================

def _compute_realized_fuel_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
    scenario: UncertaintyScenario,
) -> float:
    """
    Compute realized fuel cost using the scenario fuel price.

    Modeling assumption
    -------------------
    Weather affects realized travel time through effective speed, but it
    does not alter fuel burn directly in the current approximation.
    Fuel burn is computed from the selected planned speed on each leg.
    """
    from core.emissions import fuel_consumption_tonnes

    total_fuel_t = 0.0
    for leg in solution.route_legs:
        dist_nm = instance.distance_matrix_nm[
            leg.from_port_idx
        ][leg.to_port_idx]
        fuel_t = fuel_consumption_tonnes(
            distance_nm=dist_nm,
            speed_knots=leg.speed_knots,
        )
        total_fuel_t += fuel_t

    return total_fuel_t * scenario.realized_fuel_price_usd


def _compute_realized_ets_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
    scenario: UncertaintyScenario,
) -> float:
    """
    Compute realized ETS cost using the scenario carbon price.

    ETS exposure is based on realized carbon price but the current
    implementation still uses route-leg fuel burn implied by the
    selected planned speed.
    """
    from core.emissions import (
        DEFAULT_EU_ETS_PHASE_IN,
        co2_from_fuel_tonnes,
        fuel_consumption_tonnes,
    )

    total_co2_t = 0.0
    for leg in solution.route_legs:
        dist_nm = instance.distance_matrix_nm[
            leg.from_port_idx
        ][leg.to_port_idx]
        fuel_t = fuel_consumption_tonnes(
            distance_nm=dist_nm,
            speed_knots=leg.speed_knots,
        )
        total_co2_t += co2_from_fuel_tonnes(fuel_t)

    phase_in = DEFAULT_EU_ETS_PHASE_IN.get(2026, 1.0)
    return total_co2_t * phase_in * scenario.realized_carbon_price_eur