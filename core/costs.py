# =============================================================================
# Solver-agnostic cost and KPI calculations for the VSRP.
#
# Purpose
# -------
# This module recomputes the objective decomposition from a canonical
# `VSRPSolution`. It provides a post-solve audit layer that is independent
# of the solver backend.
#
# Main uses
# ---------
# - verify that the extracted solution is cost-consistent with the solver
#   objective
# - provide detailed operational and service cost breakdowns
# - populate benchmark, sensitivity, CFA, and reporting tables
#
# Architectural role
# ------------------
# Cost recomputation is performed outside the optimization backend so that:
# - different solver adapters can be compared consistently
# - post-solve diagnostics can catch extraction or accounting mismatches
# - experiments can report decomposed KPIs instead of only total objective
#
# Important design choice
# -----------------------
# Fuel cost uses `instance.fuel_price_usd_per_tonne` rather than a hardcoded
# constant. This is essential for consistency in stochastic fuel-price and
# sensitivity experiments.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass

from core.entities import VSRPInstance, VSRPSolution


@dataclass(slots=True)
class CostBreakdown:
    """
    Structured post-solve cost decomposition in USD.

    Fields
    ------
    fuel_cost_usd : float
        Route-level fuel cost.
    port_call_cost_usd : float
        Port handling cost from selected duration options.
    strategy_penalty_usd : float
        Penalty cost from tactical recovery actions.
    port_penalty_cost_usd : float
        Additional port-specific penalties active in the instance.
    fueleu_penalty_usd : float
        FuelEU Maritime penalty included in the operational cost when enabled.
    delay_cost_usd : float
        Service cost from containers marked delayed.
    misconnection_cost_usd : float
        Service cost from containers marked misconnected.
    operational_cost_usd : float
        Total operational cost.
    service_cost_usd : float
        Total service cost.
    weighted_objective_usd : float
        Final weighted objective:
        $$ (1-\alpha)\,C_{op} + \alpha\,C_{svc} $$
    """
    fuel_cost_usd: float = 0.0
    port_call_cost_usd: float = 0.0
    strategy_penalty_usd: float = 0.0
    port_penalty_cost_usd: float = 0.0
    fueleu_penalty_usd: float = 0.0

    delay_cost_usd: float = 0.0
    misconnection_cost_usd: float = 0.0

    operational_cost_usd: float = 0.0
    service_cost_usd: float = 0.0

    weighted_objective_usd: float = 0.0


def compute_cost_breakdown(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> CostBreakdown:
    """
    Compute a full post-solve cost breakdown from a canonical solution.

    Objective form
    --------------
    $$
    \text{Objective} = (1 - \alpha)\,C_{op} + \alpha\,C_{svc}
    $$

    where:
    - $$C_{op}$$ is operational cost
    - $$C_{svc}$$ is service cost

    Returns
    -------
    CostBreakdown
        Fully decomposed cost record.
    """
    fuel_cost = compute_fuel_cost(instance, solution)
    port_call_cost = compute_port_call_cost(instance, solution)
    strategy_penalty = compute_strategy_penalty_cost(instance, solution)
    port_penalty_cost = compute_port_specific_penalty_cost(instance, solution)
    fueleu_penalty = compute_fueleu_penalty_cost(instance, solution)

    delay_cost = compute_delay_cost(instance, solution)
    misconnection_cost = compute_misconnection_cost(instance, solution)

    operational_cost = (
        fuel_cost
        + port_call_cost
        + strategy_penalty
        + port_penalty_cost
        + fueleu_penalty
    )
    service_cost = delay_cost + misconnection_cost

    weighted_objective = (
        (1.0 - instance.alpha) * operational_cost
        + instance.alpha * service_cost
    )

    return CostBreakdown(
        fuel_cost_usd=fuel_cost,
        port_call_cost_usd=port_call_cost,
        strategy_penalty_usd=strategy_penalty,
        port_penalty_cost_usd=port_penalty_cost,
        fueleu_penalty_usd=fueleu_penalty,
        delay_cost_usd=delay_cost,
        misconnection_cost_usd=misconnection_cost,
        operational_cost_usd=operational_cost,
        service_cost_usd=service_cost,
        weighted_objective_usd=weighted_objective,
    )


# =============================================================================
# OPERATIONAL COSTS
# =============================================================================

def compute_fuel_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Sum fuel cost over all selected route legs.

    The fuel price is read from `instance.fuel_price_usd_per_tonne` so that
    post-solve recomputation remains consistent with experiments that vary
    fuel price across scenarios or episodes.
    """
    route = solution.route_legs
    if not route:
        return 0.0

    total = 0.0
    distance_matrix = instance.distance_matrix_nm

    for leg in route:
        i = leg.from_port_idx
        j = leg.to_port_idx
        distance_nm = distance_matrix[i][j]

        total += _fuel_cost_usd(
            distance_nm=distance_nm,
            speed_knots=leg.speed_knots,
            fuel_price_usd_per_tonne=instance.fuel_price_usd_per_tonne,
        )

    return total


def compute_port_call_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Sum port handling costs for visited interior ports.

    The selected `duration_idx` on each route leg determines which cost
    entry from `instance.port_call_profile.costs_usd` is charged.
    """
    route = solution.route_legs
    if not route:
        return 0.0

    total = 0.0
    cost_vector = instance.port_call_profile.costs_usd
    final_port_idx = instance.n_ports - 1

    for leg in route:
        arrival_port = leg.to_port_idx
        if arrival_port == 0 or arrival_port == final_port_idx:
            continue

        d = leg.duration_idx
        if 0 <= d < len(cost_vector):
            total += cost_vector[d]

    return total

def compute_strategy_penalty_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Sum tactical strategy penalties.

    Charging convention
    -------------------
    - SPEED_UP, EXPEDITED_PORT, and PORT_OMISSION are charged per
      reported strategy decision in the canonical solution.
    - PORT_SWAP is charged once per active swap group, not once per
      swapped port.

    This distinction is important because the extracted solution reports
    PORT_SWAP tags at the port level, while the optimization model treats
    a swap as one structural grouped action.
    """
    penalties = instance.penalties
    total = 0.0

    for s in solution.strategy_decisions:
        if s.strategy == "SPEED_UP":
            total += penalties.speed_up_usd
        elif s.strategy == "EXPEDITED_PORT":
            total += penalties.expedited_port_usd
        elif s.strategy == "PORT_OMISSION":
            total += penalties.omission_usd

    active_swap_groups = {
        leg.swap_group_id
        for leg in solution.route_legs
        if leg.is_swap and leg.swap_group_id is not None
    }
    total += penalties.swap_usd * len(active_swap_groups)

    return total


def compute_port_specific_penalty_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Sum additional instance-specific port penalties for visited ports.

    These penalties are separate from the generic tactical recovery
    penalties and are applied only when explicitly present in
    `instance.port_penalties_usd`.
    """
    if not instance.port_penalties_usd:
        return 0.0

    visited_destinations = {leg.to_port_idx for leg in solution.route_legs}
    total = 0.0

    for port_idx, penalty in instance.port_penalties_usd.items():
        if port_idx in visited_destinations:
            total += penalty

    return total


def compute_fueleu_penalty_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Read the FuelEU penalty from the attached emissions summary.

    Returns
    -------
    float
        FuelEU penalty in USD.

    Notes
    -----
    - Returns 0.0 when FuelEU is not active for the instance.
    - Returns 0.0 when no emissions summary is attached.
    - This function assumes emissions have already been computed and
      attached to the canonical solution.
    """
    if not instance.include_fueleu_penalty:
        return 0.0

    if solution.emissions is None:
        return 0.0

    return solution.emissions.total_fueleu_penalty_usd or 0.0


# =============================================================================
# SERVICE COSTS
# =============================================================================

def compute_delay_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Sum delay penalties for all containers marked delayed.

    The penalties are looked up from the original instance container
    records using container ID.
    """
    if not solution.container_outcomes:
        return 0.0

    penalties_by_id = {c.id: c.penalty_delay for c in instance.containers}
    total = 0.0

    for c_id, outcome in solution.container_outcomes.items():
        if outcome.delayed:
            total += penalties_by_id.get(c_id, 0.0)

    return total


def compute_misconnection_cost(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float:
    """
    Sum misconnection penalties for all containers marked misconnected.

    The penalties are looked up from the original instance container
    records using container ID.
    """
    if not solution.container_outcomes:
        return 0.0

    penalties_by_id = {
        c.id: c.penalty_misconnect for c in instance.containers
    }
    total = 0.0

    for c_id, outcome in solution.container_outcomes.items():
        if outcome.misconnected:
            total += penalties_by_id.get(c_id, 0.0)

    return total


# =============================================================================
# COMPARISON / DIAGNOSTIC HELPERS
# =============================================================================

def objective_gap_to_reported(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> float | None:
    """
    Compare the recomputed weighted objective with the solver-reported value.

    Returns
    -------
    float | None
        Absolute objective gap, or `None` if the solution does not carry
        a reported objective value.
    """
    if solution.objective_value is None:
        return None

    breakdown = compute_cost_breakdown(instance, solution)
    return abs(breakdown.weighted_objective_usd - solution.objective_value)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _fuel_consumption_tonnes(
    distance_nm: float,
    speed_knots: float,
    *,
    fuel_base_consumption_tpd: float = 100.0,
    reference_speed_knots: float = 20.0,
) -> float:
    """
    Internal helper: cubic-speed fuel-consumption calculation.
    """
    travel_time_h = distance_nm / speed_knots
    daily_burn_tpd = fuel_base_consumption_tpd * (
        speed_knots / reference_speed_knots
    ) ** 3
    return daily_burn_tpd * travel_time_h / 24.0


def _fuel_cost_usd(
    distance_nm: float,
    speed_knots: float,
    *,
    fuel_price_usd_per_tonne: float = 600.0,
    fuel_base_consumption_tpd: float = 100.0,
    reference_speed_knots: float = 20.0,
) -> float:
    """
    Internal helper: fuel-cost calculation from distance, speed, and fuel price.
    """
    return fuel_price_usd_per_tonne * _fuel_consumption_tonnes(
        distance_nm=distance_nm,
        speed_knots=speed_knots,
        fuel_base_consumption_tpd=fuel_base_consumption_tpd,
        reference_speed_knots=reference_speed_knots,
    )