from __future__ import annotations

from dataclasses import dataclass

from core.costs import CostBreakdown, compute_cost_breakdown
from core.entities import FleetInstance, FleetSolution


@dataclass(slots=True)
class FleetCostBreakdown:
    """
    Aggregated cost breakdown across all vessels in a fleet.

    Each field is the sum of the corresponding per-vessel cost
    component. The individual per-vessel breakdowns are retained
    in vessel_breakdowns for detailed inspection.

    Fields
    ------
    vessel_breakdowns : list[CostBreakdown]
        Per-vessel cost breakdowns in vessel order.
    fuel_cost_usd : float
        Total fuel cost across all vessels.
    port_call_cost_usd : float
        Total port handling cost across all vessels.
    strategy_penalty_usd : float
        Total strategy penalty cost across all vessels.
    port_penalty_cost_usd : float
        Total port-specific penalty cost across all vessels.
    fueleu_penalty_usd : float
        Total FuelEU penalty across all vessels.
    delay_cost_usd : float
        Total delay penalty cost across all vessels.
    misconnection_cost_usd : float
        Total misconnection penalty cost across all vessels.
    operational_cost_usd : float
        Total operational cost across all vessels.
    service_cost_usd : float
        Total service cost across all vessels.
    fleet_objective_usd : float
        Sum of per-vessel weighted objectives.
    """
    vessel_breakdowns: list[CostBreakdown]

    fuel_cost_usd: float = 0.0
    port_call_cost_usd: float = 0.0
    strategy_penalty_usd: float = 0.0
    port_penalty_cost_usd: float = 0.0
    fueleu_penalty_usd: float = 0.0
    delay_cost_usd: float = 0.0
    misconnection_cost_usd: float = 0.0
    operational_cost_usd: float = 0.0
    service_cost_usd: float = 0.0
    fleet_objective_usd: float = 0.0


def compute_fleet_cost_breakdown(
    fleet: FleetInstance,
    fleet_solution: FleetSolution,
) -> FleetCostBreakdown:
    """
    Compute an aggregated cost breakdown for an entire fleet.

    Iterates over all vessel instance and solution pairs, computes
    the per-vessel CostBreakdown using the existing single-vessel
    cost module, and aggregates the results into a FleetCostBreakdown.

    Parameters
    ----------
    fleet : FleetInstance
        Fleet-level instance containing per-vessel VSRPInstance objects.
    fleet_solution : FleetSolution
        Fleet-level solution containing per-vessel VSRPSolution objects.

    Returns
    -------
    FleetCostBreakdown
        Aggregated cost record with both fleet totals and per-vessel
        breakdowns retained for detailed inspection.
    """
    vessel_breakdowns = [
        compute_cost_breakdown(instance, solution)
        for instance, solution in zip(
            fleet.vessel_instances,
            fleet_solution.vessel_solutions,
        )
    ]

    return FleetCostBreakdown(
        vessel_breakdowns=vessel_breakdowns,
        fuel_cost_usd=sum(
            b.fuel_cost_usd for b in vessel_breakdowns
        ),
        port_call_cost_usd=sum(
            b.port_call_cost_usd for b in vessel_breakdowns
        ),
        strategy_penalty_usd=sum(
            b.strategy_penalty_usd for b in vessel_breakdowns
        ),
        port_penalty_cost_usd=sum(
            b.port_penalty_cost_usd for b in vessel_breakdowns
        ),
        fueleu_penalty_usd=sum(
            b.fueleu_penalty_usd for b in vessel_breakdowns
        ),
        delay_cost_usd=sum(
            b.delay_cost_usd for b in vessel_breakdowns
        ),
        misconnection_cost_usd=sum(
            b.misconnection_cost_usd for b in vessel_breakdowns
        ),
        operational_cost_usd=sum(
            b.operational_cost_usd for b in vessel_breakdowns
        ),
        service_cost_usd=sum(
            b.service_cost_usd for b in vessel_breakdowns
        ),
        fleet_objective_usd=sum(
            b.weighted_objective_usd for b in vessel_breakdowns
        ),
    )


def fleet_objective_gap_to_reported(
    fleet: FleetInstance,
    fleet_solution: FleetSolution,
) -> float | None:
    """
    Compare the recomputed fleet objective with the reported value.

    Returns
    -------
    float | None
        Absolute gap between recomputed and reported fleet objective,
        or None if the fleet solution does not carry a reported value.
    """
    if fleet_solution.fleet_objective_value is None:
        return None

    breakdown = compute_fleet_cost_breakdown(fleet, fleet_solution)
    return abs(
        breakdown.fleet_objective_usd
        - fleet_solution.fleet_objective_value
    )