# =============================================================================
# Base case-study instance data for the Vessel Schedule Recovery Problem (VSRP).
#
# Purpose
# -------
# This module defines the fixed route template and default operational
# settings used throughout the repository.
#
# Main responsibilities
# ---------------------
# - provide the nominal ordered port sequence
# - provide the aligned distance matrix
# - define default speed levels, port-call options, and penalties
# - build canonical `VSRPInstance` objects from generated container sets
#
# Architectural role
# ------------------
# This file is the bridge between:
# - the static case-study route data
# - the dynamic container-generation layer
# - the canonical `VSRPInstance` used by solver backends
# =============================================================================

from __future__ import annotations

from core.entities import (
    PenaltyProfile,
    PortCallProfile,
    VSRPInstance,
)


# Ordered nominal route used throughout the case study.
BASE_PORTS: list[str] = [
    "KWY", "BUS", "QIN", "NAG", "YOK1",
    "LGB", "OAK", "DHB", "YOK2", "KWY2",
]

# Pairwise distance matrix in nautical miles aligned with BASE_PORTS.
BASE_DISTANCE_MATRIX_NM: list[list[float]] = [
    [   0,  103,  421,  724,   837, 5377, 5061, 3044,  837,    0],
    [ 103,    0,  476,  699,   813, 5294, 4978, 2961,  813,  103],
    [ 421,  476,    0, 1029,  1143, 5748, 5431, 3414, 1143,  421],
    [ 724,  699, 1029,    0,   205, 4977, 4674, 2692,  205,  724],
    [ 837,  813, 1143,  205,     0, 4844, 4536, 2550,    0,  837],
    [5377, 5294, 5748, 4977,  4844,    0,  364, 2404, 4844, 5377],
    [5061, 4978, 5431, 4674,  4536,  364,    0, 2062, 4536, 5061],
    [3044, 2961, 3414, 2692,  2550, 2404, 2062,    0, 2550, 3044],
    [ 837,  813, 1143,  205,     0, 4844, 4536, 2550,    0,  837],
    [   0,  103,  421,  724,   837, 5377, 5061, 3044,  837,    0],
]

# Discrete sailing speed options used by the optimization model.
DEFAULT_SPEED_LEVELS_KNOTS: list[float] = [15.0, 20.0, 25.0]

# Default discrete port-call profile:
# - STANDARD: 12h, 100 USD
# - EXPEDITED: 8h, 200 USD
DEFAULT_PORT_CALL_PROFILE = PortCallProfile(
    duration_labels=["STANDARD", "EXPEDITED"],
    durations_h=[12.0, 8.0],
    costs_usd=[100.0, 200.0],
)

# Default tactical penalty settings used in the base case.
DEFAULT_PENALTIES = PenaltyProfile(
    speed_up_usd=500.0,
    expedited_port_usd=300.0,
    omission_usd=1_000.0,
    swap_usd=750.0,
)


def build_base_instance(
    containers,
    *,
    instance_id: str = "base_instance",
    initial_delay_h: float = 48.0,
    alpha: float = 0.5,
    allow_swap: bool = True,
    swap_ordering_vars_enabled: bool = True,
    max_skip: int = 1,
    max_swap_distance: int = 2,
    speed_levels_knots: list[float] | None = None,
    port_call_profile: PortCallProfile | None = None,
    penalties: PenaltyProfile | None = None,
    port_penalties_usd: dict[int, float] | None = None,
    fuel_price_usd_per_tonne: float = 600.0,
    include_fueleu_penalty: bool = False,
    fueleu_penalty_eur_per_tonne_excess: float = 2_400.0,
    fueleu_eur_to_usd_rate: float = 1.08,
    metadata: dict | None = None,
) -> VSRPInstance:
    """
    Build a canonical `VSRPInstance` using the fixed case-study route.

    This helper combines:
    - the static route template in this file
    - a provided container set
    - default or overridden operational settings

    Parameters
    ----------
    containers
        Container demand records to attach to the instance.
    instance_id : str, default="base_instance"
        Unique instance identifier.
    initial_delay_h : float, default=48.0
        Initial disruption delay at the voyage origin.
    alpha : float, default=0.5
        Objective trade-off weight on service cost.
    allow_swap : bool, default=True
        Whether port swapping is available.
    swap_ordering_vars_enabled : bool, default=True
        Whether the full non-adjacent swap ordering formulation is enabled.
    max_skip : int, default=1
        Maximum number of consecutive planned ports that may be skipped.
    max_swap_distance : int, default=2
        Maximum planned-sequence distance allowed for a swap pair.
    speed_levels_knots : list[float] | None
        Optional override of default speed levels.
    port_call_profile : PortCallProfile | None
        Optional override of the default port-call profile.
    penalties : PenaltyProfile | None
        Optional override of default tactical penalties.
    port_penalties_usd : dict[int, float] | None
        Optional additional port-specific penalties.
    fuel_price_usd_per_tonne : float, default=600.0
        Fuel price stored on the instance so optimization and post-solve
        cost recomputation remain aligned.
    include_fueleu_penalty : bool, default=False
        Whether the FuelEU penalty proxy should be included in the
        operational objective.
    fueleu_penalty_eur_per_tonne_excess : float, default=2400.0
        FuelEU penalty-rate parameter.
    fueleu_eur_to_usd_rate : float, default=1.08
        EUR/USD conversion rate used for FuelEU reporting.
    metadata : dict | None
        Optional experiment metadata.

    Returns
    -------
    VSRPInstance
        Canonical optimization instance.
    """
    return VSRPInstance(
        instance_id=instance_id,
        ports=BASE_PORTS.copy(),
        distance_matrix_nm=[row.copy() for row in BASE_DISTANCE_MATRIX_NM],
        containers=list(containers),
        initial_delay_h=initial_delay_h,
        speed_levels_knots=(
            speed_levels_knots.copy()
            if speed_levels_knots is not None
            else DEFAULT_SPEED_LEVELS_KNOTS.copy()
        ),
        port_call_profile=port_call_profile or DEFAULT_PORT_CALL_PROFILE,
        penalties=penalties or DEFAULT_PENALTIES,
        allow_swap=allow_swap,
        swap_ordering_vars_enabled=swap_ordering_vars_enabled,
        max_skip=max_skip,
        max_swap_distance=max_swap_distance,
        alpha=alpha,
        fuel_price_usd_per_tonne=fuel_price_usd_per_tonne,
        include_fueleu_penalty=include_fueleu_penalty,
        fueleu_penalty_eur_per_tonne_excess=fueleu_penalty_eur_per_tonne_excess,
        fueleu_eur_to_usd_rate=fueleu_eur_to_usd_rate,
        port_penalties_usd=port_penalties_usd or {},
        metadata=metadata or {},
    )