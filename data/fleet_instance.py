from __future__ import annotations

from core.entities import FleetInstance, VesselConfig
from data.base_instance import build_base_instance


def build_fleet_instance(
    vessel_configs: list[VesselConfig],
    *,
    fleet_id: str = "fleet",
    metadata: dict | None = None,
) -> FleetInstance:
    """
    Build a FleetInstance from a list of VesselConfig objects.

    All vessels share the fixed case-study route defined in
    data/base_instance.py but carry vessel-specific operational
    parameters declared in their VesselConfig.

    Parameters
    ----------
    vessel_configs : list[VesselConfig]
        One configuration object per vessel. The vessel order in
        the resulting FleetInstance matches the order of this list.
    fleet_id : str, default="fleet"
        Unique fleet scenario identifier.
    metadata : dict | None, default=None
        Optional fleet-level metadata attached to the FleetInstance.

    Returns
    -------
    FleetInstance
        Ready-to-solve fleet instance with one VSRPInstance per vessel.
    """
    vessel_instances = []

    for v_idx, cfg in enumerate(vessel_configs):
        instance = build_base_instance(
            containers=cfg.containers,
            instance_id=f"{fleet_id}_v{v_idx + 1}_{cfg.vessel_id}",
            initial_delay_h=cfg.initial_delay_h,
            alpha=cfg.alpha,
            allow_swap=cfg.allow_swap,
            max_skip=cfg.max_skip,
            speed_levels_knots=cfg.speed_levels_knots,
            port_penalties_usd=cfg.port_penalties_usd,
            fuel_price_usd_per_tonne=cfg.fuel_price_usd_per_tonne,
            include_fueleu_penalty=cfg.include_fueleu_penalty,
            metadata={
                "vessel_id": cfg.vessel_id,
                "vessel_idx": v_idx,
                "fleet_id": fleet_id,
                **cfg.metadata,
            },
        )
        vessel_instances.append(instance)

    return FleetInstance(
        fleet_id=fleet_id,
        vessel_instances=vessel_instances,
        metadata=metadata or {},
    )


def build_fleet_from_delays(
    vessel_delays_h: list[float],
    containers_per_vessel: list[list],
    *,
    fleet_id: str = "fleet",
    alpha: float = 0.5,
    allow_swap: bool = True,
    max_skip: int = 1,
    port_penalties_usd: dict[int, float] | None = None,
    include_fueleu_penalty: bool = False,
    fuel_price_usd_per_tonne: float = 600.0,
    speed_levels_knots: list[float] | None = None,
    metadata: dict | None = None,
) -> FleetInstance:
    """
    Convenience factory for building a fleet from parallel delay and
    container lists.

    All vessels share the same alpha, swap settings, and port penalties.
    Use build_fleet_instance() with explicit VesselConfig objects when
    vessels require heterogeneous operational parameters.

    Parameters
    ----------
    vessel_delays_h : list[float]
        Initial disruption delay in hours for each vessel.
    containers_per_vessel : list[list[Container]]
        Container demand list for each vessel.
    fleet_id : str, default="fleet"
        Unique fleet scenario identifier.
    alpha : float, default=0.5
        Shared objective trade-off weight applied to all vessels.
    allow_swap : bool, default=True
        Whether port swapping is available for all vessels.
    max_skip : int, default=1
        Maximum consecutive port skips for all vessels.
    port_penalties_usd : dict[int, float] | None, default=None
        Shared port-specific penalties applied to all vessels.
    include_fueleu_penalty : bool, default=False
        Whether the FuelEU penalty proxy is included for all vessels.
    fuel_price_usd_per_tonne : float, default=600.0
        Shared fuel price applied to all vessels.
    speed_levels_knots : list[float] | None, default=None
        Optional shared speed level override for all vessels.
    metadata : dict | None, default=None
        Optional fleet-level metadata.

    Returns
    -------
    FleetInstance
        Ready-to-solve fleet instance.

    Raises
    ------
    ValueError
        If vessel_delays_h and containers_per_vessel have different lengths.
    """
    if len(vessel_delays_h) != len(containers_per_vessel):
        raise ValueError(
            "vessel_delays_h and containers_per_vessel must have "
            "the same length"
        )

    configs = [
        VesselConfig(
            vessel_id=f"V{v_idx + 1}",
            containers=containers,
            initial_delay_h=delay_h,
            alpha=alpha,
            allow_swap=allow_swap,
            max_skip=max_skip,
            speed_levels_knots=speed_levels_knots,
            port_penalties_usd=port_penalties_usd or {},
            include_fueleu_penalty=include_fueleu_penalty,
            fuel_price_usd_per_tonne=fuel_price_usd_per_tonne,
        )
        for v_idx, (delay_h, containers) in enumerate(
            zip(vessel_delays_h, containers_per_vessel)
        )
    ]

    return build_fleet_instance(
        configs,
        fleet_id=fleet_id,
        metadata=metadata,
    )