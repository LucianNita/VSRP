# =============================================================================
# Canonical domain entities for the Vessel Schedule Recovery Problem (VSRP).
#
# Design principles
# -----------------
# - Use port indices internally for all optimization-facing objects.
# - Resolve human-readable port names through helper methods.
# - Keep these entities solver-agnostic.
# - Reuse the same canonical instance / solution objects across
#   optimization, validation, experiments, and reporting.
#
# This file defines the shared dataclasses that make the repository
# modular. All solver backends are expected to consume VSRPInstance
# objects and, where possible, return VSRPSolution objects.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. CORE INPUT ENTITIES
# =============================================================================

@dataclass(slots=True)
class Container:
    """
    Canonical container demand record.

    Internal convention
    -------------------
    - `origin_idx`, `destination_idx`, and `transshipment_port_indices`
      are integer indices into `VSRPInstance.ports`.
    - Human-readable port names should be recovered with the helper
      methods defined on this class.

    Parameters
    ----------
    id : str
        Unique container identifier.
    origin_idx : int
        Origin port index.
    destination_idx : int
        Destination port index.
    promised_arrival_h : float
        Promised latest arrival time used in service evaluation.
    penalty_delay : float
        Cost incurred if the container is classified as delayed.
    penalty_misconnect : float
        Cost incurred if the container is classified as misconnected.
    quantity_teu : int, default=1
        Container quantity in TEU.
    transshipment_port_indices : list[int], default=[]
        Intermediate transshipment port indices, if any.
    priority : str, default="MEDIUM"
        Priority label for reporting or future policy extensions.
    connecting_service_deadline_h : float | None, default=None
        Latest arrival time at the transshipment port for the container
        to make its connecting service. `None` for direct containers.
    """
    id: str
    origin_idx: int
    destination_idx: int
    promised_arrival_h: float

    penalty_delay: float
    penalty_misconnect: float

    quantity_teu: int = 1
    transshipment_port_indices: list[int] = field(default_factory=list)
    priority: str = "MEDIUM"
    connecting_service_deadline_h: float | None = None

    def origin_name(self, ports: list[str]) -> str:
        return ports[self.origin_idx]

    def destination_name(self, ports: list[str]) -> str:
        return ports[self.destination_idx]

    def transshipment_names(self, ports: list[str]) -> list[str]:
        return [ports[i] for i in self.transshipment_port_indices]


@dataclass(slots=True)
class Edge:
    """
    Feasible sailing / network arc in the recovery graph.

    Parameters
    ----------
    from_port_idx : int
        Start port index.
    to_port_idx : int
        End port index.
    speed_knots : float
        Sailing speed used on this edge.
    travel_time_h : float
        Sailing time in hours.
    fuel_cost_usd : float
        Fuel cost associated with this edge.
    is_swap : bool, default=False
        Whether the edge belongs to a swap sequence.
    swap_group_id : int | None, default=None
        Identifier of the swap group to which this edge belongs.

        Under the current non-adjacent swap formulation, a swap between
        ports `i` and `j` is represented by three legs sharing the same
        `swap_group_id`:
        - Leg A: predecessor(i) -> j
        - Leg B: j -> i
        - Leg C: i -> successor(j)

        Non-swap edges have `swap_group_id=None`.
    skipped_port_indices : list[int], default=[]
        Planned intermediate ports skipped by traversing this edge.
    """
    from_port_idx: int
    to_port_idx: int
    speed_knots: float
    travel_time_h: float
    fuel_cost_usd: float
    is_swap: bool = False
    swap_group_id: int | None = None
    skipped_port_indices: list[int] = field(default_factory=list)

    def from_name(self, ports: list[str]) -> str:
        return ports[self.from_port_idx]

    def to_name(self, ports: list[str]) -> str:
        return ports[self.to_port_idx]


@dataclass(slots=True)
class PenaltyProfile:
    """
    Operational penalty settings for a single instance or experiment run.

    These penalties enter the operational part of the objective and
    control the relative attractiveness of tactical recovery actions.
    """
    speed_up_usd: float
    expedited_port_usd: float
    omission_usd: float
    swap_usd: float


@dataclass(slots=True)
class PortCallProfile:
    """
    Port service configuration.

    The lists `duration_labels`, `durations_h`, and `costs_usd` are
    aligned by index. For example:
    - index 0 -> STANDARD
    - index 1 -> EXPEDITED

    This allows the optimization model to choose a discrete port-call
    duration option at each visited port.
    """
    duration_labels: list[str]
    durations_h: list[float]
    costs_usd: list[float]


@dataclass(slots=True)
class VSRPInstance:
    """
    Canonical optimization instance.

    This object bundles the full input data required by a solver backend:
    route structure, demand, tactical settings, economic parameters, and
    regulatory options.

    Parameters
    ----------
    instance_id : str
        Unique instance identifier.
    ports : list[str]
        Ordered port sequence of the nominal route.
    distance_matrix_nm : list[list[float]]
        Pairwise port distance matrix in nautical miles.
    containers : list[Container]
        Container demand records.
    initial_delay_h : float
        Initial disruption delay at the voyage origin in hours.
    speed_levels_knots : list[float]
        Discrete sailing speed options available to the model.
    port_call_profile : PortCallProfile
        Discrete port-call duration / cost options.
    penalties : PenaltyProfile
        Tactical penalty settings.
    allow_swap : bool, default=True
        Whether swap edges may be used.
    swap_ordering_vars_enabled : bool, default=True
        Whether the solver should build ordering variables for the
        non-adjacent swap formulation.
    max_skip : int, default=1
        Maximum number of planned ports that may be skipped on a forward edge.
    max_swap_distance : int, default=2
        Maximum planned-sequence distance between ports considered
        swappable in the network generator.
    alpha : float, default=0.5
        Weight on service cost in the objective.
    fuel_price_usd_per_tonne : float, default=600.0
        Fuel price used in edge costs and cost recomputation.
    port_penalties_usd : dict[int, float], default={}
        Optional additional port-specific penalties.
    metadata : dict[str, Any], default={}
        Free-form metadata used by experiments and reporting.
    include_fueleu_penalty : bool, default=False
        Whether the FuelEU penalty proxy is included in the objective.
    fueleu_penalty_eur_per_tonne_excess : float, default=2400.0
        FuelEU penalty rate parameter.
    fueleu_eur_to_usd_rate : float, default=1.08
        EUR/USD conversion rate for FuelEU penalty reporting.
    """
    instance_id: str
    ports: list[str]
    distance_matrix_nm: list[list[float]]
    containers: list[Container]

    initial_delay_h: float

    speed_levels_knots: list[float]
    port_call_profile: PortCallProfile
    penalties: PenaltyProfile

    allow_swap: bool = True
    swap_ordering_vars_enabled: bool = True
    max_skip: int = 1
    max_swap_distance: int = 2
    alpha: float = 0.5

    fuel_price_usd_per_tonne: float = 600.0

    port_penalties_usd: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    include_fueleu_penalty: bool = False
    fueleu_penalty_eur_per_tonne_excess: float = 2_400.0
    fueleu_eur_to_usd_rate: float = 1.08

    @property
    def n_ports(self) -> int:
        return len(self.ports)

    @property
    def n_containers(self) -> int:
        return len(self.containers)

    def port_name(self, port_idx: int) -> str:
        return self.ports[port_idx]

    def port_index(self, port_name: str) -> int:
        return self.ports.index(port_name)


# =============================================================================
# 2. SOLUTION ENTITIES
# =============================================================================

@dataclass(slots=True)
class RouteLeg:
    """
    One leg of the realized vessel route extracted from a solved model.

    This is the canonical route representation used in reporting,
    validation, cost recomputation, and emissions analysis.
    """
    from_port_idx: int
    to_port_idx: int
    speed_knots: float
    speed_label: str
    duration_idx: int
    duration_label: str
    is_swap: bool = False
    swap_group_id: int | None = None

    def from_name(self, ports: list[str]) -> str:
        return ports[self.from_port_idx]

    def to_name(self, ports: list[str]) -> str:
        return ports[self.to_port_idx]


@dataclass(slots=True)
class TimelineEntry:
    """
    Timeline entry for one realized port event.

    Attributes
    ----------
    port_idx : int
        Port index.
    planned_arrival_h : float
        Nominal planned arrival time.
    actual_arrival_h : float
        Reconstructed realized arrival time.
    delay_h : float
        Difference between realized and planned arrival.
    departure_h : float
        Reconstructed departure time after port handling.
    status : str
        Human-readable status label such as ORIGIN, ON_TIME, DELAYED, or EARLY.
    """
    port_idx: int
    planned_arrival_h: float
    actual_arrival_h: float
    delay_h: float
    departure_h: float
    status: str

    def port_name(self, ports: list[str]) -> str:
        return ports[self.port_idx]


@dataclass(slots=True)
class StrategyDecision:
    """
    Strategy tag attached to a port in the extracted canonical solution.

    Examples include:
    - SPEED_UP
    - EXPEDITED_PORT
    - PORT_OMISSION
    - PORT_SWAP
    """
    port_idx: int
    strategy: str

    def port_name(self, ports: list[str]) -> str:
        return ports[self.port_idx]


@dataclass(slots=True)
class ContainerOutcome:
    """
    Post-solve service outcome for one container.

    This is the canonical reporting representation of whether a
    container was delayed and/or misconnected in the solved plan.
    """
    container_id: str
    origin_idx: int
    destination_idx: int
    delayed: bool
    misconnected: bool
    transshipment_port_indices: list[int] = field(default_factory=list)

    def origin_name(self, ports: list[str]) -> str:
        return ports[self.origin_idx]

    def destination_name(self, ports: list[str]) -> str:
        return ports[self.destination_idx]

    def transshipment_names(self, ports: list[str]) -> list[str]:
        return [ports[i] for i in self.transshipment_port_indices]


@dataclass(slots=True)
class SolverStats:
    """
    Solver metadata and runtime statistics.

    This record is backend-agnostic and is attached to every
    `VSRPSolution`, including empty/error solutions.
    """
    solver_name: str
    status: str

    runtime_s: float | None = None
    mip_gap: float | None = None
    best_bound: float | None = None
    time_to_first_feasible_s: float | None = None
    node_count: int | None = None
    iteration_count: int | None = None

    feasible: bool = False
    optimal: bool = False

    raw_status_code: Any = None
    message: str | None = None


@dataclass(slots=True)
class ValidationResult:
    """
    Structured validation summary attached to a solution.

    The validation layer separates different classes of checks so that
    experiments and scripts can inspect them programmatically rather
    than parsing free-form error strings.
    """
    route_valid: bool = False
    route_errors: list[str] = field(default_factory=list)

    strategy_consistent: bool = False
    strategy_warnings: list[str] = field(default_factory=list)

    timeline_monotone: bool = False
    timeline_warnings: list[str] = field(default_factory=list)

    container_valid: bool = False
    container_warnings: list[str] = field(default_factory=list)

    skipped_ports_valid: bool = False
    skipped_port_warnings: list[str] = field(default_factory=list)

    max_constraint_violation: float | None = None
    n_violated_constraints: int | None = None

    overall_valid: bool = False


@dataclass(slots=True)
class EmissionsSummary:
    """
    Solution-level emissions and regulatory-compliance summary.

    This record is attached to canonical solutions after post-solve
    emissions processing and provides the key environmental and
    regulatory outputs used in experiments and reporting.
    """
    total_fuel_t: float | None = None
    total_co2_t: float | None = None
    total_ets_eur: float | None = None
    total_ets_usd: float | None = None
    total_fueleu_penalty_usd: float | None = None
    avg_ghg_gco2eq_per_mj: float | None = None
    fueleu_compliant: bool | None = None
    fueleu_limit_gco2eq_per_mj: float | None = None
    eexi_compliant: bool | None = None
    cii_rating: str | None = None
    attained_cii: float | None = None
    required_cii: float | None = None


@dataclass(slots=True)
class VSRPSolution:
    """
    Canonical solution object returned by solver adapters.

    The goal of this class is to normalize solver outputs into a shared
    structure that can be consumed uniformly by validation, emissions,
    benchmarking, sensitivity analysis, CFA, and reporting.
    """
    instance_id: str

    objective_value: float | None
    route_legs: list[RouteLeg] = field(default_factory=list)

    skipped_port_indices: list[int] = field(default_factory=list)
    swapped_port_indices: list[int] = field(default_factory=list)

    strategy_decisions: list[StrategyDecision] = field(default_factory=list)
    timeline: list[TimelineEntry] = field(default_factory=list)

    container_outcomes: dict[str, ContainerOutcome] = field(default_factory=dict)

    solver_stats: SolverStats | None = None
    validation: ValidationResult | None = None
    emissions: EmissionsSummary | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return bool(self.solver_stats.feasible) if self.solver_stats else False

    @property
    def optimal(self) -> bool:
        return bool(self.solver_stats.optimal) if self.solver_stats else False

    @property
    def n_delayed(self) -> int:
        return sum(1 for c in self.container_outcomes.values() if c.delayed)

    @property
    def n_misconnected(self) -> int:
        return sum(1 for c in self.container_outcomes.values() if c.misconnected)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_port_indices)

    @property
    def n_swapped(self) -> int:
        return len(self.swapped_port_indices)


# =============================================================================
# 3. EXPERIMENT / REPORTING ENTITIES
# =============================================================================

@dataclass(slots=True)
class BenchmarkRecord:
    """
    Flat record for one solver run on one instance.

    This structure is used to simplify conversion from canonical
    solutions into benchmark tables and summaries.
    """
    solver_name: str
    instance_id: str

    available: bool = True
    feasible: bool = False
    optimal: bool = False

    objective_value: float | None = None
    runtime_s: float | None = None
    mip_gap: float | None = None
    best_bound: float | None = None
    time_to_first_feasible_s: float | None = None
    node_count: int | None = None
    iteration_count: int | None = None

    n_delayed: int | None = None
    n_misconnected: int | None = None
    n_skipped: int | None = None
    n_swapped: int | None = None

    total_co2_t: float | None = None
    total_ets_eur: float | None = None

    route_valid: bool | None = None
    strategy_consistent: bool | None = None
    timeline_monotone: bool | None = None
    container_valid: bool | None = None
    skipped_ports_valid: bool | None = None
    max_constraint_violation: float | None = None
    n_violated_constraints: int | None = None

    status: str | None = None
    raw_status_code: Any = None
    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CFADayResult:
    """
    One episode-level result record for CFA training or testing.

    This is a compact experiment-oriented summary rather than a full
    canonical optimization solution.
    """
    day: int
    train_flag: bool
    step_size: float

    objective_value: float | None
    total_missed: float

    n_skipped: int
    n_delayed: int

    estimated_delay_h: float
    effective_delay_h: float

    theta_mean: float
    theta_max: float

    metadata: dict[str, Any] = field(default_factory=dict)

# =============================================================================
# 4. FLEET ENTITIES
# =============================================================================

@dataclass(slots=True)
class VesselConfig:
    """
    Per-vessel configuration for fleet instance construction.

    This is the declarative input specification for one vessel in a
    fleet scenario. It is consumed by fleet factory functions to
    construct per-vessel VSRPInstance objects that share the same
    port network but carry vessel-specific operational parameters.

    Parameters
    ----------
    vessel_id : str
        Unique vessel identifier within the fleet.
    containers : list[Container]
        Container demand records assigned to this vessel.
    initial_delay_h : float, default=0.0
        Initial disruption delay at the voyage origin in hours.
    alpha : float, default=0.5
        Objective trade-off weight on service cost.
    fuel_price_usd_per_tonne : float, default=600.0
        Fuel price used in edge costs and post-solve recomputation.
    port_penalties_usd : dict[int, float], default={}
        Additional port-specific penalties for this vessel.
    speed_levels_knots : list[float] | None, default=None
        Optional override of the default speed levels.
    allow_swap : bool, default=True
        Whether port swapping is available for this vessel.
    max_skip : int, default=1
        Maximum number of consecutive planned ports that may be skipped.
    include_fueleu_penalty : bool, default=False
        Whether the FuelEU penalty proxy is included in the objective.
    metadata : dict[str, Any], default={}
        Free-form vessel-level metadata.
    """
    vessel_id: str
    containers: list[Container]
    initial_delay_h: float = 0.0
    alpha: float = 0.5
    fuel_price_usd_per_tonne: float = 600.0
    port_penalties_usd: dict[int, float] = field(default_factory=dict)
    speed_levels_knots: list[float] | None = None
    allow_swap: bool = True
    max_skip: int = 1
    include_fueleu_penalty: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FleetInstance:
    """
    Fleet-level optimization instance.

    Wraps a list of per-vessel VSRPInstance objects that share the
    same port network and distance matrix but carry vessel-specific
    operational parameters such as initial delay, containers, and
    port penalties.

    The fleet layer is additive over the per-vessel canonical objects.
    All existing single-vessel solver backends, validation, emissions,
    and cost modules operate on the individual VSRPInstance objects
    contained here.

    Parameters
    ----------
    fleet_id : str
        Unique fleet scenario identifier.
    vessel_instances : list[VSRPInstance]
        One canonical VSRPInstance per vessel, in vessel order.
    metadata : dict[str, Any], default={}
        Free-form fleet-level metadata.
    """
    fleet_id: str
    vessel_instances: list["VSRPInstance"]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vessels(self) -> int:
        """Number of vessels in the fleet."""
        return len(self.vessel_instances)

    @property
    def n_ports(self) -> int:
        """Number of ports in the shared route."""
        return self.vessel_instances[0].n_ports if self.vessel_instances else 0

    @property
    def ports(self) -> list[str]:
        """Ordered port sequence of the shared route."""
        return self.vessel_instances[0].ports if self.vessel_instances else []

    @property
    def total_containers(self) -> int:
        """Total number of containers across all vessels."""
        return sum(vi.n_containers for vi in self.vessel_instances)

    def vessel_instance(self, vessel_idx: int) -> "VSRPInstance":
        """Return the VSRPInstance for vessel at position vessel_idx."""
        return self.vessel_instances[vessel_idx]


@dataclass(slots=True)
class FleetSolution:
    """
    Fleet-level solution object.

    Aggregates one canonical VSRPSolution per vessel and exposes
    fleet-level summary properties. The individual vessel solutions
    retain full fidelity and can be consumed by all existing
    single-vessel validation, emissions, and cost modules.

    Parameters
    ----------
    fleet_id : str
        Matches the fleet_id of the originating FleetInstance.
    vessel_solutions : list[VSRPSolution]
        One canonical solution per vessel, in the same order as
        the originating FleetInstance.vessel_instances.
    fleet_objective_value : float | None
        Sum of per-vessel weighted objective values.
        None if any vessel solution did not produce an objective.
    solver_name : str, default="unknown"
        Name of the solver backend used.
    metadata : dict[str, Any], default={}
        Free-form fleet-level metadata.
    """
    fleet_id: str
    vessel_solutions: list["VSRPSolution"]
    fleet_objective_value: float | None
    solver_name: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vessels(self) -> int:
        """Number of vessel solutions in this fleet solution."""
        return len(self.vessel_solutions)

    @property
    def feasible(self) -> bool:
        """True if every vessel solution is feasible."""
        return all(s.feasible for s in self.vessel_solutions)

    @property
    def optimal(self) -> bool:
        """True if every vessel solution is optimal."""
        return all(s.optimal for s in self.vessel_solutions)

    @property
    def total_delayed(self) -> int:
        """Total delayed containers across all vessels."""
        return sum(s.n_delayed for s in self.vessel_solutions)

    @property
    def total_misconnected(self) -> int:
        """Total misconnected containers across all vessels."""
        return sum(s.n_misconnected for s in self.vessel_solutions)

    @property
    def total_skipped(self) -> int:
        """Total skipped ports across all vessels."""
        return sum(s.n_skipped for s in self.vessel_solutions)

    @property
    def total_swapped(self) -> int:
        """Total swapped ports across all vessels."""
        return sum(s.n_swapped for s in self.vessel_solutions)

    @property
    def avg_runtime_s(self) -> float | None:
        """
        Average per-vessel solve runtime in seconds.

        Returns None if no runtime information is available.
        """
        runtimes = [
            s.solver_stats.runtime_s
            for s in self.vessel_solutions
            if s.solver_stats and s.solver_stats.runtime_s is not None
        ]
        return sum(runtimes) / len(runtimes) if runtimes else None

    @property
    def total_runtime_s(self) -> float | None:
        """
        Total wall-clock solve time across all vessels in seconds.

        Returns None if no runtime information is available.
        """
        runtimes = [
            s.solver_stats.runtime_s
            for s in self.vessel_solutions
            if s.solver_stats and s.solver_stats.runtime_s is not None
        ]
        return sum(runtimes) if runtimes else None

    def vessel_solution(self, vessel_idx: int) -> "VSRPSolution":
        """Return the VSRPSolution for vessel at position vessel_idx."""
        return self.vessel_solutions[vessel_idx]


@dataclass(slots=True)
class FleetBenchmarkRecord:
    """
    Flat benchmark record for one solver run on one fleet scenario.

    This structure mirrors BenchmarkRecord at the fleet level and is
    used to build fleet benchmark tables and summaries.
    """
    solver_name: str
    fleet_id: str
    n_vessels: int
    scenario: str

    feasible: bool = False
    optimal: bool = False

    fleet_objective_value: float | None = None
    avg_runtime_s: float | None = None
    total_runtime_s: float | None = None

    total_delayed: int | None = None
    total_misconnected: int | None = None
    total_skipped: int | None = None
    total_swapped: int | None = None

    total_co2_t: float | None = None
    total_ets_eur: float | None = None

    all_routes_valid: bool | None = None
    all_strategies_consistent: bool | None = None

    status: str | None = None
    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
