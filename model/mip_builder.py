# =============================================================================
# Solver-agnostic MIP builder for the Vessel Schedule Recovery Problem (VSRP).
#
# Purpose
# -------
# This module builds the full VSRP mixed-integer programming formulation
# using PuLP as a solver-independent modeling layer.
#
# Main role
# ---------
# The builder exists so that open-source solver backends such as HiGHS
# and CBC can solve the VSRP without relying on the native Xpress API.
#
# Formulation scope
# -----------------
# The formulation is intended to match the native Xpress model as closely
# as possible at the modeling level, including:
# - flow conservation
# - port-call consistency
# - strategy classification
# - non-adjacent swap ordering constraints
# - delayed-container indicators
# - transshipment misconnection logic
# - FuelEU penalty term when enabled
# - weighted operational/service objective
#
# Architectural role
# ------------------
# This file is the open-source modeling engine of the repository.
# It produces a PuLP model that can be exported to MPS and consumed by:
# - HiGHS
# - CBC
#
# This is what makes the repository genuinely multi-solver rather than
# merely Xpress-specific.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.emissions import (
    DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ,
    compute_fueleu_penalty_per_fuel_tonne_usd,
    fuel_consumption_tonnes,
)
from core.entities import VSRPInstance
from core.network import (
    build_network_from_instance,
    get_swap_group_ids,
    get_swap_groups,
)

try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:
    pulp = None
    PULP_AVAILABLE = False


# =============================================================================
# 1. MODEL BUNDLE
# =============================================================================

@dataclass
class PuLPModelBundle:
    """
    Container for a built PuLP model and its variable references.

    This bundle mirrors the structure of the internal Xpress model bundle
    so that the formulation can be reasoned about consistently across
    solver backends.

    Fields
    ------
    model : object
        Underlying PuLP `LpProblem`.
    edges : list
        Feasible edge list used to build the formulation.
    x : list
        Edge-selection variables indexed by edge and duration option.
    w : list
        Port-call choice variables indexed by port and duration option.
    b : list
        Strategy-classification variables indexed by port and strategy.
    y : list
        Delayed-container binary variables.
    o : list
        Misconnected-container binary variables.
    z : dict
        Ordering variables for swap logic.
    swap_active : dict
        Binary activation variables for swap groups.
    planned_arrivals_h : list[float]
        Planned arrival timeline used by approximation-based constraints.
    """
    model: object
    edges: list

    x: list = field(default_factory=list)
    w: list = field(default_factory=list)
    b: list = field(default_factory=list)
    y: list = field(default_factory=list)
    o: list = field(default_factory=list)
    z: dict = field(default_factory=dict)
    swap_active: dict = field(default_factory=dict)

    planned_arrivals_h: list[float] = field(default_factory=list)


# =============================================================================
# 2. BUILDER CLASS
# =============================================================================

class VSRPMIPBuilder:
    """
    Build the VSRP mixed-integer model using PuLP.

    Typical usage
    -------------
    ```python
    builder = VSRPMIPBuilder()
    bundle = builder.build(instance)
    bundle.model.writeMPS("path/to/model.mps")
    ```
    """

    def build(self, instance: VSRPInstance) -> PuLPModelBundle:
        """
        Build and return a full PuLP model bundle for one canonical instance.
        """
        if not PULP_AVAILABLE:
            raise ImportError(
                "PuLP is required for VSRPMIPBuilder. "
                "Install with: pip install pulp"
            )

        from core.network import compute_planned_arrivals

        edges = build_network_from_instance(instance)
        planned_arrivals_h = compute_planned_arrivals(
            distance_matrix_nm=instance.distance_matrix_nm,
            nominal_speed_knots=20.0,
            nominal_port_call_duration_h=(
                instance.port_call_profile.durations_h[0]
            ),
        )

        model = pulp.LpProblem(
            name=instance.instance_id.replace(" ", "_"),
            sense=pulp.LpMinimize,
        )

        n_edges = len(edges)
        n_ports = instance.n_ports
        n_dur = len(instance.port_call_profile.durations_h)
        n_cont = instance.n_containers

        x = [
            [
                pulp.LpVariable(
                    name=f"x_e{e}_d{d}",
                    cat=pulp.constants.LpBinary,
                )
                for d in range(n_dur)
            ]
            for e in range(n_edges)
        ]
        w = [
            [
                pulp.LpVariable(
                    name=f"w_p{p}_d{d}",
                    cat=pulp.constants.LpBinary,
                )
                for d in range(n_dur)
            ]
            for p in range(n_ports)
        ]
        b = [
            [
                pulp.LpVariable(
                    name=f"b_p{p}_s{s}",
                    cat=pulp.constants.LpBinary,
                )
                for s in range(4)
            ]
            for p in range(n_ports)
        ]
        y = [
            pulp.LpVariable(name=f"y_c{c}", cat=pulp.constants.LpBinary)
            for c in range(n_cont)
        ]
        o = [
            pulp.LpVariable(name=f"o_c{c}", cat=pulp.constants.LpBinary)
            for c in range(n_cont)
        ]

        z: dict = {}
        swap_active: dict = {}

        if instance.allow_swap and instance.swap_ordering_vars_enabled:
            interior = list(range(1, n_ports - 1))
            for idx_i, pi in enumerate(interior):
                for pj in interior[idx_i + 1:]:
                    z[(pi, pj)] = pulp.LpVariable(
                        name=f"z_{pi}_{pj}",
                        cat=pulp.constants.LpBinary,
                    )

        bundle = PuLPModelBundle(
            model=model,
            edges=edges,
            x=x,
            w=w,
            b=b,
            y=y,
            o=o,
            z=z,
            swap_active=swap_active,
            planned_arrivals_h=planned_arrivals_h,
        )

        # Swap constraints must be added before strategy constraints
        # because strategy classification reads bundle.swap_active to
        # assign PORT_SWAP to both ports in an active swap pair.
        self._add_flow_constraints(instance, bundle)
        self._add_port_call_constraints(instance, bundle)
        self._add_swap_constraints(instance, bundle)
        self._add_strategy_constraints(instance, bundle)
        self._add_delay_constraints(instance, bundle)
        self._add_misconnection_constraints(instance, bundle)
        self._add_objective(instance, bundle)

        return bundle

    def write_mps(
        self,
        instance: VSRPInstance,
        mps_path: str | Path,
    ) -> PuLPModelBundle:
        """
        Build the model and export it to an MPS file.

        Returns the model bundle so callers may still inspect variables
        and edge structure if needed.
        """
        bundle = self.build(instance)
        bundle.model.writeMPS(str(mps_path))
        return bundle

    # -----------------------------------------------------------------
    # Flow conservation
    # -----------------------------------------------------------------

    def _add_flow_constraints(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Add route flow-conservation constraints.

        This enforces:
        - one origin departure
        - one sink arrival
        - interior flow balance
        - at most one incoming and outgoing arc per interior port
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        n_dur = len(instance.port_call_profile.durations_h)
        n_ports = instance.n_ports

        for p in range(n_ports):
            outgoing = [
                (e_idx, d)
                for e_idx, edge in enumerate(edges)
                for d in range(n_dur)
                if edge.from_port_idx == p
            ]
            incoming = [
                (e_idx, d)
                for e_idx, edge in enumerate(edges)
                for d in range(n_dur)
                if edge.to_port_idx == p
            ]

            if p == 0:
                model += (
                    pulp.lpSum(x[e][d] for e, d in outgoing) == 1,
                    "flow_origin_out",
                )
            elif p == n_ports - 1:
                model += (
                    pulp.lpSum(x[e][d] for e, d in incoming) == 1,
                    "flow_sink_in",
                )
            else:
                model += (
                    pulp.lpSum(x[e][d] for e, d in outgoing)
                    == pulp.lpSum(x[e][d] for e, d in incoming),
                    f"flow_balance_p{p}",
                )

        for p in range(1, n_ports - 1):
            out_all = [
                (e_idx, d)
                for e_idx, edge in enumerate(edges)
                for d in range(n_dur)
                if edge.from_port_idx == p
            ]
            in_all = [
                (e_idx, d)
                for e_idx, edge in enumerate(edges)
                for d in range(n_dur)
                if edge.to_port_idx == p
            ]
            if out_all:
                model += (
                    pulp.lpSum(x[e][d] for e, d in out_all) <= 1,
                    f"flow_out_le1_p{p}",
                )
            if in_all:
                model += (
                    pulp.lpSum(x[e][d] for e, d in in_all) <= 1,
                    f"flow_in_le1_p{p}",
                )

    # -----------------------------------------------------------------
    # Port-call consistency
    # -----------------------------------------------------------------

    def _add_port_call_constraints(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Link discrete port-call choices to route visitation.

        A port can choose at most one duration option, and a duration
        option can only be active if the port is actually visited.
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        w = bundle.w
        n_dur = len(instance.port_call_profile.durations_h)

        for p in range(1, instance.n_ports - 1):
            model += (
                pulp.lpSum(w[p][d] for d in range(n_dur)) <= 1,
                f"portcall_atmost1_p{p}",
            )

            arriving = [
                e_idx for e_idx, edge in enumerate(edges)
                if edge.to_port_idx == p
            ]
            if arriving:
                incoming_flow = pulp.lpSum(
                    pulp.lpSum(x[e][d] for d in range(n_dur))
                    for e in arriving
                )
                model += (
                    pulp.lpSum(w[p][d] for d in range(n_dur)) == incoming_flow,
                    f"portcall_visited_eq_p{p}",
                )
                for d in range(n_dur):
                    model += (
                        w[p][d] <= pulp.lpSum(x[e][d] for e in arriving),
                        f"portcall_link_p{p}_d{d}",
                    )

    # -----------------------------------------------------------------
    # Strategy classification
    # -----------------------------------------------------------------

    def _add_strategy_constraints(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Add strategy-classification constraints.

        Strategy indices
        ----------------
        s = 0 : SPEED_UP
        s = 1 : EXPEDITED_PORT
        s = 2 : PORT_OMISSION
        s = 3 : PORT_SWAP

        Notes
        -----
        - PORT_OMISSION is linked to non-visitation of the port.
        - EXPEDITED_PORT is linked to the expedited duration option.
        - SPEED_UP is linked to a non-swap incoming edge at the fastest speed.
        - PORT_SWAP is linked to the swap-group activation variable for
          both ports participating in the swapped pair.
        """
        from core.network import get_swap_group_port_pair, get_swap_groups

        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        w = bundle.w
        b = bundle.b
        n_dur = len(instance.port_call_profile.durations_h)
        fast_speed = max(instance.speed_levels_knots)

        port_to_swap_active: dict[int, object] = {}

        if instance.allow_swap and instance.swap_ordering_vars_enabled:
            all_swap_groups = get_swap_groups(edges)
            for gid, group_edges in all_swap_groups.items():
                pair = get_swap_group_port_pair(group_edges)
                if pair is not None:
                    i_port, j_port = pair
                    swap_act = bundle.swap_active.get(gid)
                    if swap_act is not None:
                        port_to_swap_active[i_port] = swap_act
                        port_to_swap_active[j_port] = swap_act

        for p in range(1, instance.n_ports - 1):
            visited = pulp.lpSum(w[p][d] for d in range(n_dur))

            model += (b[p][2] == 1 - visited, f"strat_omit_p{p}")

            if n_dur > 1:
                model += (b[p][1] == w[p][1], f"strat_exp_p{p}")
            else:
                model += (b[p][1] == 0, f"strat_exp_p{p}")

            fast_in = [
                e_idx for e_idx, edge in enumerate(edges)
                if edge.to_port_idx == p
                and edge.speed_knots == fast_speed
                and not edge.is_swap
            ]
            if fast_in:
                fast_sum = pulp.lpSum(
                    pulp.lpSum(x[e][d] for d in range(n_dur))
                    for e in fast_in
                )
                model += (b[p][0] <= fast_sum, f"strat_speedup_ub_p{p}")
                model += (
                    b[p][0] >= fast_sum / len(fast_in),
                    f"strat_speedup_lb_p{p}",
                )
            else:
                model += (b[p][0] == 0, f"strat_speedup_zero_p{p}")

            swap_act = port_to_swap_active.get(p)
            if swap_act is not None and instance.allow_swap:
                model += (b[p][3] == swap_act, f"strat_swap_p{p}")
            else:
                model += (b[p][3] == 0, f"strat_swap_zero_p{p}")

    # -----------------------------------------------------------------
    # Non-adjacent swap ordering constraints
    # -----------------------------------------------------------------

    def _add_swap_constraints(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Add non-adjacent swap ordering constraints.

        This formulation enforces:
        - ordering consistency between swapped ports
        - integrity of three-leg swap groups
        - at most one active swap group in the route
        """
        if not instance.allow_swap or not instance.swap_ordering_vars_enabled:
            return

        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        z = bundle.z
        n_dur = len(instance.port_call_profile.durations_h)
        n_ports = instance.n_ports

        swap_groups = get_swap_groups(edges)
        group_ids = get_swap_group_ids(edges)

        if not group_ids:
            return

        interior = list(range(1, n_ports - 1))

        # Antisymmetry
        for idx_i, pi in enumerate(interior):
            for pj in interior[idx_i + 1:]:
                if (pi, pj) in z and (pj, pi) in z:
                    model += (
                        z[(pi, pj)] + z[(pj, pi)] == 1,
                        f"antisym_{pi}_{pj}",
                    )

        # Transitivity
        for pi in interior:
            for pj in interior:
                if pj == pi:
                    continue
                for pk in interior:
                    if pk == pi or pk == pj:
                        continue

                    key_ij = (min(pi, pj), max(pi, pj))
                    key_jk = (min(pj, pk), max(pj, pk))
                    key_ik = (min(pi, pk), max(pi, pk))

                    if key_ij in z and key_jk in z and key_ik in z:
                        zij = z[key_ij] if pi < pj else (1 - z[key_ij])
                        zjk = z[key_jk] if pj < pk else (1 - z[key_jk])
                        zik = z[key_ik] if pi < pk else (1 - z[key_ik])
                        model += (
                            zij + zjk - zik <= 1,
                            f"transit_{pi}_{pj}_{pk}",
                        )

        swap_active_vars = []

        for gid in group_ids:
            group_edges = swap_groups[gid]

            swap_act = pulp.LpVariable(
                name=f"swap_active_g{gid}",
                cat=pulp.constants.LpBinary,
            )
            bundle.swap_active[gid] = swap_act
            swap_active_vars.append(swap_act)

            # Group structural swap legs by (from, to), ignoring speed.
            leg_types: dict[tuple, list[int]] = {}
            for e_idx, edge in enumerate(edges):
                if edge.swap_group_id == gid:
                    key = (edge.from_port_idx, edge.to_port_idx)
                    leg_types.setdefault(key, []).append(e_idx)

            for leg_num, e_indices in enumerate(leg_types.values()):
                leg_sum = pulp.lpSum(
                    pulp.lpSum(x[e][d] for d in range(n_dur))
                    for e in e_indices
                )
                model += (
                    leg_sum >= swap_act,
                    f"swap_g{gid}_leg{leg_num}_lb",
                )
                model += (
                    leg_sum <= swap_act,
                    f"swap_g{gid}_leg{leg_num}_ub",
                )

            # The reverse leg j -> i identifies the swapped pair ordering.
            leg_B = [
                (fp, tp) for (fp, tp) in leg_types if fp > tp
            ]
            if leg_B:
                j_port, i_port = leg_B[0]
                key_ji = (min(i_port, j_port), max(i_port, j_port))
                if key_ji in z:
                    if j_port > i_port:
                        model += (
                            z[key_ji] <= 1 - swap_act,
                            f"swap_g{gid}_order_ub",
                        )
                    else:
                        model += (
                            z[key_ji] >= swap_act,
                            f"swap_g{gid}_order_lb",
                        )

        if swap_active_vars:
            model += (
                pulp.lpSum(swap_active_vars) <= 1,
                "swap_atmost_one_group",
            )

    # -----------------------------------------------------------------
    # Delay constraints
    # -----------------------------------------------------------------

    def _add_delay_constraints(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Add delayed-container indicator constraints using a conservative
        arrival approximation.

        The approximation uses:
        - nominal speed on prior legs
        - selected edge speed on the final leg to destination
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        y = bundle.y
        n_dur = len(instance.port_call_profile.durations_h)
        nominal_port_duration = instance.port_call_profile.durations_h[0]
        nominal_speed = 20.0
        tol = 1e-6

        for c_idx, container in enumerate(instance.containers):
            dest = container.destination_idx
            on_time_edges = []

            for e_idx, edge in enumerate(edges):
                if edge.to_port_idx != dest:
                    continue

                prior_time = self._estimate_prior_leg_time(
                    edge,
                    instance,
                    nominal_speed=nominal_speed,
                    nominal_port_duration=nominal_port_duration,
                )
                arrival_est_h = instance.initial_delay_h + prior_time
                arrival_est_h += (
                    instance.distance_matrix_nm[edge.from_port_idx][dest]
                    / edge.speed_knots
                )
                arrival_est_h += nominal_port_duration

                if arrival_est_h <= container.promised_arrival_h + tol:
                    on_time_edges.append(e_idx)

            if on_time_edges:
                on_time_sum = pulp.lpSum(
                    pulp.lpSum(x[e][d] for d in range(n_dur))
                    for e in on_time_edges
                )
                model += (
                    y[c_idx] >= 1 - on_time_sum,
                    f"delay_lb_c{c_idx}",
                )
            else:
                model += (y[c_idx] == 1, f"delay_forced_c{c_idx}")

    # -----------------------------------------------------------------
    # Misconnection constraints
    # -----------------------------------------------------------------

    def _add_misconnection_constraints(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Add transshipment misconnection logic.

        Direct containers
        -----------------
        A direct container is misconnected if its destination is not reached.

        Transshipment containers
        ------------------------
        A transshipment container is misconnected if it reaches its
        transshipment port after its connecting-service deadline, or if
        its destination is not reached at all.

        Timing uses the same conservative arrival approximation as the
        delay-indicator constraints.
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        y = bundle.y
        o = bundle.o
        n_dur = len(instance.port_call_profile.durations_h)
        nominal_port_duration = instance.port_call_profile.durations_h[0]
        nominal_speed = 20.0
        tol = 1e-6

        for c_idx, container in enumerate(instance.containers):

            if not container.transshipment_port_indices:
                dest = container.destination_idx
                arriving = [
                    e_idx for e_idx, edge in enumerate(edges)
                    if edge.to_port_idx == dest
                ]
                if arriving:
                    reach_sum = pulp.lpSum(
                        pulp.lpSum(x[e][d] for d in range(n_dur))
                        for e in arriving
                    )
                    model += (
                        o[c_idx] >= 1 - reach_sum,
                        f"miscon_direct_lb_c{c_idx}",
                    )
                    model += (
                        o[c_idx] <= 1 - reach_sum + y[c_idx],
                        f"miscon_direct_ub_c{c_idx}",
                    )
                else:
                    model += (
                        o[c_idx] == 1,
                        f"miscon_direct_forced_c{c_idx}",
                    )
                continue

            for t_num, trans_idx in enumerate(
                container.transshipment_port_indices
            ):
                if container.connecting_service_deadline_h is not None:
                    deadline_h = container.connecting_service_deadline_h
                else:
                    deadline_h = (
                        container.promised_arrival_h - nominal_port_duration
                    )

                for e_idx, edge in enumerate(edges):
                    if edge.to_port_idx != trans_idx:
                        continue

                    prior_time = self._estimate_prior_leg_time(
                        edge,
                        instance,
                        nominal_speed=nominal_speed,
                        nominal_port_duration=nominal_port_duration,
                    )
                    arrival_est_h = instance.initial_delay_h + prior_time
                    arrival_est_h += (
                        instance.distance_matrix_nm[
                            edge.from_port_idx
                        ][trans_idx]
                        / edge.speed_knots
                    )
                    arrival_est_h += nominal_port_duration

                    if arrival_est_h > deadline_h + tol:
                        for d in range(n_dur):
                            model += (
                                o[c_idx] >= x[e_idx][d],
                                f"miscon_trans_c{c_idx}_t{t_num}"
                                f"_e{e_idx}_d{d}",
                            )

            dest = container.destination_idx
            dest_arriving = [
                e_idx for e_idx, edge in enumerate(edges)
                if edge.to_port_idx == dest
            ]
            if dest_arriving:
                reach_sum = pulp.lpSum(
                    pulp.lpSum(x[e][d] for d in range(n_dur))
                    for e in dest_arriving
                )
                model += (
                    o[c_idx] >= 1 - reach_sum,
                    f"miscon_trans_dest_lb_c{c_idx}",
                )
            else:
                model += (
                    o[c_idx] == 1,
                    f"miscon_trans_dest_forced_c{c_idx}",
                )

    # -----------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------

    def _add_objective(
        self,
        instance: VSRPInstance,
        bundle: PuLPModelBundle,
    ) -> None:
        """
        Add the weighted operational/service objective.

        Operational terms include:
        - fuel cost
        - port-call handling cost
        - strategy penalties
        - port-specific penalties
        - FuelEU penalty when enabled

        Service terms include:
        - delay penalties
        - misconnection penalties
        """
        edges = bundle.edges
        x = bundle.x
        w = bundle.w
        b = bundle.b
        y = bundle.y
        o = bundle.o
        n_dur = len(instance.port_call_profile.durations_h)

        operational_terms = []
        service_terms = []

        for e_idx, edge in enumerate(edges):
            for d in range(n_dur):
                operational_terms.append(edge.fuel_cost_usd * x[e_idx][d])

        for p in range(1, instance.n_ports - 1):
            for d, cost_usd in enumerate(
                instance.port_call_profile.costs_usd
            ):
                operational_terms.append(cost_usd * w[p][d])

        strategy_penalties = [
            instance.penalties.speed_up_usd,
            instance.penalties.expedited_port_usd,
            instance.penalties.omission_usd,
            instance.penalties.swap_usd,
        ]
        for p in range(instance.n_ports):
            for s, penalty in enumerate(strategy_penalties):
                if s == 3:
                    # PORT_SWAP is charged through swap_active instead of
                    # port-level strategy tags.
                    continue
                operational_terms.append(penalty * b[p][s])

        for swap_act in bundle.swap_active.values():
            operational_terms.append(instance.penalties.swap_usd * swap_act)

        for port_idx, penalty_usd in instance.port_penalties_usd.items():
            for d in range(n_dur):
                operational_terms.append(penalty_usd * w[port_idx][d])

        if instance.include_fueleu_penalty:
            penalty_per_fuel_tonne_usd = (
                compute_fueleu_penalty_per_fuel_tonne_usd(
                    DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ,
                    year=2026,
                    penalty_eur_per_tonne_vlsfo_equiv=(
                        instance.fueleu_penalty_eur_per_tonne_excess
                    ),
                    eur_to_usd_rate=instance.fueleu_eur_to_usd_rate,
                )
            )

            if penalty_per_fuel_tonne_usd > 0:
                for e_idx, edge in enumerate(edges):
                    fuel_t = fuel_consumption_tonnes(
                        distance_nm=instance.distance_matrix_nm[
                            edge.from_port_idx
                        ][edge.to_port_idx],
                        speed_knots=edge.speed_knots,
                    )
                    fueleu_cost = penalty_per_fuel_tonne_usd * fuel_t
                    for d in range(n_dur):
                        operational_terms.append(fueleu_cost * x[e_idx][d])

        for c_idx, container in enumerate(instance.containers):
            service_terms.append(container.penalty_delay * y[c_idx])
            service_terms.append(container.penalty_misconnect * o[c_idx])

        bundle.model += (
            (1.0 - instance.alpha) * pulp.lpSum(operational_terms)
            + instance.alpha * pulp.lpSum(service_terms)
        )

    def _estimate_prior_leg_time(
        self,
        edge,
        instance: VSRPInstance,
        *,
        nominal_speed: float = 20.0,
        nominal_port_duration: float = 12.0,
    ) -> float:
        """
        Estimate cumulative travel time from origin to edge.from_port_idx.

        This helper mirrors the Xpress approximation logic and is used
        in delayed-container and misconnection constraints.

        For forward edges:
            use sequential nominal travel.

        For swap edges:
            use a lightweight path-aware approximation based on whether
            the edge is a forward jump or a reverse leg.
        """
        from_p = edge.from_port_idx
        to_p = edge.to_port_idx

        if not edge.is_swap:
            cumulative = 0.0
            for p in range(from_p):
                dist = instance.distance_matrix_nm[p][p + 1]
                cumulative += dist / nominal_speed + nominal_port_duration
            return cumulative

        if from_p < to_p:
            # Leg A: predecessor(i) -> j
            cumulative = 0.0
            for p in range(from_p):
                dist = instance.distance_matrix_nm[p][p + 1]
                cumulative += dist / nominal_speed + nominal_port_duration
            return cumulative

        # Leg B: j -> i
        i_port = to_p
        j_port = from_p
        pred_i = i_port - 1

        cumulative = 0.0
        for p in range(pred_i):
            dist = instance.distance_matrix_nm[p][p + 1]
            cumulative += dist / nominal_speed + nominal_port_duration

        dist_A = instance.distance_matrix_nm[pred_i][j_port]
        cumulative += dist_A / nominal_speed + nominal_port_duration

        return cumulative