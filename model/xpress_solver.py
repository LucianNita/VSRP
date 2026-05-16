# =============================================================================
# Xpress solver backend for the Vessel Schedule Recovery Problem (VSRP).
#
# This is the reference native backend in the codebase. It implements:
#   - path flow conservation on the recovery network
#   - non-adjacent port swapping with ordering variables
#   - strategy classification:
#       SPEED_UP / EXPEDITED_PORT / PORT_OMISSION / PORT_SWAP
#   - conservative delayed-container classification
#   - transshipment misconnection logic
#   - weighted operational/service objective
#
# Unlike the export-based HiGHS and CBC adapters, this backend also
# reconstructs a full canonical solution object including:
#   - route legs
#   - skipped and swapped ports
#   - strategy decisions
#   - reporting timeline
#   - container delayed / misconnected outcomes
#   - validation and emissions summaries
# =============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass

import xpress as xp

from core.costs import compute_cost_breakdown, objective_gap_to_reported
from core.emissions import (
    DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ,
    compute_fueleu_penalty_per_fuel_tonne_usd,
    compute_solution_emissions_summary,
    fuel_consumption_tonnes,
)
from core.entities import (
    ContainerOutcome,
    RouteLeg,
    SolverStats,
    StrategyDecision,
    TimelineEntry,
    VSRPInstance,
    VSRPSolution,
)
from core.network import (
    build_network_from_instance,
    compute_planned_arrivals,
    get_swap_group_ids,
    get_swap_groups,
)
from core.validation import validate_solution
from model.base import BaseSolver, SolveOptions, build_empty_solution


# =============================================================================
# 1. INTERNAL MODEL BUNDLE
# =============================================================================

@dataclass(slots=True)
class _XpressModelBundle:
    """
    Internal container for the Xpress model and variable references.
    """
    model: xp.problem
    edges: list

    x: list
    w: list
    b: list
    y: list
    o: list

    # Ordering variables: z[(i, j)] = 1 if interior port i is visited
    # before interior port j, for i < j.
    z: dict

    # Swap-group activation variables populated in _add_swap_constraints.
    swap_active: dict

    planned_arrivals_h: list[float]


# =============================================================================
# 2. XPRESS SOLVER
# =============================================================================

class XpressSolver(BaseSolver):
    solver_name = "Xpress"

    def solve(
        self,
        instance: VSRPInstance,
        options: SolveOptions | None = None,
    ) -> VSRPSolution:
        """
        Build, solve, and extract a canonical VSRP solution using Xpress.

        This is the full-feature backend in the codebase: unlike the
        export-based HiGHS and CBC adapters, it reconstructs route legs,
        strategy decisions, timeline entries, container outcomes,
        validation results, and emissions summaries.
        """
        options = options or SolveOptions()

        try:
            bundle = self._build_model(instance, options)
            self._apply_solver_options(bundle.model, options)

            # Capture time to first feasible solution via incumbent callback.
            first_feasible_time: list[float] = []
            t0 = time.perf_counter()

            def _on_solution(prob):
                """
                Record the elapsed time of the first incumbent found.
                """
                if not first_feasible_time:
                    first_feasible_time.append(time.perf_counter() - t0)

            try:
                bundle.model.addcbsolution(_on_solution, None, 0)
            except Exception:
                # Some Xpress builds may not support this callback
                # interface cleanly; if registration fails, keep the
                # solve path running and report no timing value.
                pass

            bundle.model.solve()
            runtime_s = time.perf_counter() - t0
            time_to_first = (
                first_feasible_time[0] if first_feasible_time else None
            )

            status_code = bundle.model.attributes.solvestatus
            status_text = self._map_status(status_code)

            if status_code not in (1, 2, 3):
                return build_empty_solution(
                    instance=instance,
                    solver_name=self.solver_name,
                    status=status_text,
                    feasible=False,
                    optimal=False,
                    runtime_s=runtime_s,
                    raw_status_code=status_code,
                    message="No feasible solution found by Xpress",
                )

            try:
                objective_value = float(bundle.model.attributes.objval)
            except Exception:
                objective_value = None

            solution = self._extract_solution(
                instance=instance,
                bundle=bundle,
                objective_value=objective_value,
                runtime_s=runtime_s,
                raw_status_code=status_code,
                status_text=status_text,
                time_to_first_feasible_s=time_to_first,
            )

            solution.validation = validate_solution(instance, solution)
            solution.emissions = compute_solution_emissions_summary(
                instance=instance,
                solution=solution,
                year=2026,
            )

            gap_to_reported = objective_gap_to_reported(instance, solution)
            solution.metadata["objective_recompute_abs_gap"] = gap_to_reported

            cost_breakdown = compute_cost_breakdown(instance, solution)
            solution.metadata["cost_breakdown"] = {
                "fuel_cost_usd": cost_breakdown.fuel_cost_usd,
                "port_call_cost_usd": cost_breakdown.port_call_cost_usd,
                "strategy_penalty_usd": cost_breakdown.strategy_penalty_usd,
                "port_penalty_cost_usd": cost_breakdown.port_penalty_cost_usd,
                "fueleu_penalty_usd": cost_breakdown.fueleu_penalty_usd,
                "delay_cost_usd": cost_breakdown.delay_cost_usd,
                "misconnection_cost_usd": cost_breakdown.misconnection_cost_usd,
                "operational_cost_usd": cost_breakdown.operational_cost_usd,
                "service_cost_usd": cost_breakdown.service_cost_usd,
                "weighted_objective_usd": cost_breakdown.weighted_objective_usd,
            }

            return solution

        except Exception as exc:
            return build_empty_solution(
                instance=instance,
                solver_name=self.solver_name,
                status="ERROR",
                feasible=False,
                optimal=False,
                runtime_s=None,
                raw_status_code=None,
                message=str(exc),
            )

    # -----------------------------------------------------------------
    # Model building
    # -----------------------------------------------------------------

    def _build_model(
        self,
        instance: VSRPInstance,
        options: SolveOptions,
    ) -> _XpressModelBundle:
        """
        Build the full Xpress model and return the internal model bundle.
        """
        edges = build_network_from_instance(instance)

        planned_arrivals_h = compute_planned_arrivals(
            distance_matrix_nm=instance.distance_matrix_nm,
            nominal_speed_knots=20.0,
            nominal_port_call_duration_h=instance.port_call_profile.durations_h[0],
        )

        model = xp.problem()
        model.controls.outputlog = 1 if options.log_to_console else 0

        n_edges = len(edges)
        n_ports = instance.n_ports
        n_dur = len(instance.port_call_profile.durations_h)
        n_cont = instance.n_containers
        n_strat = 4

        x = [
            [
                model.addVariable(vartype=xp.binary, name=f"x_e{e}_d{d}")
                for d in range(n_dur)
            ]
            for e in range(n_edges)
        ]
        w = [
            [
                model.addVariable(vartype=xp.binary, name=f"w_p{p}_d{d}")
                for d in range(n_dur)
            ]
            for p in range(n_ports)
        ]
        b = [
            [
                model.addVariable(vartype=xp.binary, name=f"b_p{p}_s{s}")
                for s in range(n_strat)
            ]
            for p in range(n_ports)
        ]
        y = [
            model.addVariable(vartype=xp.binary, name=f"y_c{c}")
            for c in range(n_cont)
        ]
        o = [
            model.addVariable(vartype=xp.binary, name=f"o_c{c}")
            for c in range(n_cont)
        ]

        z: dict = {}
        if instance.allow_swap and instance.swap_ordering_vars_enabled:
            interior = list(range(1, n_ports - 1))
            for idx_i, pi in enumerate(interior):
                for pj in interior[idx_i + 1:]:
                    z[(pi, pj)] = model.addVariable(
                        vartype=xp.binary,
                        name=f"z_{pi}_{pj}",
                    )

        bundle = _XpressModelBundle(
            model=model,
            edges=edges,
            x=x,
            w=w,
            b=b,
            y=y,
            o=o,
            z=z,
            swap_active={},
            planned_arrivals_h=planned_arrivals_h,
        )

        # Swap constraints must be added before strategy constraints
        # because strategy classification uses bundle.swap_active to
        # assign PORT_SWAP to both ports in an active swap pair.
        self._add_flow_constraints(instance, bundle)
        self._add_port_call_constraints(instance, bundle)
        self._add_swap_constraints(instance, bundle)
        self._add_strategy_constraints(instance, bundle)
        self._add_delay_constraints(instance, bundle)
        self._add_misconnection_constraints(instance, bundle)
        self._add_objective(instance, bundle)

        return bundle

    def _apply_solver_options(self, model, options):
        """
        Apply common solve controls to the Xpress model.
        """
        model.controls.maxtime = options.time_limit_s
        model.controls.miprelstop = options.mip_gap

    # -----------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------

    def _add_flow_constraints(self, instance, bundle):
        """
        Add path flow-conservation constraints over selected edges.

        Subtours are ruled out by the network structure together with:
        - one origin departure
        - one sink arrival
        - interior flow balance
        - at-most-one in / out at interior ports
        - at-most-one active swap group
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        n_dur = len(instance.port_call_profile.durations_h)

        for p in range(instance.n_ports):
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
                model.addConstraint(
                    xp.Sum(x[e][d] for e, d in outgoing) == 1
                )
            elif p == instance.n_ports - 1:
                model.addConstraint(
                    xp.Sum(x[e][d] for e, d in incoming) == 1
                )
            else:
                model.addConstraint(
                    xp.Sum(x[e][d] for e, d in outgoing)
                    == xp.Sum(x[e][d] for e, d in incoming)
                )

        for p in range(1, instance.n_ports - 1):
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
                model.addConstraint(
                    xp.Sum(x[e][d] for e, d in out_all) <= 1
                )
            if in_all:
                model.addConstraint(
                    xp.Sum(x[e][d] for e, d in in_all) <= 1
                )

    def _add_port_call_constraints(self, instance, bundle):
        """
        Link port-call duration choices to route visitation.
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        w = bundle.w
        n_dur = len(instance.port_call_profile.durations_h)

        for p in range(1, instance.n_ports - 1):
            model.addConstraint(
                xp.Sum(w[p][d] for d in range(n_dur)) <= 1
            )
            arriving_edges = [
                e_idx
                for e_idx, edge in enumerate(edges)
                if edge.to_port_idx == p
            ]
            if arriving_edges:
                # Force w sum to equal total incoming flow so that the port
                # visitation indicator is tight in both directions. Without
                # this equality, w variables can remain zero even when the
                # port is visited, causing PORT_OMISSION to fire incorrectly.
                incoming_flow = xp.Sum(
                    xp.Sum(x[e][d] for d in range(n_dur))
                    for e in arriving_edges
                )
                model.addConstraint(
                    xp.Sum(w[p][d] for d in range(n_dur)) == incoming_flow
                )
                for d in range(n_dur):
                    model.addConstraint(
                        w[p][d] <= xp.Sum(x[e][d] for e in arriving_edges)
                    )

    def _add_strategy_constraints(
        self,
        instance,
        bundle,
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
        - EXPEDITED_PORT is linked to the expedited port-call duration.
        - SPEED_UP is linked to arrival on a non-swap incoming edge at
          the fastest available sailing speed.
        - PORT_SWAP is linked directly to the swap-group activation
          variable for both ports participating in the swapped pair.

        This method assumes swap constraints have already been added so
        that bundle.swap_active has been populated.
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
            visited = xp.Sum(w[p][d] for d in range(n_dur))

            # PORT_OMISSION
            model.addConstraint(b[p][2] == 1 - visited)

            # EXPEDITED_PORT
            if n_dur > 1:
                model.addConstraint(b[p][1] == w[p][1])
            else:
                model.addConstraint(b[p][1] == 0)

            # SPEED_UP
            fast_in = [
                e_idx
                for e_idx, edge in enumerate(edges)
                if edge.to_port_idx == p
                and edge.speed_knots == fast_speed
                and not edge.is_swap
            ]
            if fast_in:
                fast_sum = xp.Sum(
                    xp.Sum(x[e][d] for d in range(n_dur))
                    for e in fast_in
                )
                model.addConstraint(b[p][0] <= fast_sum)
                model.addConstraint(
                    b[p][0] >= fast_sum / len(fast_in)
                )
            else:
                model.addConstraint(b[p][0] == 0)

            # PORT_SWAP
            swap_act = port_to_swap_active.get(p)
            if swap_act is not None and instance.allow_swap:
                model.addConstraint(b[p][3] == swap_act)
            else:
                model.addConstraint(b[p][3] == 0)

    def _add_swap_constraints(self, instance, bundle):
        """
        Add non-adjacent swap ordering constraints.

        For each swap group:
        - all three swap legs must be selected together or not at all
        - ordering variables are linked to the swap activation
        - at most one swap group may be active overall

        Ordering variable semantics
        ---------------------------
        z[(i, j)] = 1  =>  port i is visited before port j
        z[(i, j)] = 0  =>  port j is visited before port i
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
                    model.addConstraint(z[(pi, pj)] + z[(pj, pi)] == 1)
                elif (pi, pj) in z:
                    # Reverse key is not explicitly stored because only
                    # i < j variable pairs are created.
                    pass

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
                        model.addConstraint(zij + zjk - zik <= 1)

        swap_active_vars = []

        for gid in group_ids:
            group_edges = swap_groups[gid]

            swap_active = model.addVariable(
                vartype=xp.binary,
                name=f"swap_active_g{gid}",
            )
            bundle.swap_active[gid] = swap_active
            swap_active_vars.append(swap_active)

            # Group swap legs by (from_port_idx, to_port_idx), ignoring speed.
            leg_types: dict[tuple, list[int]] = {}
            for e_idx, edge in enumerate(edges):
                if edge.swap_group_id == gid:
                    key = (edge.from_port_idx, edge.to_port_idx)
                    leg_types.setdefault(key, []).append(e_idx)

            # Exactly one speed on each structural leg when the group is active.
            for e_indices in leg_types.values():
                leg_sum = xp.Sum(
                    xp.Sum(x[e][d] for d in range(n_dur))
                    for e in e_indices
                )
                model.addConstraint(leg_sum >= swap_active)
                model.addConstraint(leg_sum <= swap_active)

            # Link group activation to ordering reversal using the reverse leg.
            leg_B_candidates = [
                (fp, tp)
                for (fp, tp) in leg_types
                if fp > tp
            ]
            if leg_B_candidates:
                j_port, i_port = leg_B_candidates[0]
                key_ji = (min(i_port, j_port), max(i_port, j_port))
                if key_ji in z:
                    if j_port > i_port:
                        model.addConstraint(z[key_ji] <= 1 - swap_active)
                    else:
                        model.addConstraint(z[key_ji] >= swap_active)

        if swap_active_vars:
            model.addConstraint(xp.Sum(swap_active_vars) <= 1)

    def _add_delay_constraints(
        self,
        instance,
        bundle,
    ) -> None:
        """
        Add binary delayed-indicator constraints for containers.

        Arrival classification uses a conservative approximation:
        - nominal speed on prior legs
        - actual selected edge speed on the final leg to destination

        This approximation is aligned with the promised-arrival
        calibration used in instance generation, where promised arrivals
        are offset by the expected initial delay under nominal travel.

        The goal is to classify whether a selected route pattern can
        still be considered on-time relative to a promised arrival
        without introducing a full exact container timing model.
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
                on_time_sum = xp.Sum(
                    xp.Sum(x[e][d] for d in range(n_dur))
                    for e in on_time_edges
                )
                model.addConstraint(y[c_idx] >= 1 - on_time_sum)
            else:
                model.addConstraint(y[c_idx] == 1)

    def _add_misconnection_constraints(self, instance, bundle):
        """
        Add transshipment misconnection logic.

        Direct containers
        -----------------
        A direct container is misconnected if its destination is not
        reached. If the destination is reached, misconnection can only
        remain active when delay is active.

        Transshipment containers
        ------------------------
        A transshipment container is misconnected if the vessel reaches
        its transshipment port after the connecting-service deadline, or
        if the destination is not reached at all.

        Arrival estimates use the same conservative timing approximation
        as delayed-container classification.
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

            # Direct containers
            if not container.transshipment_port_indices:
                dest = container.destination_idx
                arriving = [
                    e_idx
                    for e_idx, edge in enumerate(edges)
                    if edge.to_port_idx == dest
                ]
                if arriving:
                    reach_sum = xp.Sum(
                        xp.Sum(x[e][d] for d in range(n_dur))
                        for e in arriving
                    )
                    model.addConstraint(o[c_idx] >= 1 - reach_sum)
                    model.addConstraint(
                        o[c_idx] <= 1 - reach_sum + y[c_idx]
                    )
                else:
                    model.addConstraint(o[c_idx] == 1)
                continue

            # Transshipment containers
            for trans_idx in container.transshipment_port_indices:
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
                        instance.distance_matrix_nm[edge.from_port_idx][trans_idx]
                        / edge.speed_knots
                    )
                    arrival_est_h += nominal_port_duration

                    if arrival_est_h > deadline_h + tol:
                        for d in range(n_dur):
                            model.addConstraint(o[c_idx] >= x[e_idx][d])

            dest = container.destination_idx
            dest_arriving = [
                e_idx
                for e_idx, edge in enumerate(edges)
                if edge.to_port_idx == dest
            ]
            if dest_arriving:
                reach_sum = xp.Sum(
                    xp.Sum(x[e][d] for d in range(n_dur))
                    for e in dest_arriving
                )
                model.addConstraint(o[c_idx] >= 1 - reach_sum)
            else:
                model.addConstraint(o[c_idx] == 1)

    # -----------------------------------------------------------------
    # Objective
    # -----------------------------------------------------------------

    def _add_objective(self, instance, bundle):
        """
        Add the weighted operational/service objective.

        Operational terms include:
        - fuel cost
        - port-call handling cost
        - strategy penalties
        - port-specific penalties
        - FuelEU penalty (when enabled)

        Service terms include:
        - delay penalties
        - misconnection penalties
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        w = bundle.w
        b = bundle.b
        y = bundle.y
        o = bundle.o
        n_dur = len(instance.port_call_profile.durations_h)

        operational_terms = []
        service_terms = []

        # 1. Fuel cost
        for e_idx, edge in enumerate(edges):
            for d in range(n_dur):
                operational_terms.append(edge.fuel_cost_usd * x[e_idx][d])

        # 2. Port-call handling cost
        for p in range(1, instance.n_ports - 1):
            for d, cost_usd in enumerate(instance.port_call_profile.costs_usd):
                operational_terms.append(cost_usd * w[p][d])

        # 3. Strategy penalties
        strategy_penalties = [
            instance.penalties.speed_up_usd,
            instance.penalties.expedited_port_usd,
            instance.penalties.omission_usd,
            instance.penalties.swap_usd,
        ]
        for p in range(instance.n_ports):
            for s, penalty in enumerate(strategy_penalties):
                if s == 3:
                    # PORT_SWAP is charged at the active swap-group level.
                    continue
                operational_terms.append(penalty * b[p][s])

        # Charge swap penalty once per active swap group.
        # PORT_SWAP tags are reported per swapped port in the extracted
        # canonical solution, but the optimization model charges the
        # structural swap action once at the group level.
        for swap_act in bundle.swap_active.values():
            operational_terms.append(instance.penalties.swap_usd * swap_act)

        # 4. Port-specific penalties
        for port_idx, penalty_usd in instance.port_penalties_usd.items():
            for d in range(n_dur):
                operational_terms.append(penalty_usd * w[port_idx][d])

        # 5. FuelEU penalty
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

        # 6. Service cost
        for c_idx, container in enumerate(instance.containers):
            service_terms.append(container.penalty_delay * y[c_idx])
            service_terms.append(container.penalty_misconnect * o[c_idx])

        objective = (
            (1.0 - instance.alpha) * xp.Sum(operational_terms)
            + instance.alpha * xp.Sum(service_terms)
        )
        model.setObjective(objective, sense=xp.minimize)

    # -----------------------------------------------------------------
    # Solution extraction
    # -----------------------------------------------------------------

    def _speed_label(self, speed: float, levels: list[float]) -> str:
        """
        Return a robust range-based speed label for arbitrary speed sets.
        """
        min_s = min(levels)
        max_s = max(levels)
        span = max_s - min_s

        if span < 1e-6:
            return f"{speed:.0f}kn"

        ratio = (speed - min_s) / span
        if ratio <= 0.33:
            return f"SLOW_{speed:.0f}"
        if ratio >= 0.67:
            return f"FAST_{speed:.0f}"
        return f"NORMAL_{speed:.0f}"

    def _extract_solution(
        self,
        instance,
        bundle,
        objective_value,
        runtime_s,
        raw_status_code,
        status_text,
        time_to_first_feasible_s: float | None = None,
    ):
        """
        Extract a canonical VSRPSolution from the solved Xpress model.

        Responsibilities
        ----------------
        - recover selected route legs
        - derive skipped and swapped ports
        - recover strategy decisions
        - construct a simple realised timeline
        - attach container outcome indicators from model variables

        Notes
        -----
        - swapped_port_indices are derived from active swap-group pairs,
          not from all swap legs individually, to avoid falsely tagging
          continuation ports in a swap sequence.
        - PORT_SWAP strategy tags are ensured for both swapped ports in
          the canonical extracted solution even when the model-side
          strategy variables are only partially informative.
        """
        model = bundle.model
        edges = bundle.edges
        x = bundle.x
        b = bundle.b
        y = bundle.y
        o = bundle.o
        n_dur = len(instance.port_call_profile.durations_h)

        duration_labels = instance.port_call_profile.duration_labels
        strategy_labels = [
            "SPEED_UP",
            "EXPEDITED_PORT",
            "PORT_OMISSION",
            "PORT_SWAP",
        ]

        selected_edges = []
        for e_idx, edge in enumerate(edges):
            for d in range(n_dur):
                if model.getSolution(x[e_idx][d]) > 0.5:
                    selected_edges.append((e_idx, d, edge))

        route_legs = self._reconstruct_route(
            instance,
            selected_edges,
            duration_labels,
        )

        visited = {route_legs[0].from_port_idx} if route_legs else set()
        for leg in route_legs:
            visited.add(leg.to_port_idx)

        skipped_port_indices = [
            p
            for p in range(1, instance.n_ports - 1)
            if p not in visited
        ]

        from core.network import get_swap_group_port_pair, get_swap_groups

        # Derive swapped ports from active swap-group pairs rather than
        # from all swap legs individually. This avoids incorrectly
        # tagging continuation ports in a swap sequence as swapped ports.
        swapped_port_indices: list[int] = []

        if instance.allow_swap:
            active_swap_group_ids = {
                leg.swap_group_id
                for leg in route_legs
                if leg.is_swap and leg.swap_group_id is not None
            }

            if active_swap_group_ids:
                all_swap_groups = get_swap_groups(edges)

                for gid in active_swap_group_ids:
                    group_edges = all_swap_groups.get(gid, [])
                    pair = get_swap_group_port_pair(group_edges)
                    if pair is not None:
                        i_port, j_port = pair
                        for p in (i_port, j_port):
                            if (
                                p != 0
                                and p != instance.n_ports - 1
                                and p not in swapped_port_indices
                            ):
                                swapped_port_indices.append(p)

        strategy_decisions = []
        for p in range(instance.n_ports):
            for s in range(4):
                if model.getSolution(b[p][s]) > 0.5:
                    strategy_decisions.append(
                        StrategyDecision(
                            port_idx=p,
                            strategy=strategy_labels[s],
                        )
                    )

        # Ensure both ports in an active swap pair are explicitly tagged.
        existing_swap_ports = {
            s.port_idx
            for s in strategy_decisions
            if s.strategy == "PORT_SWAP"
        }
        for p in swapped_port_indices:
            if p not in existing_swap_ports:
                strategy_decisions.append(
                    StrategyDecision(port_idx=p, strategy="PORT_SWAP")
                )

        timeline = self._build_timeline(
            instance,
            route_legs,
            skipped_port_indices,
        )

        container_outcomes = {}
        for c_idx, container in enumerate(instance.containers):
            container_outcomes[container.id] = ContainerOutcome(
                container_id=container.id,
                origin_idx=container.origin_idx,
                destination_idx=container.destination_idx,
                delayed=(model.getSolution(y[c_idx]) > 0.5),
                misconnected=(model.getSolution(o[c_idx]) > 0.5),
                transshipment_port_indices=list(
                    container.transshipment_port_indices
                ),
            )

        best_bound = self._extract_best_bound(model)
        mip_gap = self._extract_mip_gap(model)
        node_count = self._extract_node_count(model)
        iteration_count = self._extract_iteration_count(model)

        is_optimal = (raw_status_code == 1)
        if mip_gap is not None and mip_gap <= 1e-9:
            is_optimal = True

        solver_stats = SolverStats(
            solver_name=self.solver_name,
            status=status_text,
            runtime_s=runtime_s,
            mip_gap=mip_gap,
            best_bound=best_bound,
            time_to_first_feasible_s=time_to_first_feasible_s,
            node_count=node_count,
            iteration_count=iteration_count,
            feasible=True,
            optimal=is_optimal,
            raw_status_code=raw_status_code,
            message=None,
        )

        return VSRPSolution(
            instance_id=instance.instance_id,
            objective_value=objective_value,
            route_legs=route_legs,
            skipped_port_indices=skipped_port_indices,
            swapped_port_indices=swapped_port_indices,
            strategy_decisions=strategy_decisions,
            timeline=timeline,
            container_outcomes=container_outcomes,
            solver_stats=solver_stats,
            validation=None,
            emissions=None,
            metadata={
                "partial_solution_backend": False,
                "export_based_backend": False,
                "objective_extracted": objective_value is not None,
                "best_bound_extracted": best_bound is not None,
                "mip_gap_extracted": mip_gap is not None,
            },
        )

    def _reconstruct_route(self, instance, selected_edges, duration_labels):
        """
        Reconstruct the selected path by adjacency traversal from origin.

        Swap groups are handled by following realised edge adjacency
        rather than planned route order.
        """
        if not selected_edges:
            return []

        outgoing_map: dict[int, list[tuple]] = {}
        for e_idx, d, edge in selected_edges:
            outgoing_map.setdefault(edge.from_port_idx, []).append((d, edge))

        route_legs: list[RouteLeg] = []
        current = 0
        seen_from: set[int] = set()

        while current in outgoing_map:
            if current in seen_from:
                break
            seen_from.add(current)

            d, edge = outgoing_map[current][0]

            route_legs.append(
                RouteLeg(
                    from_port_idx=edge.from_port_idx,
                    to_port_idx=edge.to_port_idx,
                    speed_knots=edge.speed_knots,
                    speed_label=self._speed_label(
                        edge.speed_knots,
                        instance.speed_levels_knots,
                    ),
                    duration_idx=d,
                    duration_label=duration_labels[d],
                    is_swap=edge.is_swap,
                    swap_group_id=edge.swap_group_id,
                )
            )

            current = edge.to_port_idx
            if current == instance.n_ports - 1:
                break

        return route_legs
    
    def _build_timeline(self, instance, route_legs, skipped_port_indices):
        """
        Reconstruct a simple reporting timeline from selected route legs.

        This timeline is intended for reporting and validation, not as an
        exact re-expression of the conservative timing approximation used
        in delayed and misconnected container classification.
        """
        if not route_legs:
            return []

        planned_arrivals = compute_planned_arrivals(
            distance_matrix_nm=instance.distance_matrix_nm,
            nominal_speed_knots=20.0,
            nominal_port_call_duration_h=instance.port_call_profile.durations_h[0],
        )

        timeline = []
        current_port = route_legs[0].from_port_idx
        current_arrival_h = instance.initial_delay_h
        current_departure_h = (
            current_arrival_h
            + instance.port_call_profile.durations_h[0]
        )

        timeline.append(
            TimelineEntry(
                port_idx=current_port,
                planned_arrival_h=planned_arrivals[current_port],
                actual_arrival_h=current_arrival_h,
                delay_h=current_arrival_h - planned_arrivals[current_port],
                departure_h=current_departure_h,
                status="ORIGIN",
            )
        )

        current_time = current_departure_h

        for leg in route_legs:
            travel_h = (
                instance.distance_matrix_nm[leg.from_port_idx][leg.to_port_idx]
                / leg.speed_knots
            )
            arrival_h = current_time + travel_h
            departure_h = (
                arrival_h
                + instance.port_call_profile.durations_h[leg.duration_idx]
            )
            delay_h = arrival_h - planned_arrivals[leg.to_port_idx]

            if delay_h > 1.0:
                status = "DELAYED"
            elif delay_h < -1.0:
                status = "EARLY"
            else:
                status = "ON_TIME"

            timeline.append(
                TimelineEntry(
                    port_idx=leg.to_port_idx,
                    planned_arrival_h=planned_arrivals[leg.to_port_idx],
                    actual_arrival_h=arrival_h,
                    delay_h=delay_h,
                    departure_h=departure_h,
                    status=status,
                )
            )
            current_time = departure_h

        return timeline

    # -----------------------------------------------------------------
    # Xpress attribute helpers
    # -----------------------------------------------------------------

    def _safe_get_attr(self, model, attr_name):
        """
        Safely read an Xpress model attribute, returning None on failure.
        """
        try:
            return getattr(model.attributes, attr_name)
        except Exception:
            return None

    def _extract_best_bound(self, model):
        """
        Extract the best available bound reported by Xpress.
        """
        for name in ["bestbound", "mipbestbound", "bestdualbound"]:
            value = self._safe_get_attr(model, name)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        return None

    def _extract_mip_gap(self, model):
        """
        Extract the relative MIP gap, falling back to objective/bound
        reconstruction when needed.
        """
        for name in ["miprelgap", "relmipgap"]:
            value = self._safe_get_attr(model, name)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

        objval = self._safe_get_attr(model, "objval")
        best_bound = self._extract_best_bound(model)
        try:
            if objval is not None and best_bound is not None:
                objval = float(objval)
                best_bound = float(best_bound)
                if abs(objval) <= 1e-12 and abs(best_bound) <= 1e-12:
                    return 0.0
                if abs(objval) > 1e-12:
                    return abs(objval - best_bound) / abs(objval)
        except Exception:
            pass
        return None

    def _extract_node_count(self, model):
        """
        Extract node count from whichever Xpress attribute is available.
        """
        for name in ["nodes", "mipnodes", "nodecount"]:
            value = self._safe_get_attr(model, name)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return None

    def _extract_iteration_count(self, model):
        """
        Extract iteration count from whichever Xpress attribute is available.
        """
        for name in ["simplexiter", "lpiterations", "bariter", "iters"]:
            value = self._safe_get_attr(model, name)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return None

    def _map_status(self, status_code):
        """
        Map selected Xpress solve status codes to human-readable text.
        """
        return {
            1: "OPTIMAL",
            2: "FEASIBLE",
            3: "MIP_SOLUTION",
        }.get(status_code, f"STATUS_{status_code}")

    def _estimate_prior_leg_time(
        self,
        edge,
        instance,
        *,
        nominal_speed: float = 20.0,
        nominal_port_duration: float = 12.0,
    ) -> float:
        """
        Estimate prior-route travel time to edge.from_port_idx under the
        conservative timing approximation used in delayed and
        misconnected container classification.

        For forward edges:
            use sequential nominal travel through the planned route.

        For swap edges:
            use a path-aware estimate based on swap-leg type.

        Swap leg interpretation
        -----------------------
        Leg A: pred_i -> j
            Forward jump into the later swapped port.
            Prior path is sequential from origin to pred_i.

        Leg B: j -> i
            Reverse leg returning from j to i.
            Prior path is sequential to pred_i, then Leg A to j.

        Leg C is not explicitly distinguished here because the current
        approximation only needs to estimate cumulative travel up to
        edge.from_port_idx for the selected edge being evaluated.
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
            # Leg A: pred_i -> j
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