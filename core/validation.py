# =============================================================================
# Solver-agnostic validation for VSRP instances and solutions.
#
# Validation philosophy
# ---------------------
# Hard validation checks focus on:
#   - route structure and continuity
#   - strategy consistency
#   - timeline monotonicity / internal consistency
#   - container record consistency
#   - skipped-port derivation consistency
#   - simple reconstructed structural constraint residuals
#
# Delay-classification mismatches are NOT treated as hard numerical
# violations in the current codebase. The delayed indicator in the MIP
# is based on a conservative timing approximation, while the post-solve
# timeline reconstructs a fuller realised route chronology. Those two
# views can diverge without implying model infeasibility.
# =============================================================================

from __future__ import annotations

from core.entities import ValidationResult, VSRPInstance, VSRPSolution


# =============================================================================
# 1. ROUTE VALIDATION
# =============================================================================

def validate_route(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> tuple[bool, list[str]]:
    """
    Validate route structure and continuity.

    Checks
    ------
    1. Route is non-empty
    2. Route starts at port 0
    3. Route ends at the final port
    4. Consecutive legs are continuous
    5. No interior port is visited more than once
    6. All route indices are within valid bounds
    """
    errors: list[str] = []
    route = solution.route_legs
    n_ports = instance.n_ports

    if not route:
        return False, ["Route is empty"]

    for k, leg in enumerate(route):
        if not (0 <= leg.from_port_idx < n_ports):
            errors.append(
                f"Leg {k}: invalid from_port_idx={leg.from_port_idx}"
            )
        if not (0 <= leg.to_port_idx < n_ports):
            errors.append(
                f"Leg {k}: invalid to_port_idx={leg.to_port_idx}"
            )

    if errors:
        return False, errors

    if route[0].from_port_idx != 0:
        errors.append(
            f"Route starts at port index {route[0].from_port_idx}, "
            f"expected 0"
        )

    if route[-1].to_port_idx != n_ports - 1:
        errors.append(
            f"Route ends at port index {route[-1].to_port_idx}, "
            f"expected {n_ports - 1}"
        )

    for i in range(len(route) - 1):
        if route[i].to_port_idx != route[i + 1].from_port_idx:
            errors.append(
                f"Route discontinuity between legs {i} and {i + 1}: "
                f"{route[i].to_port_idx} != {route[i + 1].from_port_idx}"
            )

    visited_sequence = (
        [route[0].from_port_idx] + [leg.to_port_idx for leg in route]
    )
    interior = visited_sequence[1:-1]
    duplicates = sorted({p for p in interior if interior.count(p) > 1})
    if duplicates:
        errors.append(
            f"Interior ports visited more than once: {duplicates}"
        )

    return len(errors) == 0, errors


# =============================================================================
# 2. TIMELINE VALIDATION
# =============================================================================

def validate_timeline(
    instance: VSRPInstance,
    solution: VSRPSolution,
    *,
    tol: float = 1e-6,
) -> tuple[bool, list[str]]:
    """
    Validate timeline monotonicity and internal consistency.

    Checks
    ------
    1. Timeline entries have valid port indices
    2. Arrival times do not move backwards across realised visits
    3. Departure time is not earlier than arrival time
    4. Stored delay equals actual_arrival - planned_arrival within tolerance
    """
    warnings: list[str] = []
    timeline = solution.timeline
    n_ports = instance.n_ports

    if not timeline:
        return False, ["Timeline is empty"]

    prev_departure = None

    for i, entry in enumerate(timeline):
        if not (0 <= entry.port_idx < n_ports):
            warnings.append(
                f"Timeline entry {i}: invalid port_idx={entry.port_idx}"
            )
            continue

        if entry.departure_h + tol < entry.actual_arrival_h:
            warnings.append(
                f"Timeline entry {i} ({entry.port_idx}): "
                f"departure {entry.departure_h:.4f} < "
                f"arrival {entry.actual_arrival_h:.4f}"
            )

        expected_delay = entry.actual_arrival_h - entry.planned_arrival_h
        if abs(entry.delay_h - expected_delay) > tol:
            warnings.append(
                f"Timeline entry {i} ({entry.port_idx}): "
                f"delay mismatch, stored={entry.delay_h:.4f}, "
                f"expected={expected_delay:.4f}"
            )

        if prev_departure is not None and entry.status != "SKIPPED":
            if entry.actual_arrival_h + tol < prev_departure:
                warnings.append(
                    f"Timeline entry {i} ({entry.port_idx}): "
                    f"arrival {entry.actual_arrival_h:.4f} before "
                    f"previous departure {prev_departure:.4f}"
                )

        if entry.status != "SKIPPED":
            prev_departure = entry.departure_h

    return len(warnings) == 0, warnings


# =============================================================================
# 3. STRATEGY VALIDATION
# =============================================================================

def validate_strategy_consistency(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> tuple[bool, list[str]]:
    """
    Validate consistency between route-level outcomes and strategy tags.

    Checks
    ------
    1. Every skipped port has a PORT_OMISSION strategy
    2. Every PORT_OMISSION strategy corresponds to a skipped port
    3. Every swapped port has a PORT_SWAP strategy
    4. Every PORT_SWAP strategy corresponds to a swapped port
    5. Swap-group route legs appear in complete groups (at least 2 legs)
    """
    warnings: list[str] = []

    skipped = set(solution.skipped_port_indices)
    swapped = set(solution.swapped_port_indices)

    omission_ports = {
        s.port_idx
        for s in solution.strategy_decisions
        if s.strategy == "PORT_OMISSION"
    }
    swap_ports = {
        s.port_idx
        for s in solution.strategy_decisions
        if s.strategy == "PORT_SWAP"
    }

    for p in skipped:
        if p not in omission_ports:
            warnings.append(
                f"Port {p} is skipped but has no PORT_OMISSION strategy"
            )

    for p in omission_ports:
        if p not in skipped:
            warnings.append(
                f"PORT_OMISSION strategy for port {p} "
                f"but port not skipped"
            )

    for p in swapped:
        if p not in swap_ports:
            warnings.append(
                f"Port {p} is swapped but has no PORT_SWAP strategy"
            )

    for p in swap_ports:
        if p not in swapped:
            warnings.append(
                f"PORT_SWAP strategy for port {p} "
                f"but port not in swapped_port_indices"
            )

    swap_group_ids = {
        leg.swap_group_id
        for leg in solution.route_legs
        if leg.is_swap and leg.swap_group_id is not None
    }
    for gid in swap_group_ids:
        group_legs = [
            leg for leg in solution.route_legs
            if leg.swap_group_id == gid
        ]
        if len(group_legs) < 2:
            warnings.append(
                f"Swap group {gid} has only {len(group_legs)} leg(s); "
                f"expected at least 2"
            )

    return len(warnings) == 0, warnings


# =============================================================================
# 4. CONTAINER VALIDATION
# =============================================================================

def validate_container_outcomes(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> tuple[bool, list[str]]:
    """
    Validate that reported container outcomes align with the input instance.

    Checks
    ------
    1. Every instance container appears in solution.container_outcomes
    2. Outcome origin/destination match instance data
    3. Outcome transshipment indices match instance data
    """
    warnings: list[str] = []

    outcome_ids = set(solution.container_outcomes.keys())
    instance_ids = {c.id for c in instance.containers}

    missing = sorted(instance_ids - outcome_ids)
    extra = sorted(outcome_ids - instance_ids)

    if missing:
        warnings.append(f"Missing container outcomes for: {missing}")
    if extra:
        warnings.append(f"Unexpected container outcomes for: {extra}")

    instance_by_id = {c.id: c for c in instance.containers}

    for c_id in sorted(instance_ids & outcome_ids):
        c_inst = instance_by_id[c_id]
        c_out = solution.container_outcomes[c_id]

        if c_out.origin_idx != c_inst.origin_idx:
            warnings.append(
                f"Container {c_id}: origin mismatch "
                f"({c_out.origin_idx} != {c_inst.origin_idx})"
            )

        if c_out.destination_idx != c_inst.destination_idx:
            warnings.append(
                f"Container {c_id}: destination mismatch "
                f"({c_out.destination_idx} != {c_inst.destination_idx})"
            )

        if (
            list(c_out.transshipment_port_indices)
            != list(c_inst.transshipment_port_indices)
        ):
            warnings.append(
                f"Container {c_id}: transshipment mismatch "
                f"({c_out.transshipment_port_indices} != "
                f"{c_inst.transshipment_port_indices})"
            )

    return len(warnings) == 0, warnings


# =============================================================================
# 5. SKIPPED-PORT DERIVATION CHECK
# =============================================================================

def validate_skipped_ports(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> tuple[bool, list[str]]:
    """
    Cross-check reported skipped ports against reconstructed route visitation.
    """
    warnings: list[str] = []
    route = solution.route_legs

    if not route:
        return False, ["Cannot validate skipped ports: route is empty"]

    visited = {route[0].from_port_idx}
    for leg in route:
        visited.add(leg.to_port_idx)

    expected_skipped = {
        p for p in range(1, instance.n_ports - 1)
        if p not in visited
    }
    reported_skipped = set(solution.skipped_port_indices)

    missing = sorted(expected_skipped - reported_skipped)
    extra = sorted(reported_skipped - expected_skipped)

    if missing:
        warnings.append(
            f"Ports skipped by route but not reported: {missing}"
        )
    if extra:
        warnings.append(
            f"Ports reported as skipped but visited in route: {extra}"
        )

    return len(warnings) == 0, warnings


# =============================================================================
# 6. NUMERICAL CONSTRAINT VIOLATION
# =============================================================================

def compute_constraint_violations(
    instance: VSRPInstance,
    solution: VSRPSolution,
    *,
    tol: float = 1e-6,
) -> dict[str, float]:
    """
    Evaluate simple structural constraint residuals from the canonical
    solution object.

    This check is solver-agnostic: it reconstructs residuals from the
    extracted solution without accessing any solver-internal constraint
    matrix.

    Constraints evaluated
    ---------------------
    1. flow_balance
       Origin outflow, sink inflow, and interior flow conservation.

    2. port_call_consistency
       No interior port should appear as visited more than once.

    3. omission_consistency
       A port should be reported as skipped if and only if it is not
       visited in the reconstructed route.

    Notes
    -----
    - Values greater than tolerance indicate a structural inconsistency.
    - Delay-classification consistency is intentionally NOT treated as a
      hard numerical violation in the current implementation because the
      MIP delayed indicator uses a conservative timing approximation,
      while the reconstructed timeline is more explicit and may differ
      without implying infeasibility.
    - The tolerance parameter is retained for API consistency even though
      the currently active reconstructed checks are effectively exact.
    """
    violations: dict[str, float] = {}

    route = solution.route_legs
    if not route:
        return {
            "flow_balance": 1.0,
            "port_call_consistency": 0.0,
            "omission_consistency": 0.0,
        }

    # -----------------------------------------------------------------
    # 1. Flow balance
    # -----------------------------------------------------------------
    outflow: dict[int, int] = {}
    inflow: dict[int, int] = {}

    for leg in route:
        outflow[leg.from_port_idx] = (
            outflow.get(leg.from_port_idx, 0) + 1
        )
        inflow[leg.to_port_idx] = (
            inflow.get(leg.to_port_idx, 0) + 1
        )

    max_flow_violation = 0.0

    origin_out = outflow.get(0, 0)
    max_flow_violation = max(max_flow_violation, abs(origin_out - 1))

    sink_in = inflow.get(instance.n_ports - 1, 0)
    max_flow_violation = max(max_flow_violation, abs(sink_in - 1))

    for p in range(1, instance.n_ports - 1):
        p_out = outflow.get(p, 0)
        p_in = inflow.get(p, 0)
        max_flow_violation = max(max_flow_violation, abs(p_out - p_in))

    violations["flow_balance"] = float(max_flow_violation)

    # -----------------------------------------------------------------
    # 2. Port-call consistency
    # -----------------------------------------------------------------
    visited_ports: dict[int, int] = {}
    for leg in route:
        p = leg.to_port_idx
        if p != instance.n_ports - 1:
            visited_ports[p] = visited_ports.get(p, 0) + 1

    max_portcall_violation = 0.0
    for count in visited_ports.values():
        max_portcall_violation = max(
            max_portcall_violation,
            max(0, count - 1),
        )

    violations["port_call_consistency"] = float(max_portcall_violation)

    # -----------------------------------------------------------------
    # 3. Omission consistency
    # -----------------------------------------------------------------
    visited_set = {route[0].from_port_idx}
    for leg in route:
        visited_set.add(leg.to_port_idx)

    skipped_set = set(solution.skipped_port_indices)
    max_omission_violation = 0.0

    for p in range(1, instance.n_ports - 1):
        is_visited = int(p in visited_set)
        is_omitted = int(p in skipped_set)
        residual = abs(is_omitted - (1 - is_visited))
        max_omission_violation = max(max_omission_violation, residual)

    violations["omission_consistency"] = float(max_omission_violation)

    return violations


# =============================================================================
# 7. TOP-LEVEL VALIDATION
# =============================================================================

def validate_solution(
    instance: VSRPInstance,
    solution: VSRPSolution,
) -> ValidationResult:
    """
    Run all solver-agnostic validation checks and return a structured result.
    """
    route_valid, route_errors = validate_route(instance, solution)
    strat_ok, strat_warnings = validate_strategy_consistency(
        instance, solution
    )
    timeline_ok, timeline_warnings = validate_timeline(instance, solution)
    containers_ok, container_warnings = validate_container_outcomes(
        instance, solution
    )
    skipped_ok, skipped_warnings = validate_skipped_ports(
        instance, solution
    )

    violations = compute_constraint_violations(instance, solution)
    max_violation = max(violations.values()) if violations else None
    n_violated = sum(1 for v in violations.values() if v > 1e-6)

    overall_valid = (
        route_valid
        and strat_ok
        and timeline_ok
        and containers_ok
        and skipped_ok
        and (max_violation is None or max_violation <= 1e-6)
    )

    return ValidationResult(
        route_valid=route_valid,
        route_errors=route_errors,
        strategy_consistent=strat_ok,
        strategy_warnings=strat_warnings,
        timeline_monotone=timeline_ok,
        timeline_warnings=timeline_warnings,
        container_valid=containers_ok,
        container_warnings=container_warnings,
        skipped_ports_valid=skipped_ok,
        skipped_port_warnings=skipped_warnings,
        max_constraint_violation=max_violation,
        n_violated_constraints=n_violated,
        overall_valid=overall_valid,
    )