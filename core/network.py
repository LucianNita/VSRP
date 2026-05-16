# =============================================================================
# Network construction and travel-cost helpers for the Vessel Schedule Recovery
# Problem (VSRP).
#
# Purpose
# -------
# This module builds the feasible routing network used by the optimization
# model. It is solver-agnostic and depends only on canonical entities from
# `core.entities`.
#
# Main responsibilities
# ---------------------
# - compute travel time, fuel consumption, and fuel cost on candidate legs
# - construct forward recovery edges that allow limited port omission
# - construct swap-edge groups that represent local port reordering
# - provide helper utilities for swap-group inspection and planned schedule
#   reconstruction
#
# Architectural role
# ------------------
# This file isolates all optimization-independent route-network logic from
# the solver backends. Both the native Xpress solver and the PuLP-based
# open-source formulation consume the same edge set generated here.
# =============================================================================

from __future__ import annotations

from core.entities import Edge, VSRPInstance


# =============================================================================
# 1. FUEL / TRAVEL HELPERS
# =============================================================================

def fuel_consumption_tonnes(
    distance_nm: float,
    speed_knots: float,
    *,
    fuel_base_consumption_tpd: float = 100.0,
    reference_speed_knots: float = 20.0,
) -> float:
    """
    Estimate fuel consumption on a sailing leg using a cubic-speed model.

    The model assumes:
    - fuel burn per day scales with the cube of speed
    - total fuel burn equals daily burn multiplied by travel duration

    Formula
    -------
    $$
    F = F_0 \cdot \left(\frac{v}{v_0}\right)^3 \cdot \frac{d}{v \cdot 24}
    $$

    where:
    - $$F$$ is fuel consumed in tonnes
    - $$F_0$$ is reference daily fuel consumption in tonnes/day
    - $$v$$ is actual sailing speed in knots
    - $$v_0$$ is reference speed in knots
    - $$d$$ is leg distance in nautical miles
    """
    if speed_knots <= 0:
        raise ValueError("speed_knots must be strictly positive")
    if distance_nm < 0:
        raise ValueError("distance_nm must be non-negative")

    travel_time_h = distance_nm / speed_knots
    daily_burn_tpd = fuel_base_consumption_tpd * (
        speed_knots / reference_speed_knots
    ) ** 3
    return daily_burn_tpd * (travel_time_h / 24.0)


def fuel_cost_usd(
    distance_nm: float,
    speed_knots: float,
    *,
    fuel_price_usd_per_tonne: float = 600.0,
    fuel_base_consumption_tpd: float = 100.0,
    reference_speed_knots: float = 20.0,
) -> float:
    """
    Compute fuel cost for one sailing leg.

    This is a thin wrapper around `fuel_consumption_tonnes()` that multiplies
    the estimated fuel burn by the current fuel price.
    """
    fuel_t = fuel_consumption_tonnes(
        distance_nm=distance_nm,
        speed_knots=speed_knots,
        fuel_base_consumption_tpd=fuel_base_consumption_tpd,
        reference_speed_knots=reference_speed_knots,
    )
    return fuel_t * fuel_price_usd_per_tonne


def travel_time_h(distance_nm: float, speed_knots: float) -> float:
    """
    Compute sailing time in hours for one leg.
    """
    if speed_knots <= 0:
        raise ValueError("speed_knots must be strictly positive")
    if distance_nm < 0:
        raise ValueError("distance_nm must be non-negative")
    return distance_nm / speed_knots


# =============================================================================
# 2. NETWORK CONSTRUCTION
# =============================================================================

def build_network(
    *,
    ports: list[str],
    distance_matrix_nm: list[list[float]],
    speed_levels_knots: list[float],
    max_skip: int = 1,
    allow_swap: bool = True,
    max_swap_distance: int = 2,
    fuel_price_usd_per_tonne: float = 600.0,
    fuel_base_consumption_tpd: float = 100.0,
    reference_speed_knots: float = 20.0,
) -> list[Edge]:
    """
    Build the feasible route network.

    Edge types
    ----------
    1. Forward edges
       A forward edge connects port `i` to a later port `j` with `j > i`.
       It represents normal forward progression through the route, possibly
       skipping some intermediate planned ports.

       The parameter `max_skip` limits how many planned ports may be omitted
       in a single forward jump.

    2. Swap edge groups
       When `allow_swap=True`, the network also includes local port-reordering
       patterns represented as three-edge swap groups.

       For a swapped pair of interior ports `(i, j)` with `i < j`, the group
       consists of:

       - Leg A: predecessor(i) -> j
         The vessel jumps forward to `j`, postponing `i`.

       - Leg B: j -> i
         The vessel backtracks from `j` to `i`.

       - Leg C: i -> successor(j)
         The vessel resumes forward progression after serving `i`.

       All three legs:
       - have `is_swap=True`
       - share the same `swap_group_id`
       - are intended to be selected together by the optimization model

    Swap-distance restriction
    -------------------------
    `max_swap_distance` limits how far apart swapped ports may be in the
    planned sequence.

    This is important operationally and computationally:
    - it prevents unrealistic long-range reorderings
    - it avoids degenerate routes that skip too much of the nominal plan
    - it controls edge-set growth and the number of ordering relations

    With the default `max_swap_distance=2`, only:
    - adjacent swaps (`j = i + 1`)
    - next-adjacent swaps (`j = i + 2`)
    are allowed.

    Parameters
    ----------
    ports : list[str]
        Ordered planned port sequence.
    distance_matrix_nm : list[list[float]]
        Distance matrix aligned with `ports`.
    speed_levels_knots : list[float]
        Discrete sailing speed options.
    max_skip : int, default=1
        Maximum number of planned ports that may be skipped on a forward edge.
    allow_swap : bool, default=True
        Whether swap edge groups are generated.
    max_swap_distance : int, default=2
        Maximum planned-sequence distance between swapped ports.
    fuel_price_usd_per_tonne : float, default=600.0
        Fuel price used to compute edge-level fuel cost.
    fuel_base_consumption_tpd : float, default=100.0
        Reference daily fuel consumption used in the cubic-speed model.
    reference_speed_knots : float, default=20.0
        Reference speed used in the cubic-speed model.

    Returns
    -------
    list[Edge]
        Feasible network edges for the recovery model.
    """
    _validate_network_inputs(
        ports=ports,
        distance_matrix_nm=distance_matrix_nm,
        speed_levels_knots=speed_levels_knots,
        max_skip=max_skip,
        max_swap_distance=max_swap_distance,
    )

    n_ports = len(ports)
    edges: list[Edge] = []

    cost_kwargs = dict(
        fuel_price_usd_per_tonne=fuel_price_usd_per_tonne,
        fuel_base_consumption_tpd=fuel_base_consumption_tpd,
        reference_speed_knots=reference_speed_knots,
    )

    # -----------------------------------------------------------------
    # Forward edges
    # -----------------------------------------------------------------
    # For each port i, allow forward travel to later ports j, limited by
    # max_skip. Since skipping k intermediate ports means jumping from i
    # to i + k + 1, the largest feasible destination index is:
    #
    #   j <= i + max_skip + 1
    #
    # The loop bound uses Python's exclusive upper range endpoint.
    # -----------------------------------------------------------------
    for i in range(n_ports - 1):
        max_j = min(i + max_skip + 2, n_ports)
        for j in range(i + 1, max_j):
            distance_nm = distance_matrix_nm[i][j]
            if distance_nm <= 0:
                continue

            skipped = list(range(i + 1, j))

            for speed in speed_levels_knots:
                edges.append(
                    Edge(
                        from_port_idx=i,
                        to_port_idx=j,
                        speed_knots=speed,
                        travel_time_h=travel_time_h(distance_nm, speed),
                        fuel_cost_usd=fuel_cost_usd(
                            distance_nm,
                            speed,
                            **cost_kwargs,
                        ),
                        is_swap=False,
                        swap_group_id=None,
                        skipped_port_indices=skipped,
                    )
                )

    # -----------------------------------------------------------------
    # Swap edge groups
    # -----------------------------------------------------------------
    if allow_swap:
        swap_group_id = 0

        # Only interior ports may be swapped. The origin and sink remain fixed.
        interior = list(range(1, n_ports - 1))

        for idx_i, i in enumerate(interior):
            for j in interior[idx_i + 1:]:

                # Restrict swaps to local reorderings.
                if j - i > max_swap_distance:
                    continue

                pred_i = i - 1
                succ_j = j + 1 if j + 1 < n_ports else n_ports - 1

                # ---------------------------------------------------------
                # Leg A: pred_i -> j
                #
                # This edge jumps from the predecessor of i directly to j.
                # Port i itself is not considered skipped, because it will
                # still be visited later within the swap sequence.
                #
                # The skipped ports attached to Leg A therefore include only
                # planned ports strictly between pred_i and j, excluding i.
                # ---------------------------------------------------------
                dist_A = distance_matrix_nm[pred_i][j]
                if dist_A <= 0:
                    continue

                skipped_on_A = [
                    p for p in range(pred_i + 1, j)
                    if p != i
                ]

                # ---------------------------------------------------------
                # Leg B: j -> i
                #
                # This is the reverse leg that serves i after j.
                # It does not represent omission of any planned ports.
                # ---------------------------------------------------------
                dist_B = distance_matrix_nm[j][i]
                if dist_B <= 0:
                    continue

                # ---------------------------------------------------------
                # Leg C: i -> succ_j
                #
                # This leg resumes forward progression after the swap pair
                # has been served in reversed order.
                # ---------------------------------------------------------
                dist_C = distance_matrix_nm[i][succ_j]
                if dist_C <= 0:
                    continue

                # Generate one structural swap group per speed level.
                for speed in speed_levels_knots:
                    edges.append(
                        Edge(
                            from_port_idx=pred_i,
                            to_port_idx=j,
                            speed_knots=speed,
                            travel_time_h=travel_time_h(dist_A, speed),
                            fuel_cost_usd=fuel_cost_usd(
                                dist_A,
                                speed,
                                **cost_kwargs,
                            ),
                            is_swap=True,
                            swap_group_id=swap_group_id,
                            skipped_port_indices=skipped_on_A,
                        )
                    )
                    edges.append(
                        Edge(
                            from_port_idx=j,
                            to_port_idx=i,
                            speed_knots=speed,
                            travel_time_h=travel_time_h(dist_B, speed),
                            fuel_cost_usd=fuel_cost_usd(
                                dist_B,
                                speed,
                                **cost_kwargs,
                            ),
                            is_swap=True,
                            swap_group_id=swap_group_id,
                            skipped_port_indices=[],
                        )
                    )
                    edges.append(
                        Edge(
                            from_port_idx=i,
                            to_port_idx=succ_j,
                            speed_knots=speed,
                            travel_time_h=travel_time_h(dist_C, speed),
                            fuel_cost_usd=fuel_cost_usd(
                                dist_C,
                                speed,
                                **cost_kwargs,
                            ),
                            is_swap=True,
                            swap_group_id=swap_group_id,
                            skipped_port_indices=[],
                        )
                    )

                swap_group_id += 1

    return edges


def build_network_from_instance(
    instance: VSRPInstance,
    *,
    fuel_base_consumption_tpd: float = 100.0,
    reference_speed_knots: float = 20.0,
) -> list[Edge]:
    """
    Build the feasible network directly from a canonical `VSRPInstance`.

    This wrapper ensures that edge fuel costs inherit the instance-level
    fuel price. That is important for experiments such as CFA or
    stochastic fuel-price studies, where different episodes may solve
    otherwise identical networks under different fuel-price assumptions.
    """
    return build_network(
        ports=instance.ports,
        distance_matrix_nm=instance.distance_matrix_nm,
        speed_levels_knots=instance.speed_levels_knots,
        max_skip=instance.max_skip,
        allow_swap=instance.allow_swap,
        max_swap_distance=getattr(instance, "max_swap_distance", 2),
        fuel_price_usd_per_tonne=instance.fuel_price_usd_per_tonne,
        fuel_base_consumption_tpd=fuel_base_consumption_tpd,
        reference_speed_knots=reference_speed_knots,
    )


# =============================================================================
# 3. SWAP GROUP HELPERS
# =============================================================================

def get_swap_groups(edges: list[Edge]) -> dict[int, list[Edge]]:
    """
    Group swap edges by `swap_group_id`.

    Returns
    -------
    dict[int, list[Edge]]
        Mapping from swap-group identifier to the list of edges that
        belong to that structural swap pattern.
    """
    groups: dict[int, list[Edge]] = {}
    for edge in edges:
        if edge.is_swap and edge.swap_group_id is not None:
            groups.setdefault(edge.swap_group_id, []).append(edge)
    return groups


def get_swap_group_ids(edges: list[Edge]) -> list[int]:
    """
    Return the sorted list of unique swap-group identifiers in an edge list.
    """
    return sorted({
        edge.swap_group_id
        for edge in edges
        if edge.is_swap and edge.swap_group_id is not None
    })


def get_swap_group_port_pair(
    group_edges: list[Edge],
) -> tuple[int, int] | None:
    """
    Recover the logical swapped port pair `(i, j)` from a swap group.

    The current swap construction identifies the reversed leg `j -> i`
    as the only structural leg with `from_port_idx > to_port_idx`.
    That reverse leg uniquely identifies the swapped pair.

    Returns
    -------
    tuple[int, int] | None
        `(i, j)` with `i < j`, or `None` if the group does not match the
        expected structure.
    """
    for edge in group_edges:
        if edge.from_port_idx > edge.to_port_idx:
            j_port = edge.from_port_idx
            i_port = edge.to_port_idx
            return (i_port, j_port)
    return None


# =============================================================================
# 4. SCHEDULE HELPERS
# =============================================================================

def compute_planned_arrivals(
    *,
    distance_matrix_nm: list[list[float]],
    nominal_speed_knots: float = 20.0,
    nominal_port_call_duration_h: float = 12.0,
) -> list[float]:
    """
    Compute nominal planned arrival times along the original route order.

    Assumptions
    -----------
    - the vessel follows the planned route order
    - the same nominal speed is used on all legs
    - the same nominal port-call duration is used at each visited port

    Returns
    -------
    list[float]
        Planned arrival time at each port index.
    """
    n_ports = len(distance_matrix_nm)
    arrivals = [0.0] * n_ports

    departure_h = nominal_port_call_duration_h
    for p in range(1, n_ports):
        dist = distance_matrix_nm[p - 1][p]
        arrivals[p] = departure_h + dist / nominal_speed_knots
        departure_h = arrivals[p] + nominal_port_call_duration_h

    return arrivals


def compute_planned_departures(
    *,
    distance_matrix_nm: list[list[float]],
    nominal_speed_knots: float = 20.0,
    nominal_port_call_duration_h: float = 12.0,
) -> list[float]:
    """
    Compute nominal planned departure times along the original route order.

    This is derived from `compute_planned_arrivals()` by adding the
    nominal port-call duration at each port.
    """
    arrivals = compute_planned_arrivals(
        distance_matrix_nm=distance_matrix_nm,
        nominal_speed_knots=nominal_speed_knots,
        nominal_port_call_duration_h=nominal_port_call_duration_h,
    )

    departures = [0.0] * len(arrivals)
    departures[0] = nominal_port_call_duration_h
    for p in range(1, len(arrivals)):
        departures[p] = arrivals[p] + nominal_port_call_duration_h

    return departures


# =============================================================================
# 5. INTERNAL VALIDATION
# =============================================================================

def _validate_network_inputs(
    *,
    ports: list[str],
    distance_matrix_nm: list[list[float]],
    speed_levels_knots: list[float],
    max_skip: int,
    max_swap_distance: int = 2,
) -> None:
    """
    Validate the structural inputs used for network construction.

    This is a lightweight consistency check intended to catch malformed
    route definitions or parameter values before edge generation begins.
    """
    n_ports = len(ports)

    if n_ports < 2:
        raise ValueError("At least two ports are required")

    if len(distance_matrix_nm) != n_ports:
        raise ValueError(
            "distance_matrix_nm must have one row per port"
        )

    for row in distance_matrix_nm:
        if len(row) != n_ports:
            raise ValueError(
                "distance_matrix_nm must be square and aligned with ports"
            )

    if not speed_levels_knots:
        raise ValueError("speed_levels_knots must not be empty")

    if any(v <= 0 for v in speed_levels_knots):
        raise ValueError("All speed levels must be strictly positive")

    if max_skip < 0:
        raise ValueError("max_skip must be non-negative")

    if max_swap_distance < 1:
        raise ValueError(
            "max_swap_distance must be >= 1 "
            "(minimum meaningful swap distance is 1, i.e. adjacent ports)"
        )