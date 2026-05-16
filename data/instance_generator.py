# =============================================================================
# Random container-set generation for the Vessel Schedule Recovery Problem.
#
# Purpose
# -------
# This module generates reproducible container demand sets that are
# consistent with the route and timing assumptions of the case study.
#
# Main responsibilities
# ---------------------
# - generate random origin/destination pairs
# - calibrate promised arrivals against a nominal disrupted schedule
# - assign delay and misconnection penalties
# - optionally generate transshipment containers and connection deadlines
#
# Important modeling choice
# -------------------------
# Promised arrivals are calibrated relative to an expected initial delay
# rather than a zero-delay baseline. This avoids the degenerate behavior
# where all containers become structurally late as soon as the disruption
# delay is non-zero.
# =============================================================================

from __future__ import annotations

import numpy as np

from core.entities import Container


# =============================================================================
# 1. DEFAULT GENERATION SETTINGS
# =============================================================================

DEFAULT_RANDOM_SEED: int = 42

DEFAULT_CONTAINER_QTY_MIN: int = 100
DEFAULT_CONTAINER_QTY_MAX: int = 500

# Notebook-compatible penalty defaults used for comparability with the
# original baseline implementation.
NOTEBOOK_DELAY_RATE_USD_PER_H: float = 200.0
NOTEBOOK_MISCONNECT_PENALTY_USD: float = 1_000.0

# Wider penalty ranges for stress-test style experiments.
STRESS_PENALTY_DELAY_MIN: float = 5_000.0
STRESS_PENALTY_DELAY_MAX: float = 20_000.0
STRESS_PENALTY_MISCONNECT_MIN: float = 1_000.0
STRESS_PENALTY_MISCONNECT_MAX: float = 5_000.0

# Promised slack added on top of nominal arrival time.
# This value is calibrated so that at the expected disruption level,
# some containers remain on time while others may become late under
# more aggressive or slower recovery plans.
DEFAULT_PROMISED_SLACK_H: float = 24.0

DEFAULT_PRIORITY_LEVELS: list[str] = ["HIGH", "MEDIUM", "LOW"]

DEFAULT_TRANSSHIPMENT_PROBABILITY: float = 0.2
DEFAULT_CONNECTING_SERVICE_SLACK_H: float = 12.0

# Expected initial delay used when calibrating promised arrivals.
# This anchors service promises to a realistic disrupted baseline.
DEFAULT_EXPECTED_INITIAL_DELAY_H: float = 48.0


# =============================================================================
# 2. HELPERS
# =============================================================================

def cumulative_nominal_arrivals(
    distance_matrix_nm: list[list[float]],
    *,
    speed_knots: float = 20.0,
    port_call_duration_h: float = 12.0,
    initial_delay_h: float = 0.0,
) -> list[float]:
    """
    Compute cumulative nominal arrival times along the planned route.

    Parameters
    ----------
    distance_matrix_nm : list[list[float]]
        Planned-route distance matrix.
    speed_knots : float, default=20.0
        Nominal sailing speed used for calibration.
    port_call_duration_h : float, default=12.0
        Nominal port-call duration used for calibration.
    initial_delay_h : float, default=0.0
        Baseline disruption delay used to shift the entire nominal
        schedule forward.

    Returns
    -------
    list[float]
        Nominal arrival time at each port index.
    """
    n_ports = len(distance_matrix_nm)
    arrivals = [0.0] * n_ports

    arrivals[0] = initial_delay_h
    current_departure = initial_delay_h + port_call_duration_h

    for j in range(1, n_ports):
        dist = distance_matrix_nm[j - 1][j]
        travel_time_h = dist / speed_knots
        arrivals[j] = current_departure + travel_time_h
        current_departure = arrivals[j] + port_call_duration_h

    return arrivals


def _draw_priority(
    rng: np.random.Generator,
    levels: list[str],
) -> str:
    """
    Draw a random priority label.
    """
    return str(rng.choice(levels))


def _draw_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
) -> float:
    """
    Draw a uniform random float on [low, high].
    """
    return float(rng.uniform(low, high))


def _draw_transshipment_port(
    rng: np.random.Generator,
    origin_idx: int,
    destination_idx: int,
    n_ports: int,
) -> int | None:
    """
    Draw a feasible transshipment port strictly between origin and destination.

    Returns `None` if no intermediate port exists.
    """
    candidates = [p for p in range(origin_idx + 1, destination_idx)]
    if not candidates:
        return None
    return int(rng.choice(candidates))


# =============================================================================
# 3. CONTAINER GENERATION
# =============================================================================

def generate_containers(
    *,
    ports: list[str],
    distance_matrix_nm: list[list[float]],
    n: int = 5,
    seed: int = DEFAULT_RANDOM_SEED,
    quantity_min: int = DEFAULT_CONTAINER_QTY_MIN,
    quantity_max: int = DEFAULT_CONTAINER_QTY_MAX,
    promised_slack_h: float = DEFAULT_PROMISED_SLACK_H,
    nominal_speed_knots: float = 20.0,
    nominal_port_call_duration_h: float = 12.0,
    expected_initial_delay_h: float = DEFAULT_EXPECTED_INITIAL_DELAY_H,
    allow_same_origin_destination: bool = False,
    priority_levels: list[str] | None = None,
    notebook_compatible: bool = True,
    penalty_delay_min: float = STRESS_PENALTY_DELAY_MIN,
    penalty_delay_max: float = STRESS_PENALTY_DELAY_MAX,
    penalty_misconnect_min: float = STRESS_PENALTY_MISCONNECT_MIN,
    penalty_misconnect_max: float = STRESS_PENALTY_MISCONNECT_MAX,
    transshipment_probability: float = 0.0,
    connecting_service_slack_h: float = DEFAULT_CONNECTING_SERVICE_SLACK_H,
) -> list[Container]:
    """
    Generate a reproducible random set of container demands.

    Promise calibration
    -------------------
    Promised arrival times are built from:
    - nominal route travel time
    - nominal port-call time
    - expected initial delay
    - promised slack

    This means promises are referenced to a realistic disrupted baseline
    rather than a zero-delay schedule.

    Parameters
    ----------
    ports : list[str]
        Port list aligned with the distance matrix.
    distance_matrix_nm : list[list[float]]
        Distance matrix aligned with `ports`.
    n : int, default=5
        Number of containers to generate.
    seed : int, default=42
        Random seed for reproducibility.
    quantity_min, quantity_max : int
        Range for randomly generated TEU quantities.
    promised_slack_h : float, default=24.0
        Slack added to the nominal destination arrival time.
    nominal_speed_knots : float, default=20.0
        Nominal speed used for promise calibration.
    nominal_port_call_duration_h : float, default=12.0
        Nominal port-call duration used for promise calibration.
    expected_initial_delay_h : float, default=48.0
        Baseline disruption delay used in promised-arrival calibration.
    allow_same_origin_destination : bool, default=False
        Whether origin and destination may initially be drawn equal.
    priority_levels : list[str] | None
        Optional priority labels.
    notebook_compatible : bool, default=True
        When True, use notebook-style penalty calibration.
        When False, draw penalties from wider stress-test ranges.
    penalty_delay_min, penalty_delay_max : float
        Delay-penalty draw range when not notebook-compatible.
    penalty_misconnect_min, penalty_misconnect_max : float
        Misconnection-penalty draw range when not notebook-compatible.
    transshipment_probability : float, default=0.0
        Probability that a generated container is assigned a transshipment port.
    connecting_service_slack_h : float, default=12.0
        Time slack subtracted from nominal transshipment arrival to create
        a connecting-service deadline.

    Returns
    -------
    list[Container]
        Reproducible generated container set.
    """
    if n <= 0:
        return []

    if len(ports) != len(distance_matrix_nm):
        raise ValueError(
            "ports and distance_matrix_nm must have aligned dimensions"
        )

    if not 0.0 <= transshipment_probability <= 1.0:
        raise ValueError(
            "transshipment_probability must be in [0, 1]"
        )

    rng = np.random.default_rng(seed)
    priorities = priority_levels or DEFAULT_PRIORITY_LEVELS
    n_ports = len(ports)

    nominal_arrivals = cumulative_nominal_arrivals(
        distance_matrix_nm,
        speed_knots=nominal_speed_knots,
        port_call_duration_h=nominal_port_call_duration_h,
        initial_delay_h=expected_initial_delay_h,
    )

    containers: list[Container] = []

    for i in range(n):
        origin_idx = int(rng.integers(0, n_ports))
        destination_idx = int(rng.integers(0, n_ports))

        if not allow_same_origin_destination:
            while destination_idx == origin_idx:
                destination_idx = int(rng.integers(0, n_ports))

        # Normalize to forward route direction.
        if destination_idx < origin_idx:
            origin_idx, destination_idx = destination_idx, origin_idx

        # Guard against degenerate same-port case after normalization.
        if destination_idx == origin_idx:
            if origin_idx < n_ports - 1:
                destination_idx = origin_idx + 1
            else:
                origin_idx = max(0, origin_idx - 1)

        promised_arrival_h = (
            nominal_arrivals[destination_idx] + promised_slack_h
        )

        if notebook_compatible:
            penalty_delay = (
                NOTEBOOK_DELAY_RATE_USD_PER_H * promised_arrival_h
            )
            penalty_misconnect = NOTEBOOK_MISCONNECT_PENALTY_USD
        else:
            penalty_delay = _draw_uniform(
                rng,
                penalty_delay_min,
                penalty_delay_max,
            )
            penalty_misconnect = _draw_uniform(
                rng,
                penalty_misconnect_min,
                penalty_misconnect_max,
            )

        quantity_teu = int(rng.integers(quantity_min, quantity_max + 1))

        transshipment_port_indices: list[int] = []
        connecting_service_deadline_h: float | None = None

        if (
            transshipment_probability > 0.0
            and rng.random() < transshipment_probability
        ):
            trans_port = _draw_transshipment_port(
                rng,
                origin_idx,
                destination_idx,
                n_ports,
            )
            if trans_port is not None:
                transshipment_port_indices = [trans_port]
                connecting_service_deadline_h = max(
                    0.0,
                    nominal_arrivals[trans_port] - connecting_service_slack_h,
                )

        container = Container(
            id=f"C{i + 1:03d}",
            origin_idx=origin_idx,
            destination_idx=destination_idx,
            promised_arrival_h=round(float(promised_arrival_h), 2),
            penalty_delay=round(float(penalty_delay), 2),
            penalty_misconnect=round(float(penalty_misconnect), 2),
            quantity_teu=quantity_teu,
            transshipment_port_indices=transshipment_port_indices,
            priority=_draw_priority(rng, priorities),
            connecting_service_deadline_h=connecting_service_deadline_h,
        )
        containers.append(container)

    return containers


# =============================================================================
# 4. PRESET GENERATORS
# =============================================================================

def generate_small_test_set(
    *,
    ports: list[str],
    distance_matrix_nm: list[list[float]],
    seed: int = DEFAULT_RANDOM_SEED,
    notebook_compatible: bool = True,
    expected_initial_delay_h: float = DEFAULT_EXPECTED_INITIAL_DELAY_H,
) -> list[Container]:
    """
    Generate a small default test set for smoke tests and debugging.
    """
    return generate_containers(
        ports=ports,
        distance_matrix_nm=distance_matrix_nm,
        n=5,
        seed=seed,
        promised_slack_h=24.0,
        notebook_compatible=notebook_compatible,
        expected_initial_delay_h=expected_initial_delay_h,
    )


def generate_transshipment_test_set(
    *,
    ports: list[str],
    distance_matrix_nm: list[list[float]],
    seed: int = DEFAULT_RANDOM_SEED,
    n: int = 5,
    transshipment_probability: float = 0.4,
    expected_initial_delay_h: float = DEFAULT_EXPECTED_INITIAL_DELAY_H,
) -> list[Container]:
    """
    Generate a small mixed direct/transshipment demand set.

    This is mainly intended for testing and debugging the transshipment
    misconnection logic.
    """
    return generate_containers(
        ports=ports,
        distance_matrix_nm=distance_matrix_nm,
        n=n,
        seed=seed,
        promised_slack_h=24.0,
        notebook_compatible=True,
        transshipment_probability=transshipment_probability,
        expected_initial_delay_h=expected_initial_delay_h,
    )