# =============================================================================
# Base solver interface for the Vessel Schedule Recovery Problem (VSRP).
#
# Purpose
# -------
# This module defines the common contract that all solver backends must
# satisfy. It is the key abstraction that allows experiments, benchmarking,
# and reporting to remain solver-agnostic.
#
# Architectural role
# ------------------
# The rest of the codebase should interact with solver backends through:
# - canonical `VSRPInstance` inputs
# - canonical `VSRPSolution` outputs
#
# Concrete backends such as Xpress, HiGHS, and CBC inherit from the
# abstract base class defined here.
# =============================================================================

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.entities import SolverStats, VSRPInstance, VSRPSolution


# =============================================================================
# 1. SOLVER OPTIONS
# =============================================================================

@dataclass(slots=True)
class SolveOptions:
    """
    Generic solve controls shared across solver backends.

    These options provide a common configuration interface for the rest
    of the codebase, even though different solver APIs may implement or
    interpret them slightly differently.

    Attributes
    ----------
    time_limit_s : int, default=120
        Maximum solve time in seconds.
    mip_gap : float, default=0.01
        Relative MIP gap target.
    log_to_console : bool, default=False
        Whether solver logging should be printed to the console.
    random_seed : int | None, default=None
        Optional backend random seed where supported.
    """
    time_limit_s: int = 120
    mip_gap: float = 0.01
    log_to_console: bool = False
    random_seed: int | None = None


# =============================================================================
# 2. BASE SOLVER CONTRACT
# =============================================================================

class BaseSolver(ABC):
    """
    Abstract base class for all VSRP solver backends.

    Required behavior
    -----------------
    Every concrete solver backend should:
    - accept a canonical `VSRPInstance`
    - solve the instance using its own backend API
    - return a canonical `VSRPSolution`

    This abstraction allows experiment code to treat all solvers
    uniformly, regardless of whether they are:
    - native commercial backends
    - open-source backends
    - full-solution extractors
    - partial benchmark-only backends
    """

    solver_name: str = "BaseSolver"

    @abstractmethod
    def solve(
        self,
        instance: VSRPInstance,
        options: SolveOptions | None = None,
    ) -> VSRPSolution:
        """
        Solve one canonical VSRP instance and return a canonical solution.
        """
        raise NotImplementedError

    @property
    def is_available(self) -> bool:
        """
        Whether this solver backend is available in the current environment.

        Concrete subclasses should override this property when availability
        depends on optional imports or external solver installation.
        """
        return True


# =============================================================================
# 3. SHARED HELPERS
# =============================================================================

def build_empty_solution(
    *,
    instance: VSRPInstance,
    solver_name: str,
    status: str,
    feasible: bool = False,
    optimal: bool = False,
    runtime_s: float | None = None,
    mip_gap: float | None = None,
    best_bound: float | None = None,
    raw_status_code=None,
    message: str | None = None,
) -> VSRPSolution:
    """
    Build a canonical empty solution object.

    This helper is used when a backend:
    - is unavailable
    - fails during solve
    - returns no feasible solution
    - cannot produce a valid extracted result

    Returns
    -------
    VSRPSolution
        Canonical solution object with empty route/output fields and
        populated solver-status metadata.
    """
    return VSRPSolution(
        instance_id=instance.instance_id,
        objective_value=None,
        route_legs=[],
        skipped_port_indices=[],
        swapped_port_indices=[],
        strategy_decisions=[],
        timeline=[],
        container_outcomes={},
        solver_stats=SolverStats(
            solver_name=solver_name,
            status=status,
            runtime_s=runtime_s,
            mip_gap=mip_gap,
            best_bound=best_bound,
            feasible=feasible,
            optimal=optimal,
            raw_status_code=raw_status_code,
            message=message,
        ),
        validation=None,
        emissions=None,
        metadata={},
    )