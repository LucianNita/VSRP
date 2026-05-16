from __future__ import annotations

from core.entities import FleetInstance, FleetSolution, VSRPSolution
from model.base import BaseSolver, SolveOptions


class FleetSolver:
    """
    Fleet-level solver for the Vessel Schedule Recovery Problem.

    Solves each vessel sub-problem independently using any BaseSolver
    backend and aggregates the results into a FleetSolution.

    Decomposition rationale
    -----------------------
    The multi-vessel recovery problem decomposes into independent
    per-vessel sub-problems when there are no coupling constraints
    between vessels. In the current formulation, vessels share the
    same port network and distance matrix but have no inter-vessel
    coordination constraints such as shared port capacity windows
    or synchronized departure times.

    Under this decomposition, the fleet objective is the sum of
    per-vessel weighted objectives, and the optimal fleet solution
    is obtained by solving each vessel optimally in isolation.

    This approach is architecturally consistent with the notebook's
    EnhancedVSRPModel, which builds one large joint Xpress model
    but with fully separable per-vessel variable blocks and no
    cross-vessel constraints.

    Parameters
    ----------
    vessel_solver : BaseSolver
        Solver backend used for each per-vessel sub-problem.
        Any backend implementing BaseSolver is accepted, including
        XpressSolver, HighsSolver, and CBCSolver.
    """

    def __init__(self, vessel_solver: BaseSolver) -> None:
        self.vessel_solver = vessel_solver

    @property
    def solver_name(self) -> str:
        """Human-readable solver name including the backend name."""
        return f"Fleet[{self.vessel_solver.solver_name}]"

    @property
    def is_available(self) -> bool:
        """Whether the underlying vessel solver backend is available."""
        return self.vessel_solver.is_available

    def solve(
        self,
        fleet: FleetInstance,
        options: SolveOptions | None = None,
    ) -> FleetSolution:
        """
        Solve each vessel sub-problem and aggregate into a FleetSolution.

        Each vessel in the fleet is solved independently using the
        configured vessel solver backend. The resulting per-vessel
        VSRPSolution objects are collected and wrapped in a FleetSolution
        with an aggregated fleet objective value.

        Parameters
        ----------
        fleet : FleetInstance
            Fleet-level instance containing one VSRPInstance per vessel.
        options : SolveOptions | None, default=None
            Solve controls applied uniformly to every per-vessel solve.
            Uses default SolveOptions if not provided.

        Returns
        -------
        FleetSolution
            Aggregated fleet solution with one VSRPSolution per vessel
            and a fleet-level objective equal to the sum of per-vessel
            weighted objectives.
        """
        options = options or SolveOptions()
        vessel_solutions: list[VSRPSolution] = []

        for vessel_instance in fleet.vessel_instances:
            solution = self.vessel_solver.solve(
                vessel_instance,
                options=options,
            )
            vessel_solutions.append(solution)

        fleet_objective: float | None = None
        if all(s.objective_value is not None for s in vessel_solutions):
            fleet_objective = sum(
                s.objective_value
                for s in vessel_solutions
            )

        return FleetSolution(
            fleet_id=fleet.fleet_id,
            vessel_solutions=vessel_solutions,
            fleet_objective_value=fleet_objective,
            solver_name=self.solver_name,
            metadata={
                "n_vessels": fleet.n_vessels,
                "vessel_instance_ids": [
                    vi.instance_id for vi in fleet.vessel_instances
                ],
            },
        )