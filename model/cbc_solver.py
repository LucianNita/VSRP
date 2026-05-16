# =============================================================================
# CBC solver backend for the Vessel Schedule Recovery Problem (VSRP).
#
# Purpose
# -------
# This backend solves the VSRP through an open-source workflow:
# - build the formulation with the PuLP-based MIP builder
# - export the model to MPS
# - read and solve the MPS model with CBC via python-mip
#
# Architectural role
# ------------------
# This file provides an open-source benchmark backend that is independent
# of the native Xpress API. It is intended primarily for:
# - objective consistency checks
# - runtime benchmarking
# - gap and bound comparisons
#
# Current limitation
# ------------------
# The CBC backend currently returns a partial canonical solution:
# - objective value
# - runtime
# - best bound
# - gap
# - selected solver statistics
#
# It does not yet reconstruct:
# - route legs
# - timeline
# - strategy decisions
# - container outcomes
# - validation or emissions summaries
# =============================================================================

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from core.entities import SolverStats, VSRPInstance, VSRPSolution
from model.base import BaseSolver, SolveOptions, build_empty_solution
from model.mip_builder import PULP_AVAILABLE, VSRPMIPBuilder

try:
    import mip

    CBC_AVAILABLE = True
except ImportError:
    mip = None
    CBC_AVAILABLE = False


class CBCSolver(BaseSolver):
    solver_name = "CBC"

    @property
    def is_available(self) -> bool:
        """
        Whether CBC can be used in the current environment.

        This backend requires both:
        - `python-mip`
        - `pulp`
        """
        return CBC_AVAILABLE and PULP_AVAILABLE

    def solve(
        self,
        instance: VSRPInstance,
        options: SolveOptions | None = None,
    ) -> VSRPSolution:
        """
        Solve one canonical VSRP instance with CBC.

        The workflow is:
        1. build the PuLP formulation
        2. export to MPS
        3. load the MPS model into CBC
        4. solve and extract a partial canonical solution
        """
        if not self.is_available:
            missing = []
            if not CBC_AVAILABLE:
                missing.append("python-mip")
            if not PULP_AVAILABLE:
                missing.append("pulp")
            return build_empty_solution(
                instance=instance,
                solver_name=self.solver_name,
                status="UNAVAILABLE",
                feasible=False,
                optimal=False,
                message=f"Missing dependencies: {', '.join(missing)}",
            )

        options = options or SolveOptions()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = Path(tmpdir) / f"{instance.instance_id}.mps"

                self._export_model_to_mps(instance, mps_path)

                cbc_model = mip.Model(solver_name=mip.CBC)
                cbc_model.verbose = 1 if options.log_to_console else 0
                cbc_model.read(str(mps_path))
                cbc_model.max_seconds = float(options.time_limit_s)
                cbc_model.max_mip_gap = float(options.mip_gap)

                t0 = time.perf_counter()
                status = cbc_model.optimize()
                runtime_s = time.perf_counter() - t0

                # python-mip does not currently expose a clean
                # time-to-first-feasible metric.
                time_to_first_feasible_s = None

                solution = self._extract_cbc_solution(
                    instance=instance,
                    model=cbc_model,
                    runtime_s=runtime_s,
                    status=status,
                    time_to_first_feasible_s=time_to_first_feasible_s,
                )
                return solution

        except Exception as exc:
            return build_empty_solution(
                instance=instance,
                solver_name=self.solver_name,
                status="ERROR",
                feasible=False,
                optimal=False,
                message=str(exc),
            )

    # -----------------------------------------------------------------
    # Export helper
    # -----------------------------------------------------------------

    def _export_model_to_mps(
        self,
        instance: VSRPInstance,
        mps_path: Path,
    ) -> None:
        """
        Build the VSRP formulation with PuLP and export it to MPS.
        """
        builder = VSRPMIPBuilder()
        builder.write_mps(instance, mps_path)

    # -----------------------------------------------------------------
    # Solution extraction
    # -----------------------------------------------------------------

    def _extract_cbc_solution(
        self,
        *,
        instance: VSRPInstance,
        model,
        runtime_s: float,
        status,
        time_to_first_feasible_s: float | None,
    ) -> VSRPSolution:
        """
        Extract a partial canonical solution from the CBC solve result.

        The current CBC backend focuses on benchmark-relevant metadata
        rather than full route reconstruction.
        """
        status_text = str(status)

        feasible = status in (
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        )
        optimal = status == mip.OptimizationStatus.OPTIMAL

        objective_value = None
        best_bound = None
        mip_gap = None
        node_count = None

        try:
            if model.objective_value is not None:
                objective_value = float(model.objective_value)
        except Exception:
            pass

        try:
            if model.objective_bound is not None:
                best_bound = float(model.objective_bound)
        except Exception:
            pass

        try:
            if model.gap is not None:
                mip_gap = float(model.gap)
        except Exception:
            pass

        try:
            node_count = int(getattr(model, "num_nodes", None))
        except Exception:
            node_count = None

        # If CBC does not provide a direct gap value, reconstruct it
        # from objective and bound when possible.
        if mip_gap is None and objective_value is not None and best_bound is not None:
            try:
                if abs(objective_value) > 1e-12:
                    mip_gap = abs(objective_value - best_bound) / abs(
                        objective_value
                    )
            except Exception:
                pass

        metadata = {
            "export_based_backend": True,
            "partial_solution_backend": True,
            "objective_extracted": objective_value is not None,
            "best_bound_extracted": best_bound is not None,
            "mip_gap_extracted": mip_gap is not None,
            "time_to_first_feasible_note": (
                "python-mip does not expose time-to-first-feasible; "
                "reported as None"
            ),
        }

        solver_stats = SolverStats(
            solver_name=self.solver_name,
            status=status_text,
            runtime_s=runtime_s,
            mip_gap=mip_gap,
            best_bound=best_bound,
            time_to_first_feasible_s=time_to_first_feasible_s,
            node_count=node_count,
            iteration_count=None,
            feasible=feasible,
            optimal=optimal,
            raw_status_code=status_text,
            message=None,
        )

        return VSRPSolution(
            instance_id=instance.instance_id,
            objective_value=objective_value if feasible else None,
            route_legs=[],
            skipped_port_indices=[],
            swapped_port_indices=[],
            strategy_decisions=[],
            timeline=[],
            container_outcomes={},
            solver_stats=solver_stats,
            validation=None,
            emissions=None,
            metadata=metadata,
        )