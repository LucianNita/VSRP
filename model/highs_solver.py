# =============================================================================
# HiGHS solver backend for the Vessel Schedule Recovery Problem (VSRP).
#
# Purpose
# -------
# This backend solves the VSRP through an open-source workflow:
# - build the formulation with the PuLP-based MIP builder
# - export the model to MPS
# - read and solve the MPS model with HiGHS via highspy
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
# The HiGHS backend currently returns a partial canonical solution:
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
    import highspy as hg

    HIGHS_AVAILABLE = True
except ImportError:
    hg = None
    HIGHS_AVAILABLE = False


class HighsSolver(BaseSolver):
    solver_name = "HiGHS"

    @property
    def is_available(self) -> bool:
        """
        Whether HiGHS can be used in the current environment.

        This backend requires both:
        - `highspy`
        - `pulp`
        """
        return HIGHS_AVAILABLE and PULP_AVAILABLE

    def solve(
        self,
        instance: VSRPInstance,
        options: SolveOptions | None = None,
    ) -> VSRPSolution:
        """
        Solve one canonical VSRP instance with HiGHS.

        The workflow is:
        1. build the PuLP formulation
        2. export to MPS
        3. load the MPS model into HiGHS
        4. solve and extract a partial canonical solution
        """
        if not self.is_available:
            missing = []
            if not HIGHS_AVAILABLE:
                missing.append("highspy")
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

                highs = hg.Highs()
                if not options.log_to_console:
                    highs.silent()

                highs.readModel(str(mps_path))
                highs.setOptionValue(
                    "time_limit",
                    float(options.time_limit_s),
                )
                highs.setOptionValue(
                    "mip_rel_gap",
                    float(options.mip_gap),
                )

                # Attempt to capture time to first feasible incumbent via log callback.
                first_feasible_time: list[float] = []
                t0 = time.perf_counter()

                def _log_callback(log_type, msg, data):
                    """
                    HiGHS log callback used to approximate time to first feasible.

                    If the callback receives a message indicating a solution or
                    feasible incumbent, the elapsed time is recorded.
                    """
                    if not first_feasible_time:
                        lower = msg.lower() if msg else ""
                        if "solution" in lower or "feasible" in lower:
                            first_feasible_time.append(
                                time.perf_counter() - t0
                            )

                try:
                    highs.setLogCallback(_log_callback)
                except Exception:
                    # Some highspy versions do not support callback registration.
                    pass

                highs.run()
                runtime_s = time.perf_counter() - t0

                time_to_first = (
                    first_feasible_time[0]
                    if first_feasible_time
                    else None
                )

                solution = self._extract_highs_solution(
                    instance=instance,
                    highs=highs,
                    runtime_s=runtime_s,
                    time_to_first_feasible_s=time_to_first,
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

    def _extract_highs_solution(
        self,
        *,
        instance: VSRPInstance,
        highs,
        runtime_s: float,
        time_to_first_feasible_s: float | None,
    ) -> VSRPSolution:
        """
        Extract a partial canonical solution from the HiGHS solve result.

        The current HiGHS backend focuses on benchmark-relevant metadata
        rather than full route reconstruction.
        """
        model_status = self._safe_call(highs.getModelStatus)
        status_text = str(model_status)

        objective_value = self._extract_objective_value(highs)
        best_bound = self._extract_best_bound(highs)
        mip_gap = self._extract_mip_gap(highs, objective_value, best_bound)
        node_count = self._extract_node_count(highs)
        iteration_count = self._extract_iteration_count(highs)

        feasible = self._infer_feasibility(status_text, objective_value)
        optimal = self._infer_optimality(status_text, mip_gap)

        solver_stats = SolverStats(
            solver_name=self.solver_name,
            status=status_text,
            runtime_s=runtime_s,
            mip_gap=mip_gap,
            best_bound=best_bound,
            time_to_first_feasible_s=time_to_first_feasible_s,
            node_count=node_count,
            iteration_count=iteration_count,
            feasible=feasible,
            optimal=optimal,
            raw_status_code=status_text,
            message=None,
        )

        metadata = {
            "export_based_backend": True,
            "partial_solution_backend": True,
            "objective_extracted": objective_value is not None,
            "best_bound_extracted": best_bound is not None,
            "mip_gap_extracted": mip_gap is not None,
        }

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

    # -----------------------------------------------------------------
    # HiGHS stat helpers
    # -----------------------------------------------------------------

    def _safe_call(self, func):
        """
        Call a HiGHS API method safely, returning None on failure.
        """
        try:
            return func()
        except Exception:
            return None

    def _get_info_value(self, highs, key: str):
        """
        Read one HiGHS information value by key.

        Some highspy APIs return tuples, so this helper normalizes that behavior.
        """
        try:
            result = highs.getInfoValue(key)
            if isinstance(result, tuple):
                return result[1] if len(result) >= 2 else None
            return result
        except Exception:
            return None

    def _extract_objective_value(self, highs):
        """
        Extract the objective value from HiGHS using the most reliable
        available API path.
        """
        for key in ["objective_function_value"]:
            value = self._get_info_value(highs, key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        try:
            return float(highs.getObjectiveValue())
        except Exception:
            return None

    def _extract_best_bound(self, highs):
        """
        Extract the best available dual/objective bound from HiGHS.
        """
        for key in ["mip_dual_bound", "objective_bound"]:
            value = self._get_info_value(highs, key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass
        return None

    def _extract_mip_gap(self, highs, objective_value, best_bound):
        """
        Extract the relative MIP gap, falling back to reconstruction from
        objective and bound when necessary.
        """
        value = self._get_info_value(highs, "mip_gap")
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
        try:
            if objective_value is not None and best_bound is not None:
                if abs(objective_value) <= 1e-12 and abs(best_bound) <= 1e-12:
                    return 0.0
                if abs(objective_value) > 1e-12:
                    return abs(objective_value - best_bound) / abs(
                        objective_value
                    )
        except Exception:
            pass
        return None

    def _extract_node_count(self, highs):
        """
        Extract node count if exposed by the installed HiGHS build.
        """
        for key in ["mip_node_count", "node_count"]:
            value = self._get_info_value(highs, key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return None

    def _extract_iteration_count(self, highs):
        """
        Extract iteration count if exposed by the installed HiGHS build.
        """
        for key in [
            "simplex_iteration_count",
            "ipm_iteration_count",
            "iteration_count",
        ]:
            value = self._get_info_value(highs, key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return None

    def _infer_feasibility(self, status_text, objective_value) -> bool:
        """
        Infer whether the solve produced a feasible solution.

        If an objective value exists, feasibility is assumed. Otherwise,
        fallback to status-text interpretation.
        """
        if objective_value is not None:
            return True
        lower = status_text.lower()
        return "feasible" in lower or "optimal" in lower

    def _infer_optimality(self, status_text, mip_gap) -> bool:
        """
        Infer whether the solve should be treated as optimal.

        This uses:
        - explicit status text when available
        - near-zero gap as a secondary indicator
        """
        if "optimal" in status_text.lower():
            return True
        try:
            if mip_gap is not None and float(mip_gap) <= 1e-9:
                return True
        except Exception:
            pass
        return False