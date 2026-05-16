from __future__ import annotations

import traceback
import pandas as pd

from core.entities import VesselConfig
from data.base_instance import BASE_DISTANCE_MATRIX_NM, BASE_PORTS
from data.fleet_instance import build_fleet_instance
from data.instance_generator import generate_containers
from model.base import SolveOptions
from model.fleet_solver import FleetSolver
from model.xpress_solver import XpressSolver
from reporting.export import save_dataframe


def build_scaled_fleet(
    n_vessels: int,
    containers_per_vessel: int,
    *,
    base_seed: int = 42,
):
    vessel_configs = []

    for v in range(n_vessels):
        containers = generate_containers(
            ports=BASE_PORTS,
            distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
            n=containers_per_vessel,
            seed=base_seed + 100 * v,
            notebook_compatible=True,
        )

        vessel_configs.append(
            VesselConfig(
                vessel_id=f"V{v+1}",
                containers=containers,
                initial_delay_h=48.0,
                alpha=0.5,
                allow_swap=True,
                max_skip=1,
                include_fueleu_penalty=False,
                metadata={"seed": base_seed + 100 * v},
            )
        )

    return build_fleet_instance(
        vessel_configs,
        fleet_id=f"fleet_{n_vessels}v_{containers_per_vessel}c",
        metadata={
            "n_vessels": n_vessels,
            "containers_per_vessel": containers_per_vessel,
        },
    )


def main():
    print("=" * 88)
    print("FLEET LICENSE / SCALE SMOKE TEST (XPRESS COMMUNITY)")
    print("=" * 88)

    xpress_solver = XpressSolver()
    fleet_solver = FleetSolver(vessel_solver=xpress_solver)

    options = SolveOptions(
        time_limit_s=60,
        mip_gap=1e-6,
        log_to_console=False,
    )

    fleet_sizes = [1, 2, 3, 5, 10, 20]
    containers_grid = [5, 10, 20, 40, 80, 100, 200, 500]

    rows = []

    for n_vessels in fleet_sizes:
        for n_cont in containers_grid:
            case_id = f"{n_vessels} vessels x {n_cont} containers/vessel"
            print(f"\n--- {case_id} ---")

            try:
                fleet = build_scaled_fleet(
                    n_vessels=n_vessels,
                    containers_per_vessel=n_cont,
                )

                fleet_solution = fleet_solver.solve(fleet, options=options)

                first_bad_vessel = None
                first_bad_status = None
                for i, sol in enumerate(fleet_solution.vessel_solutions, start=1):
                    if not sol.feasible or not sol.optimal:
                        first_bad_vessel = i
                        first_bad_status = (
                            sol.solver_stats.status if sol.solver_stats else None
                        )
                        break

                row = {
                    "n_vessels": n_vessels,
                    "containers_per_vessel": n_cont,
                    "total_containers": n_vessels * n_cont,
                    "success": True,
                    "fleet_feasible": fleet_solution.feasible,
                    "fleet_optimal": fleet_solution.optimal,
                    "fleet_objective_value": fleet_solution.fleet_objective_value,
                    "avg_runtime_s": fleet_solution.avg_runtime_s,
                    "total_runtime_s": fleet_solution.total_runtime_s,
                    "first_bad_vessel": first_bad_vessel,
                    "first_bad_status": first_bad_status,
                    "error_message": None,
                }

                print(
                    f"success=True  feasible={fleet_solution.feasible}  "
                    f"optimal={fleet_solution.optimal}  "
                    f"avg_runtime={fleet_solution.avg_runtime_s}"
                )

            except Exception as exc:
                row = {
                    "n_vessels": n_vessels,
                    "containers_per_vessel": n_cont,
                    "total_containers": n_vessels * n_cont,
                    "success": False,
                    "fleet_feasible": None,
                    "fleet_optimal": None,
                    "fleet_objective_value": None,
                    "avg_runtime_s": None,
                    "total_runtime_s": None,
                    "first_bad_vessel": None,
                    "first_bad_status": None,
                    "error_message": str(exc),
                }

                print(f"success=False  error={exc}")
                traceback.print_exc()

            rows.append(row)

    df = pd.DataFrame(rows)

    print("\nSUMMARY")
    print(df.to_string(index=False))

    save_dataframe(
        df,
        output_dir="results/tables/fleet_license_smoke",
        filename_stem="fleet_license_smoke",
        index=False,
    )

    print("\nSaved to results/tables/fleet_license_smoke/")
    print("=" * 88)
    print("FLEET LICENSE / SCALE SMOKE COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()