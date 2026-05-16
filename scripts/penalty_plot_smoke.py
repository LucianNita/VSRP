from __future__ import annotations

import pandas as pd

from reporting.penalty_sensitivity_plots import plot_penalty_sweep_overview


def main() -> None:
    swap_df = pd.read_csv(
        "results/tables/penalty_sensitivity_smoke/swap_penalty_multi_seed.csv"
    )
    speed_df = pd.read_csv(
        "results/tables/penalty_sensitivity_smoke/speedup_penalty_multi_seed.csv"
    )

    plot_penalty_sweep_overview(
        swap_df,
        penalty_name="swap_usd",
        output_dir="results/figures/penalty_sensitivity_smoke",
        filename_stem="swap_penalty_overview",
    )

    plot_penalty_sweep_overview(
        speed_df,
        penalty_name="speed_up_usd",
        output_dir="results/figures/penalty_sensitivity_smoke",
        filename_stem="speedup_penalty_overview",
    )

    print("Penalty sensitivity figures generated.")


if __name__ == "__main__":
    main()