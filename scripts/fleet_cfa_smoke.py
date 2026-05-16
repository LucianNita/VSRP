from __future__ import annotations

import numpy as np

from core.simulation import UncertaintyConfig
from data.base_instance import BASE_PORTS
from experiments.fleet_benchmark import build_canonical_fleet
from experiments.fleet_cfa import (
    FleetCFAPolicyResult,
    compute_fleet_tail_risk_summary,
    initialize_fleet_theta,
    test_fleet_cfa_policy,
    train_fleet_cfa,
)
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from reporting.cfa_plots import plot_theta_evolution
from reporting.export import save_dataframe


def _print_policy_summary(name: str, result: FleetCFAPolicyResult) -> None:
    df = result.results_df
    if df.empty:
        print(f"\n{name}: <empty>")
        return

    print(f"\n{name}")
    print("-" * len(name))
    print(f"  episodes                        : {len(df)}")
    print(
        f"  avg fleet realized service cost : "
        f"${df['realized_service_cost_usd'].mean():,.2f}"
    )
    print(
        f"  avg fleet realized missed       : "
        f"{df['realized_total_missed'].mean():.3f}"
    )
    print(
        f"  avg fleet realized ETS (EUR)    : "
        f"{df['realized_ets_cost_eur'].mean():.2f}"
    )
    print(
        f"  final theta mean                : "
        f"{df['theta_mean_h'].iloc[-1]:.4f}h"
    )
    print(
        f"  final theta max                 : "
        f"{df['theta_max_h'].iloc[-1]:.4f}h"
    )
    print(
        f"  NOTE: fleet_objective_value is NOT directly comparable "
        f"across policies"
    )
    print(
        f"        Use realized_service_cost_usd for fair comparison."
    )


def _print_theta(theta: dict[int, float]) -> None:
    nonzero = [(p, v) for p, v in theta.items() if abs(v) > 1e-9]
    if not nonzero:
        print("  <all zero>")
        return
    for p, val in sorted(nonzero):
        print(f"  Port {p} ({BASE_PORTS[p] if p < len(BASE_PORTS) else p}): "
              f"{val:.4f}h")


def _print_tail_risk(name: str, summary: dict) -> None:
    if not summary:
        print(f"  {name}: <no data>")
        return
    print(f"\n  {name}:")
    print(f"    episodes : {summary.get('n_episodes', 'N/A')}")
    print(f"    mean     : ${summary.get('mean', 0):,.2f}")
    print(f"    std      : ${summary.get('std', 0):,.2f}")
    print(f"    p50      : ${summary.get('p50', 0):,.2f}")
    print(f"    p90      : ${summary.get('p90', 0):,.2f}")
    print(f"    p95      : ${summary.get('p95', 0):,.2f}")
    if "tail_mean_above_p95" in summary:
        print(
            f"    tail mean (above p95) : "
            f"${summary['tail_mean_above_p95']:,.2f}"
        )
    if "avg_realized_ets_cost_eur" in summary:
        print(
            f"    avg ETS (EUR) : "
            f"{summary['avg_realized_ets_cost_eur']:,.2f}"
        )


def main() -> None:
    print("=" * 88)
    print("FLEET CFA SMOKE TEST")
    print("=" * 88)

    solver = XpressSolver()
    options = SolveOptions(time_limit_s=30, mip_gap=0.01, log_to_console=False)

    uncertainty_config = UncertaintyConfig(
        est_delay_mean_h=48.0,
        est_delay_std_h=12.0,
        real_delay_mean_h=55.0,
        real_delay_std_h=10.0,
        delay_min_h=20.0,
        delay_max_h=80.0,
        port_handling_mean=1.0,
        port_handling_std=0.1,
        handling_min=0.8,
        handling_max=1.5,
        weather_factor_mean=1.0,
        weather_factor_std=0.05,
        weather_min=0.7,
        weather_max=1.0,
        carbon_price_mean_eur=65.0,
        carbon_price_std_eur=15.0,
        carbon_price_min_eur=30.0,
        carbon_price_max_eur=130.0,
        fuel_price_mean_usd=600.0,
        fuel_price_std_usd=80.0,
        fuel_price_min_usd=400.0,
        fuel_price_max_usd=900.0,
    )

    # Use Case1_Delayed as the base fleet for CFA
    base_fleet = build_canonical_fleet("Case1_Delayed")

    print(f"\nBase fleet: {base_fleet.fleet_id}")
    print(f"  Vessels  : {base_fleet.n_vessels}")
    print(f"  Ports    : {base_fleet.n_ports}")
    print(f"  Containers: {base_fleet.total_containers}")

    # ------------------------------------------------------------------
    # Baseline theta
    # ------------------------------------------------------------------
    baseline_theta = initialize_fleet_theta(base_fleet)
    print("\nBaseline theta (all zero):")
    _print_theta(baseline_theta)

    # ------------------------------------------------------------------
    # Train additive policy
    # ------------------------------------------------------------------
    print("\nTraining additive policy (10 episodes)...")
    additive_train = train_fleet_cfa(
        base_fleet,
        solver,
        n_episodes=30,
        solve_options=options,
        seed=42,
        update_policy="additive",
        step_size=1.0,
        uncertainty_config=uncertainty_config,
    )
    additive_theta = additive_train.theta_history[-1]
    print("Trained theta (additive):")
    _print_theta(additive_theta)

    # ------------------------------------------------------------------
    # Train decay policy
    # ------------------------------------------------------------------
    print("\nTraining decay policy (10 episodes)...")
    decay_train = train_fleet_cfa(
        base_fleet,
        solver,
        n_episodes=30,
        solve_options=options,
        seed=42,
        update_policy="decay",
        step_size=1.0,
        step_down=0.25,
        uncertainty_config=uncertainty_config,
    )
    decay_theta = decay_train.theta_history[-1]
    print("Trained theta (decay):")
    _print_theta(decay_theta)

    # ------------------------------------------------------------------
    # Train SPSA policy
    # ------------------------------------------------------------------
    print("\nTraining SPSA policy (10 episodes)...")
    spsa_train = train_fleet_cfa(
        base_fleet,
        solver,
        n_episodes=30,
        solve_options=options,
        seed=42,
        update_policy="spsa",
        step_size=0.5,
        perturbation_size=2.0,
        uncertainty_config=uncertainty_config,
    )
    spsa_theta = spsa_train.theta_history[-1]
    print("Trained theta (SPSA):")
    _print_theta(spsa_theta)

    # ------------------------------------------------------------------
    # Test all policies
    # ------------------------------------------------------------------
    print("\nTesting all policies (20 episodes each)...")

    baseline_test = test_fleet_cfa_policy(
        base_fleet, solver, baseline_theta,
        n_episodes=20, solve_options=options, seed=123,
        policy_name="fleet_baseline",
        uncertainty_config=uncertainty_config,
    )
    additive_test = test_fleet_cfa_policy(
        base_fleet, solver, additive_theta,
        n_episodes=20, solve_options=options, seed=123,
        policy_name="fleet_additive",
        uncertainty_config=uncertainty_config,
    )
    decay_test = test_fleet_cfa_policy(
        base_fleet, solver, decay_theta,
        n_episodes=20, solve_options=options, seed=123,
        policy_name="fleet_decay",
        uncertainty_config=uncertainty_config,
    )
    spsa_test = test_fleet_cfa_policy(
        base_fleet, solver, spsa_theta,
        n_episodes=20, solve_options=options, seed=123,
        policy_name="fleet_spsa",
        uncertainty_config=uncertainty_config,
    )

    # ------------------------------------------------------------------
    # Print training summaries
    # ------------------------------------------------------------------
    _print_policy_summary("TRAIN — Additive", additive_train)
    _print_policy_summary("TRAIN — Decay", decay_train)
    _print_policy_summary("TRAIN — SPSA", spsa_train)

    # ------------------------------------------------------------------
    # Print test summaries
    # ------------------------------------------------------------------
    _print_policy_summary("TEST — Baseline", baseline_test)
    _print_policy_summary("TEST — Additive", additive_test)
    _print_policy_summary("TEST — Decay", decay_test)
    _print_policy_summary("TEST — SPSA", spsa_test)

    # ------------------------------------------------------------------
    # Tail risk
    # ------------------------------------------------------------------
    print("\nFleet Tail Risk Summary (realized_service_cost_usd):")
    for name, result in [
        ("Baseline", baseline_test),
        ("Additive", additive_test),
        ("Decay", decay_test),
        ("SPSA", spsa_test),
    ]:
        summary = compute_fleet_tail_risk_summary(result.results_df)
        _print_tail_risk(name, summary)

    # ------------------------------------------------------------------
    # Policy comparison table
    # ------------------------------------------------------------------
    print("\nPolicy Comparison (realized metrics only):")
    print("-" * 60)
    for name, result in [
        ("Baseline", baseline_test),
        ("Additive", additive_test),
        ("Decay", decay_test),
        ("SPSA", spsa_test),
    ]:
        df = result.results_df
        missed = df["realized_total_missed"].mean()
        cost = df["realized_service_cost_usd"].mean()
        ets = df["realized_ets_cost_eur"].mean()
        print(
            f"  {name:<12} | missed={missed:.3f}  "
            f"fleet_svc=${cost:,.0f}  "
            f"ets=€{ets:.0f}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    for name, result in [
        ("additive_train", additive_train),
        ("decay_train", decay_train),
        ("spsa_train", spsa_train),
        ("baseline_test", baseline_test),
        ("additive_test", additive_test),
        ("decay_test", decay_test),
        ("spsa_test", spsa_test),
    ]:
        save_dataframe(
            result.results_df,
            output_dir="results/tables/fleet_cfa_smoke",
            filename_stem=f"fleet_cfa_{name}_episodes",
            index=False,
        )
        save_dataframe(
            result.per_vessel_df,
            output_dir="results/tables/fleet_cfa_smoke",
            filename_stem=f"fleet_cfa_{name}_per_vessel",
            index=False,
        )

    # ------------------------------------------------------------------
    # Theta evolution plots
    # ------------------------------------------------------------------
    print("\nGenerating theta evolution plots...")
    for policy_name, train_result in [
        ("additive", additive_train),
        ("decay", decay_train),
        ("spsa", spsa_train),
    ]:
        plot_theta_evolution(
            train_result.theta_history,
            ports=base_fleet.ports,
            output_dir="results/figures/fleet_cfa_smoke",
            filename_stem=f"fleet_theta_evolution_{policy_name}",
            title=f"Fleet Theta Evolution — {policy_name.capitalize()} Update",
        )

    print("\nSaved to:")
    print("  results/tables/fleet_cfa_smoke/")
    print("  results/figures/fleet_cfa_smoke/")
    print("=" * 88)
    print("FLEET CFA SMOKE TEST COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()