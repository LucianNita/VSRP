# scripts/cfa_smoke.py
# =============================================================================
# Minimal smoke test for the refactored CFA pipeline.
#
# Updated for Round 4:
#   - UncertaintyConfig passed to train_cfa and test_cfa_policy
#   - SPSA policy trained and tested
#   - plot_cfa_tail_risk and plot_regulatory_compliance_and_ets called
#   - notebook_compatible=True for penalty alignment (Fix F10)
#   - objective_comparable flag noted in summary output
# =============================================================================

from __future__ import annotations

import numpy as np

from data.base_instance import (
    BASE_DISTANCE_MATRIX_NM,
    BASE_PORTS,
    build_base_instance,
)
from data.instance_generator import generate_containers
from core.simulation import UncertaintyConfig
from experiments.cfa import (
    initialize_theta,
    test_cfa_policy,
    train_cfa,
    compute_tail_risk_summary,
)
from model.base import SolveOptions
from model.xpress_solver import XpressSolver
from reporting.cfa_plots import (
    plot_theta_evolution,
    plot_cfa_policy_comparison,
    plot_cfa_summary_bars,
    plot_cfa_tail_risk,
    plot_regulatory_compliance_and_ets,
)


def print_policy_summary(name: str, df) -> None:
    print(f"\n{name}")
    print("-" * len(name))

    if df.empty:
        print("  <empty>")
        return

    print(f"  episodes                      : {len(df)}")
    print(
        f"  avg realized service cost     : "
        f"{df['realized_service_cost_usd'].mean():,.2f}"
    )
    print(
        f"  avg realized missed           : "
        f"{df['realized_total_missed'].mean():.4f}"
    )
    print(
        f"  avg realized ets cost (EUR)   : "
        f"{df['realized_ets_cost_eur'].mean():.2f}"
    )
    print(
        f"  avg runtime (s)               : "
        f"{df['runtime_s'].mean():.4f}"
    )
    print(
        f"  avg skipped ports             : "
        f"{df['n_skipped'].mean():.4f}"
    )
    print(
        f"  final theta mean (last ep)    : "
        f"{df['theta_mean_h'].iloc[-1]:.4f}"
    )
    print(
        f"  final theta max  (last ep)    : "
        f"{df['theta_max_h'].iloc[-1]:.4f}"
    )

    # Note on objective comparability (Fix F5)
    if "objective_comparable" in df.columns:
        comparable = df["objective_comparable"].iloc[0]
        if not comparable:
            print(
                f"  NOTE: objective_value is NOT directly comparable "
                f"across policies"
            )
            print(
                f"        (each policy solves a different tightened instance)"
            )
            print(
                f"        Use realized_service_cost_usd for fair comparison."
            )


def print_theta(theta: dict[int, float], ports: list[str]) -> None:
    nonzero = [(p, v) for p, v in theta.items() if abs(v) > 1e-9]
    if not nonzero:
        print("  <all zero>")
        return
    for p, val in sorted(nonzero, key=lambda t: t[0]):
        print(f"  {ports[p]:<5}: {val:.4f}h")


def print_tail_risk(name: str, summary: dict) -> None:
    if not summary:
        print(f"  {name}: <no data>")
        return
    print(f"\n  {name} tail risk:")
    print(f"    mean  : {summary.get('mean', 0):,.2f}")
    print(f"    std   : {summary.get('std', 0):,.2f}")
    print(f"    p50   : {summary.get('p50', 0):,.2f}")
    print(f"    p90   : {summary.get('p90', 0):,.2f}")
    print(f"    p95   : {summary.get('p95', 0):,.2f}")
    if "avg_realized_ets_cost_eur" in summary:
        print(
            f"    avg ETS (EUR) : "
            f"{summary['avg_realized_ets_cost_eur']:,.2f}"
        )


def main() -> None:
    print("=" * 88)
    print("CFA REFACTORED PIPELINE — SMOKE TEST")
    print("=" * 88)

    # -----------------------------------------------------------------
    # 1. Build base instance (notebook-compatible penalties, Fix F10)
    # -----------------------------------------------------------------
    containers = generate_containers(
        ports=BASE_PORTS,
        distance_matrix_nm=BASE_DISTANCE_MATRIX_NM,
        n=5,
        seed=42,
        notebook_compatible=True,
    )

    base_instance = build_base_instance(
        containers=containers,
        instance_id="cfa_smoke_base",
        initial_delay_h=48.0,
        alpha=0.5,
        allow_swap=True,
        swap_ordering_vars_enabled=True,
        max_skip=1,
        fuel_price_usd_per_tonne=600.0,
        include_fueleu_penalty=False,
        metadata={"seed": 42},
    )

    solver = XpressSolver()
    options = SolveOptions(
        time_limit_s=30,
        mip_gap=0.01,
        log_to_console=False,
    )

    # Uncertainty configuration
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

    # -----------------------------------------------------------------
    # 2. Baseline theta
    # -----------------------------------------------------------------
    baseline_theta = initialize_theta(base_instance)
    print("\nBaseline theta:")
    print_theta(baseline_theta, base_instance.ports)

    # -----------------------------------------------------------------
    # 3. Train additive policy
    # -----------------------------------------------------------------
    print("\nTraining additive policy (10 episodes)...")
    additive_train = train_cfa(
        base_instance,
        solver,
        n_episodes=10,
        solve_options=options,
        seed=42,
        update_policy="additive",
        step_size=1.0,
        uncertainty_config=uncertainty_config,
    )
    additive_theta = additive_train.theta_history[-1]
    print("Trained theta (additive):")
    print_theta(additive_theta, base_instance.ports)

    # -----------------------------------------------------------------
    # 4. Train decay policy
    # -----------------------------------------------------------------
    print("\nTraining decay policy (10 episodes)...")
    decay_train = train_cfa(
        base_instance,
        solver,
        n_episodes=10,
        solve_options=options,
        seed=42,
        update_policy="decay",
        step_size=1.0,       # Fix F13: step_size now correctly used as step_up
        step_down=0.25,
        uncertainty_config=uncertainty_config,
    )
    decay_theta = decay_train.theta_history[-1]
    print("Trained theta (decay):")
    print_theta(decay_theta, base_instance.ports)

    # -----------------------------------------------------------------
    # 5. Train SPSA policy
    # -----------------------------------------------------------------
    print("\nTraining SPSA policy (10 episodes)...")
    spsa_train = train_cfa(
        base_instance,
        solver,
        n_episodes=10,
        solve_options=options,
        seed=42,
        update_policy="spsa",
        step_size=0.5,
        perturbation_size=2.0,
        uncertainty_config=uncertainty_config,
    )
    spsa_theta = spsa_train.theta_history[-1]
    print("Trained theta (SPSA):")
    print_theta(spsa_theta, base_instance.ports)

    # -----------------------------------------------------------------
    # 6. Test all policies
    # -----------------------------------------------------------------
    print("\nTesting all policies (20 episodes each)...")

    baseline_test = test_cfa_policy(
        base_instance,
        solver,
        baseline_theta,
        n_episodes=20,
        solve_options=options,
        seed=123,
        policy_name="baseline_zero_theta",
        uncertainty_config=uncertainty_config,
    )

    additive_test = test_cfa_policy(
        base_instance,
        solver,
        additive_theta,
        n_episodes=20,
        solve_options=options,
        seed=123,
        policy_name="trained_additive",
        uncertainty_config=uncertainty_config,
    )

    decay_test = test_cfa_policy(
        base_instance,
        solver,
        decay_theta,
        n_episodes=20,
        solve_options=options,
        seed=123,
        policy_name="trained_decay",
        uncertainty_config=uncertainty_config,
    )

    spsa_test = test_cfa_policy(
        base_instance,
        solver,
        spsa_theta,
        n_episodes=20,
        solve_options=options,
        seed=123,
        policy_name="trained_spsa",
        uncertainty_config=uncertainty_config,
    )

    # -----------------------------------------------------------------
    # 7. Print summaries
    # -----------------------------------------------------------------
    print_policy_summary("TRAIN — Additive", additive_train.results_df)
    print_policy_summary("TRAIN — Decay", decay_train.results_df)
    print_policy_summary("TRAIN — SPSA", spsa_train.results_df)

    print_policy_summary("TEST — Baseline", baseline_test.results_df)
    print_policy_summary("TEST — Additive", additive_test.results_df)
    print_policy_summary("TEST — Decay", decay_test.results_df)
    print_policy_summary("TEST — SPSA", spsa_test.results_df)

    # -----------------------------------------------------------------
    # 8. Tail risk summary
    # -----------------------------------------------------------------
    print("\nTail Risk Summary (realized_service_cost_usd):")
    for name, result in [
        ("Baseline", baseline_test.results_df),
        ("Additive", additive_test.results_df),
        ("Decay", decay_test.results_df),
        ("SPSA", spsa_test.results_df),
    ]:
        summary = compute_tail_risk_summary(result)
        print_tail_risk(name, summary)

    # -----------------------------------------------------------------
    # 9. Comparison (realized metrics only — Fix F5)
    # -----------------------------------------------------------------
    print("\nPolicy Comparison (realized metrics only):")
    print("-" * 50)
    for name, df in [
        ("Baseline", baseline_test.results_df),
        ("Additive", additive_test.results_df),
        ("Decay", decay_test.results_df),
        ("SPSA", spsa_test.results_df),
    ]:
        missed = df["realized_total_missed"].mean()
        cost = df["realized_service_cost_usd"].mean()
        ets = df["realized_ets_cost_eur"].mean()
        print(
            f"  {name:<12} | missed={missed:.3f}  "
            f"svc_cost=${cost:,.0f}  "
            f"ets=€{ets:.0f}"
        )

    # -----------------------------------------------------------------
    # 10. Plots
    # -----------------------------------------------------------------
    print("\nGenerating plots...")

    plot_theta_evolution(
        additive_train.theta_history,
        ports=base_instance.ports,
        output_dir="results/figures/cfa_smoke",
        filename_stem="theta_evolution_additive",
        title="Theta Evolution — Additive Update",
    )

    plot_theta_evolution(
        decay_train.theta_history,
        ports=base_instance.ports,
        output_dir="results/figures/cfa_smoke",
        filename_stem="theta_evolution_decay",
        title="Theta Evolution — Decay Update",
    )

    plot_theta_evolution(
        spsa_train.theta_history,
        ports=base_instance.ports,
        output_dir="results/figures/cfa_smoke",
        filename_stem="theta_evolution_spsa",
        title="Theta Evolution — SPSA Update",
    )

    plot_cfa_policy_comparison(
        baseline_test.results_df,
        additive_test.results_df,
        decay_test.results_df,
        spsa_df=spsa_test.results_df,
        output_dir="results/figures/cfa_smoke",
        filename_stem="cfa_policy_comparison",
    )

    plot_cfa_summary_bars(
        baseline_test.results_df,
        additive_test.results_df,
        decay_test.results_df,
        spsa_df=spsa_test.results_df,
        output_dir="results/figures/cfa_smoke",
        filename_stem="cfa_summary_bars",
    )

    plot_cfa_tail_risk(
        baseline_test.results_df,
        additive_test.results_df,
        decay_test.results_df,
        spsa_df=spsa_test.results_df,
        output_dir="results/figures/cfa_smoke",
        filename_stem="cfa_tail_risk",
    )

    plot_regulatory_compliance_and_ets(
        {
            "Baseline": baseline_test.results_df,
            "Additive": additive_test.results_df,
            "Decay": decay_test.results_df,
            "SPSA": spsa_test.results_df,
        },
        output_dir="results/figures/cfa_smoke",
        filename_stem="ets_compliance_rate",
    )

    print("\n" + "=" * 88)
    print("CFA SMOKE TEST COMPLETE")
    print("=" * 88)


if __name__ == "__main__":
    main()