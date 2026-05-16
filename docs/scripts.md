# Scripts Module Documentation

## Purpose of `scripts/`

The `scripts/` package contains the **user-facing executable entry points** of the repository.

These files are not core logic modules. Instead, they are thin orchestration layers that:
- build instances
- choose solver backends
- call experiment functions
- print summaries to the console
- save tables and figures into `results/`

In practical terms, this is the part of the repository that a user runs directly.

---

## Why scripts matter

Although the real logic lives in `core/`, `model/`, `experiments/`, and `reporting/`,
the scripts are important because they provide:
- reproducible end-to-end workflows
- smoke tests for debugging and validation
- convenient experiment runners
- examples of how the architecture is meant to be used

They also show the intended usage pattern of the codebase.

---

## Files in `scripts/`

| File | Main role |
|---|---|
| `smoke_test.py` | End-to-end single-instance Xpress sanity check |
| `transshipment_smoke.py` | Focused smoke test for transshipment logic across solvers |
| `benchmark_smoke.py` | Small multi-solver benchmark run |
| `sensitivity_smoke.py` | Alpha / delay / ETS sensitivity run |
| `cfa_smoke.py` | CFA training and evaluation smoke test |
| `penalty_sensitivity_smoke.py` | Multi-seed swap penalty experiment |
| `penalty_speedup_multiseed_smoke.py` | Multi-seed speed-up penalty experiment |
| `penalty_plot_smoke.py` | Plot-only runner for penalty sensitivity outputs |
| `stochastic_ets_smoke.py` | Stochastic ETS exposure experiment |
| `debug_alpha_case.py` | Detailed debugging view for one alpha scenario |

---

## General script pattern

Most scripts follow the same structure:

1. create or load a container set
2. build one or more canonical instances
3. choose a solver and solve options
4. run one experiment or one direct solve
5. print a readable summary
6. save tables and/or figures

This makes the script layer easy to understand and easy to extend.

---

## 1. `scripts/smoke_test.py`

### Purpose

This is the most important basic sanity-check script in the repository.

It runs one small Xpress solve and prints:
- generated containers
- instance settings
- route
- strategy decisions
- skipped and swapped ports
- container outcomes
- validation summary
- emissions summary
- cost breakdown

### Why this script matters

It is the fastest way to confirm that the full native workflow still works after code changes.

If this script passes, then many core pieces are working together correctly:
- instance generation
- solver build/solve
- route extraction
- validation
- emissions
- cost recomputation

---

## 2. `scripts/transshipment_smoke.py`

### Purpose

This script stress-tests the transshipment and misconnection logic.

It generates a mixed direct/transshipment container set and solves the same instance with:
- Xpress
- HiGHS
- CBC

### Why this script matters

It specifically targets one of the major model extensions: transshipment misconnection.

It also serves as a cross-solver objective-consistency check.

### Important detail

Only Xpress currently returns full route/timeline/container-output detail.
HiGHS and CBC appear as partial-solution backends in this script.

---

## 3. `scripts/benchmark_smoke.py`

### Purpose

This script runs a small benchmark over multiple solvers on a few random instances.

It prints:
- solver availability
- selected raw benchmark columns
- benchmark summary table
- stability metrics

It also saves:
- raw benchmark results
- summary benchmark table
- benchmark summary figure

### Why this script matters

It is the simplest executable proof that the benchmarking layer works end-to-end.

---

## 4. `scripts/sensitivity_smoke.py`

### Purpose

This script runs the main deterministic sensitivity experiments:
- alpha sweep
- delay sweep
- ETS exposure sweep

It saves both tables and plots.

### Outputs typically include
- objective trajectories
- cost decomposition trends
- service KPIs
- emissions and ETS metrics
- green recovery plots
- CII profile plots

### Why this script matters

It is the quickest end-to-end test of the deterministic analytical layer of the repository.

---

## 5. `scripts/cfa_smoke.py`

### Purpose

This script runs the full CFA workflow on a small problem:
- initialize baseline theta
- train additive policy
- train decay policy
- train SPSA policy
- test all policies on repeated stochastic episodes
- compute tail-risk summaries
- generate CFA plots

### Why this script matters

This is the main executable test of the stochastic learning extension of the project.

It confirms that optimization, simulation, and policy updating are connected correctly.

### Important interpretation

The script emphasizes realized metrics rather than raw optimization objective values,
because different CFA policies solve differently tightened optimization instances.

---

## 6. `scripts/penalty_sensitivity_smoke.py`

### Purpose

This script runs a multi-seed penalty sweep focused on swap penalty sensitivity.

It prints:
- seed
- penalty value
- objective
- service/strategy counts
- route and strategy signatures

It also builds a summary of route/strategy change frequency by seed.

### Why this script matters

It is useful for identifying whether tactical recovery behavior is:
- flat over a penalty range
- threshold-sensitive
- instance-dependent

---

## 7. `scripts/penalty_speedup_multiseed_smoke.py`

### Purpose

This script runs a multi-seed penalty sweep focused on speed-up penalty sensitivity.

It helps determine whether the speed-up tactic is active, inactive, or threshold-sensitive
under the current generated instances.

### Why this script matters

It complements the swap penalty script and helps assess whether the model is using
speed-based recovery aggressively enough to show meaningful environmental trade-offs.

---

## 8. `scripts/penalty_plot_smoke.py`

### Purpose

This is a plot-only helper script.

It reads already-saved CSV outputs from penalty experiments and generates:
- swap penalty overview plots
- speed-up penalty overview plots

### Why this script matters

It separates experiment execution from plotting, which can be useful when:
- rerunning plots only
- modifying figure aesthetics
- testing plot modules without rerunning optimization

---

## 9. `scripts/stochastic_ets_smoke.py`

### Purpose

This script runs the stochastic ETS exposure experiment.

It:
- solves one deterministic instance
- samples many carbon-price scenarios
- recomputes ETS exposure for the fixed route
- prints scenario rows and distribution summaries
- saves the resulting table

### Why this script matters

It is the simplest executable demonstration of the repository’s stochastic regulatory-risk analysis.

---

## 10. `scripts/debug_alpha_case.py`

### Purpose

This is a diagnostic script for one selected alpha scenario.

It prints detailed information about:
- containers
- solved route
- skipped and swapped ports
- timeline
- validation
- cost breakdown
- emissions

### Why this script matters

When a sensitivity result looks surprising, this script provides a detailed view into one case.
It is especially useful for debugging route reconstruction or timeline behavior.

---

## Relationship to the rest of the codebase

### `scripts/` depends on `experiments/`
for higher-level orchestration functions such as benchmarking, sensitivity, CFA,
penalty sweeps, and stochastic ETS.

### `scripts/` depends on `model/`
for direct solver selection in smoke and debugging workflows.

### `scripts/` depends on `reporting/`
for saving tables and figures.

### `scripts/` may depend directly on `data/`
for direct instance construction in lower-level smoke or diagnostic scripts.

---

## Why scripts are kept thin

A major design choice in the repository is to keep scripts as thin as possible.

That means:
- business logic belongs in `core/`
- optimization logic belongs in `model/`
- repeated-run logic belongs in `experiments/`
- plotting/export logic belongs in `reporting/`
- scripts mainly wire those layers together

This is good practice because it makes the scripts:
- easy to read
- easy to debug
- easy to replace or extend

---

## Strengths of the `scripts/` layer

- clear runnable entry points
- reproducible smoke tests
- good examples of intended code usage
- direct generation of report-ready outputs in `results/`

---

## Current limitations of the `scripts/` layer

- some scripts are purpose-specific and not yet generalized into a command-line interface
- several scripts rely on the full Xpress backend for rich route/KPI extraction
- script naming is mostly consistent but could be standardized further in future cleanup

---

## Summary of `scripts/`

The `scripts/` package is the executable control layer of the repository.

It provides:
- smoke tests
- benchmark runners
- sensitivity runners
- CFA runners
- penalty experiment runners
- stochastic ETS runners
- debugging tools

This layer is what makes the rest of the architecture accessible in practice.

