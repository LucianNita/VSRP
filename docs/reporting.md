# Reporting Module Documentation

## Purpose of `reporting/`

The `reporting/` package converts experiment outputs into persistent artifacts:
- tables
- figures
- compact console summaries

If `experiments/` produces the analytical results, then `reporting/` is the layer
that turns those results into something reusable for:
- the case-study report
- benchmark comparison
- exploratory analysis
- result archiving

---

## Files in `reporting/`

| File | Main role |
|---|---|
| `export.py` | Shared utilities for saving DataFrames and figures |
| `benchmark_reporting.py` | Benchmark table saving and console summaries |
| `benchmark_plots.py` | Benchmark visualization |
| `sensitivity_plots.py` | Alpha and delay sensitivity plots |
| `emissions_plots.py` | Emissions, ETS, FuelEU, and CII plots |
| `cfa_plots.py` | CFA policy comparison and risk plots |
| `penalty_sensitivity_plots.py` | Penalty sensitivity overview plots |
| `stochastic_ets_plots.py` | Stochastic ETS exposure plots |

---

## Architectural role of `reporting/`

The reporting layer is deliberately separated from:
- optimization logic
- instance generation
- simulation logic

This separation matters because it allows the repository to:
- regenerate figures without modifying optimization code
- keep plotting concerns out of the solver layer
- make experiment outputs portable and easy to reuse

The typical workflow is:

1. `experiments/*` returns a DataFrame or structured result
2. `reporting/*` saves it or visualizes it
3. files are written under `results/`

---

## 1. `reporting/export.py`

### Purpose

`export.py` contains the shared low-level output utilities used by the other reporting modules.

It is the foundation of the reporting layer.

### Main functions

#### `ensure_directory()`
Creates output directories if they do not already exist.

This avoids repeating directory-creation logic throughout plotting and export code.

#### `save_dataframe()`
Saves a pandas DataFrame to:
- CSV
- Excel

This is used by benchmark, sensitivity, CFA, penalty, and stochastic ETS scripts.

#### `save_figure()`
Saves matplotlib figures to:
- PNG
- PDF

This centralization is valuable because it:
- enforces consistent output behavior
- removes duplicated save logic across plotting modules
- makes the reporting layer easier to maintain

---

## 2. `reporting/benchmark_reporting.py`

### Purpose

This file handles benchmark-oriented table export and compact console summaries.

### Main functions

#### `save_benchmark_outputs()`
Saves:
- raw benchmark rows
- solver-level benchmark summary table

#### `print_benchmark_summary()`
Prints a compact benchmark summary to the console.

### Why this file matters

It provides a lightweight textual reporting path in addition to saved tables and plots.

---

## 3. `reporting/benchmark_plots.py`

### Purpose

This module visualizes solver benchmark summaries.

### Main function

#### `plot_benchmark_summary()`
Creates a multi-panel figure showing metrics such as:
- average objective
- average runtime
- average MIP gap

### Why this file matters

Benchmark tables are useful, but visual comparison makes performance differences much easier to communicate.

---

## 4. `reporting/sensitivity_plots.py`

### Purpose

This module visualizes deterministic sensitivity experiments.

### Main functions

#### `plot_alpha_sweep()`
Plots the effect of the service-weight parameter `alpha` on:
- objective
- cost decomposition
- service outcomes
- emissions

#### `plot_delay_sweep()`
Plots the effect of initial disruption delay on:
- objective
- cost decomposition
- service outcomes
- emissions / ETS cost

#### `plot_alpha_pareto()`
Creates a Pareto-style operational-cost vs service-cost trade-off plot.

### Why this file matters

It helps translate raw sensitivity tables into interpretable economic trade-off figures.

---

## 5. `reporting/emissions_plots.py`

### Purpose

This module focuses specifically on environmental and regulatory visualizations.

### Main functions

#### `plot_green_recovery_tradeoff()`
Shows how cost, service, and emissions move across alpha values.

#### `plot_emissions_vs_delay_penalty()`
Shows the relationship between CO2 burden and delay-service penalties across disruption scenarios.

#### `plot_strategy_mix_under_ets()`
Shows route strategy composition together with emissions / ETS exposure under varying ETS carbon prices.

Important interpretation:
this is currently best understood as an **ETS exposure** visualization,
not a fully endogenous carbon-price-in-objective policy-response plot.

#### `plot_ets_phase_in_impact()`
Illustrates how ETS burden changes across regulatory phase-in years.

#### `plot_cii_profile()`
Plots attained versus required CII across disruption scenarios, colored by rating.

### Why this file matters

This file is the main vehicle for communicating the environmental contribution of the project.

---

## 6. `reporting/cfa_plots.py`

### Purpose

This module visualizes CFA policy learning and evaluation results.

### Main functions

#### `plot_theta_evolution()`
Shows how the learned tightening vector evolves over training episodes.

#### `plot_cfa_policy_comparison()`
Compares policies across realized metrics such as:
- cumulative misses
- realized service cost
- realized ETS cost

#### `plot_cfa_summary_bars()`
Provides average realized-policy comparison in bar-chart form.

#### `plot_cfa_tail_risk()`
Shows the distribution of realized service cost with p90/p95 emphasis.

#### `plot_ets_compliance_rate()`
Visualizes regulatory compliance-style metrics and realized ETS exposure.

### Important design note

CFA plots are intentionally based on **realized metrics**, not raw optimization objectives,
because different policies solve differently tightened optimization instances.

### Why this file matters

It is what makes the CFA results interpretable and report-ready.

---

## 7. `reporting/penalty_sensitivity_plots.py`

### Purpose

This module visualizes tactical penalty-sensitivity experiments.

### Main function

#### `plot_penalty_sweep_overview()`
Creates a compact multi-panel overview showing:
- average objective
- average service/strategy counts
- average emissions
- fraction of seeds with route or strategy change

### Why this file matters

Penalty experiments often need both:
- cost interpretation
- structural-change interpretation

This module supports both in one figure.

---

## 8. `reporting/stochastic_ets_plots.py`

### Purpose

This module visualizes stochastic ETS exposure distributions.

### Main function

Typical plots here show:
- realized carbon price vs ETS cost
- ETS cost distribution / histogram

### Why this file matters

Tables of stochastic ETS scenarios are useful, but distributional plots make
tail exposure much easier to understand.

---

## Common reporting pattern

Most reporting modules follow the same simple pattern:

1. receive a DataFrame or structured result object
2. perform light filtering / aggregation
3. create a matplotlib figure or export table
4. save output into `results/`

This is intentionally lightweight. The reporting layer is not meant to contain
business logic that belongs in `core/` or `experiments/`.

---

## Relationship to the rest of the codebase

### `reporting/` depends on `experiments/`
because experiments produce the tabular outputs that are visualized or exported.

### `reporting/` depends on `export.py`
for consistent save behavior across all table and figure outputs.

### `scripts/` depends on `reporting/`
because smoke scripts call reporting functions after experiments finish.

---

## Strengths of the `reporting/` layer

- clean separation from solver logic
- reusable table and figure outputs
- centralized save helpers
- consistent output directory structure
- supports both exploratory and report-oriented workflows

---

## Current limitations of the `reporting/` layer

- most plots assume experiment outputs have already been cleaned and are consistent
- figure styles are informative but still relatively lightweight compared with publication-grade customization
- some modules depend on fields only available from the full Xpress canonical solution path

---

## Summary of `reporting/`

The `reporting/` package is the presentation and export layer of the repository.

It turns experiment outputs into:
- saved tables
- benchmark figures
- sensitivity figures
- emissions and regulatory figures
- CFA policy figures
- penalty and stochastic ETS plots

This layer is essential for reproducibility because it ensures that all major results
can be regenerated directly from code.
