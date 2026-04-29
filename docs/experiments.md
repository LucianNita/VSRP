# Experiments Module Documentation

## Purpose of `experiments/`

The `experiments/` package orchestrates repeated solver runs and converts them
into structured experimental outputs.

If `model/` is responsible for solving **one** optimization instance, then
`experiments/` is responsible for answering broader analytical questions such as:
- How do solvers compare?
- How does behavior change as parameters vary?
- How does a learned CFA policy perform under uncertainty?
- How sensitive is strategy choice to tactical penalties?
- What is the distribution of ETS exposure under stochastic carbon prices?

This package is where the repository becomes an experimental platform rather than
just a solver implementation.

---

## Files in `experiments/`

| File | Main role |
|---|---|
| `benchmark.py` | Cross-solver benchmarking and summary aggregation |
| `sensitivity.py` | Alpha, delay, and ETS sensitivity sweeps |
| `cfa.py` | CFA training, testing, and tail-risk evaluation |
| `penalty_sensitivity.py` | Tactical penalty sweeps and multi-seed stability checks |
| `stochastic_ets.py` | Post-solve stochastic ETS exposure evaluation |

---

## Architectural role of `experiments/`

The experiment layer sits between:
- the optimization layer (`model/`)
- the reporting layer (`reporting/`)

Its main jobs are to:
- generate or reuse test instances
- select solver backends
- run one or more solves
- convert solutions into tidy rows or summary tables
- expose experiment outputs in forms that can be plotted or exported

In other words, `experiments/` is the **analysis orchestration layer** of the project.

---

## 1. `experiments/benchmark.py`

### Purpose

`benchmark.py` compares solver backends on common randomly generated instances.

It is designed to answer questions like:
- Do different solvers find the same objective?
- How fast is each solver?
- How stable are the results across seeds/instances?
- Does the Xpress reference backend remain stronger than open-source backends?

### Main functions

#### `solution_to_benchmark_record()`
Converts a canonical `VSRPSolution` into a flat `BenchmarkRecord`.

This flattening is important because benchmark outputs are tabular by nature.

#### `run_benchmark()`
Main benchmark driver.

It:
- generates random container sets for multiple instances
- builds canonical base instances
- runs all requested solvers
- collects raw benchmark rows
- builds a summary DataFrame

#### `summarize_benchmark()`
Aggregates raw benchmark rows into solver-level metrics such as:
- average objective
- average runtime
- average MIP gap
- stability (objective standard deviation, runtime variation)
- feasibility and optimality rates

### Why this file matters

This file is how the repository addresses the benchmark requirement of the case study.
It transforms one-off solver runs into comparative evidence.

---

## 2. `experiments/sensitivity.py`

### Purpose

`sensitivity.py` studies how model outputs change when one parameter is varied while others are held fixed.

This file is focused on deterministic comparative statics.

### Main sweeps

#### `run_alpha_sweep()`
Varies the objective trade-off weight `alpha`.

This shows how the model transitions between:
- operationally cheaper solutions
- service-protecting solutions
- their associated emissions and regulatory exposure

#### `run_delay_sweep()`
Varies the initial disruption delay.

This helps answer:
- how much recovery action is needed under larger disruptions
- when strategy patterns stabilize
- how emissions respond to higher schedule pressure

#### `run_ets_price_sweep()`
Varies ETS carbon price in the reporting layer.

Important interpretation:
this is currently an **ETS exposure sweep**, not a fully endogenous
carbon-price-in-objective optimization experiment.

The route is solved under the current formulation, then ETS cost is recomputed
at alternative carbon prices.

### Main helper

#### `_solution_to_sensitivity_row()`
Converts one canonical solution into a flat row with:
- objective and runtime
- cost breakdown
- service KPIs
- emissions and regulatory metrics
- selected validation fields

### Why this file matters

This file provides the empirical basis for:
- alpha trade-off plots
- delay sensitivity figures
- green recovery analysis
- ETS exposure analysis

---

## 3. `experiments/cfa.py`

### Purpose

`cfa.py` implements the Cost Function Approximation (CFA) framework used to train
a policy under uncertainty.

This is one of the most advanced parts of the repository because it combines:
- optimization
- simulation
- parameter learning

### Key idea

The optimization model is not changed structurally.
Instead, a parameter vector `theta` modifies the effective service promises used
in optimization.

After solving, the resulting route is evaluated under a realized stochastic scenario.
Then `theta` is updated based on observed performance.

This creates a repeated learning loop.

### Main concepts

#### `initialize_theta()`
Creates an all-zero port-indexed tightening vector.

#### `apply_theta_to_instance()`
Returns a modified optimization instance in which promised arrivals are tightened
by destination-specific `theta` values.

This is the key mechanism by which the CFA policy influences the optimization model.

#### Update rules

##### `update_theta_additive()`
Simple rule-based increase where misses occur.

##### `update_theta_decay()`
Two-sided rule:
- increase where misses occur
- decay where no misses occur

##### `update_theta_spsa()`
Stochastic approximation update using simultaneous perturbation.

This is the most sophisticated policy-update rule in the repository.

#### `train_cfa()`
Runs repeated optimization + simulation + update episodes.

#### `test_cfa_policy()`
Evaluates a fixed theta policy on repeated stochastic episodes without updating it.

#### `compute_tail_risk_summary()`
Computes mean / standard deviation / percentile summaries for realized cost.

### Important conceptual point

Optimization objective values are **not directly comparable across policies**,
because each policy solves a differently tightened optimization instance.

The meaningful comparison metric is:
- realized service cost
- realized misses
- realized ETS cost
- tail risk

### Why this file matters

This file is how the repository addresses the CFA / sequential decision-making
part of the case-study brief.

---

## 4. `experiments/penalty_sensitivity.py`

### Purpose

`penalty_sensitivity.py` studies how route and strategy patterns respond to
changes in tactical penalty parameters.

This is slightly different from `sensitivity.py`:
- `sensitivity.py` focuses on macro scenario parameters such as alpha and delay
- `penalty_sensitivity.py` focuses on tactical recovery incentives

### Main function

#### `run_penalty_sweep()`
Runs a one-parameter sweep over one selected penalty field, such as:
- `swap_usd`
- `omission_usd`
- `speed_up_usd`
- `expedited_port_usd`

The container set is held fixed within the sweep so that the effect of the
penalty parameter can be isolated.

#### `_solution_to_penalty_row()`
Converts one solved instance into a rich flat row including:
- cost decomposition
- service KPIs
- emissions metrics
- route signature
- strategy signature

These signatures make it possible to detect structural changes in the solution.

#### `run_penalty_sweep_multi_seed()`
Repeats the same penalty sweep across multiple random seeds.

This is important because one seed alone may produce a flat sensitivity region.

#### `summarize_penalty_sweep_changes()`
Aggregates route/strategy-change behavior across seeds.

### Why this file matters

This file makes the penalty experiments much more informative by distinguishing:
- objective changes caused only by cost coefficients
- genuine structural route or strategy transitions

It is especially useful for showing when swap or omission becomes attractive or unattractive.

---

## 5. `experiments/stochastic_ets.py`

### Purpose

`stochastic_ets.py` evaluates the distribution of ETS exposure under uncertain carbon prices.

This experiment is intentionally simpler than the CFA framework:
- solve one deterministic instance once
- hold the route fixed
- sample many carbon-price scenarios
- recompute realized ETS cost

### Main function

#### `evaluate_stochastic_ets_exposure()`
Runs the full experiment and returns one row per sampled scenario.

Each row includes:
- scenario index
- realized carbon price
- realized ETS cost
- fixed route signature
- selected service statistics from the solved route

#### `summarize_stochastic_ets()`
Computes a distribution summary including:
- mean
- standard deviation
- min / max
- p50 / p90 / p95 / p99
- simple tail mean above p95

### Important interpretation

This is a **post-solve exposure experiment**, not a full stochastic optimization model.

That means it measures:
- how costly ETS may become under uncertain carbon prices for a fixed route

It does **not** re-optimize the recovery plan per carbon-price scenario.

### Why this file matters

It provides a lightweight but meaningful stochastic regulatory-risk analysis,
which complements both the deterministic ETS sweep and the CFA experiments.

---

## Common pattern across experiment files

Although the experiment files study different questions, they share a common workflow:

1. generate or reuse a container set
2. build one or more canonical instances
3. solve using a backend implementing `BaseSolver`
4. convert solutions into tidy flat rows
5. return pandas DataFrames for reporting and plotting

This is an important architectural strength because it makes experiment outputs:
- consistent
- reusable
- easy to export
- easy to plot

---

## Relationship to the rest of the codebase

### `experiments/` depends on `data/`
for generating container sets and building base instances.

### `experiments/` depends on `model/`
for solving optimization instances.

### `experiments/` depends on `core/`
for cost recomputation, emissions, simulation, and validation-aware KPIs.

### `reporting/` depends on `experiments/`
because experiments produce the DataFrames that reporting modules consume.

---

## Strengths of the `experiments/` layer

- consistent experiment orchestration
- reusable solver interface
- tidy tabular outputs
- support for deterministic and stochastic studies
- direct connection to reporting layer

---

## Current limitations of the `experiments/` layer

- many experiments rely on the Xpress backend for full route/KPI extraction
- ETS price sweep is exposure-based rather than fully endogenous
- stochastic ETS currently evaluates fixed routes rather than re-optimizing per scenario
- some sensitivity sweeps may appear flat for particular random seeds or demand sets

---

## Summary of `experiments/`

The `experiments/` package is the analytical orchestration layer of the repository.

It transforms one-off optimization solves into evidence about:
- solver performance
- sensitivity to disruption and policy parameters
- learned policy performance under uncertainty
- tactical penalty thresholds
- regulatory risk exposure

This layer is what turns the repository into a full case-study platform.
