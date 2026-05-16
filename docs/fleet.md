# Fleet Module Documentation

## Purpose

This document explains the **fleet-level extension** of the VSRP codebase.

The fleet layer was added after the single-vessel architecture and is therefore more lightly documented in the current repository. Its goal is to support:

- multi-vessel scenario construction,
- fleet-level benchmarking,
- fleet-level CFA training and testing,
- aggregated reporting across multiple vessels.

The fleet extension is intentionally designed to **reuse the canonical single-vessel VSRP model** rather than introducing an entirely separate modeling stack.

---

## Core idea

The fleet implementation in this repository is based on **decomposition**.

Instead of building one giant joint optimization model containing all vessels at once, the code:

1. builds one canonical `VSRPInstance` per vessel,
2. solves each vessel independently using any single-vessel solver backend,
3. aggregates the resulting `VSRPSolution` objects into a `FleetSolution`.

In mathematical terms, this is valid when the fleet problem is **separable across vessels**, i.e. when there are no cross-vessel coupling constraints.

The current fleet implementation assumes:

$$
\min \sum_{v \in V} f_v(x_v)
$$

subject to independent vessel-feasible sets:

$$
x_v \in X_v \qquad \forall v \in V
$$

with **no shared constraints** of the form:

$$
g(x_1, x_2, \dots, x_{|V|}) \le 0
$$

---

## Important modeling interpretation

The current fleet layer is best described as:

- **multi-vessel decomposed scenario analysis**, or
- **fleet-level aggregation of independent vessel recovery problems**

It is **not** a coupled fleet optimization model with:

- shared berth capacity,
- inter-vessel synchronization,
- cross-vessel transshipment recapture,
- shared port resource limits,
- fleet-wide emissions caps,
- joint FuelEU pooling,
- fleet-wide ETS budget constraints.

This distinction is important for accurate interpretation of the experiments.

---

## Relationship to the original enhanced notebook

The original enhanced notebook implementation indexed decisions by vessel and solved them in one larger Xpress model. However, in the code originally provided, the vessels did not appear to be coupled by shared constraints.

Under that assumption, the problem is separable and can be solved exactly by decomposition.

Therefore, the fleet implementation in this repository is:

- **similar in essence** to the enhanced notebook formulation,
- **different in implementation style**,
- and often **cleaner architecturally**, because it explicitly reuses the single-vessel model.

---

## Files involved in the fleet extension

| File | Main role |
|---|---|
| `core/entities.py` | Defines `VesselConfig`, `FleetInstance`, `FleetSolution`, `FleetBenchmarkRecord` |
| `core/fleet_costs.py` | Aggregates per-vessel cost breakdowns into fleet totals |
| `core/simulation.py` | Defines fleet uncertainty and fleet realized-performance simulation |
| `data/fleet_instance.py` | Builds fleet instances from per-vessel configurations or delays |
| `model/fleet_solver.py` | Solves each vessel independently and aggregates results |
| `experiments/fleet_benchmark.py` | Runs canonical fleet scenario benchmarking across solvers |
| `experiments/fleet_cfa.py` | Runs fleet-level CFA training and testing with shared theta |
| `reporting/fleet_plots.py` | Produces fleet benchmark and per-vessel figures |
| `scripts/fleet_smoke.py` | End-to-end Xpress smoke test for the fleet layer |
| `scripts/fleet_benchmark_smoke.py` | Multi-solver benchmark smoke for canonical fleet scenarios |
| `scripts/fleet_cfa_smoke.py` | Fleet-level CFA smoke test |

---

## 1. Fleet entities

The fleet layer is built on top of the same canonical design used in the single-vessel architecture.

### `VesselConfig`
Defined in `core/entities.py`.

This is a declarative configuration object for one vessel in a fleet scenario. It contains:

- vessel identifier,
- assigned container list,
- initial delay,
- alpha,
- fuel price,
- optional vessel-specific port penalties,
- optional vessel-specific speed levels,
- swap and skip settings,
- optional FuelEU activation flag.

It is not itself an optimization instance. Instead, it is a lightweight input specification used by fleet instance factory functions.

### `FleetInstance`
Also defined in `core/entities.py`.

This object wraps a list of per-vessel `VSRPInstance` objects. It provides:

- `fleet_id`
- `vessel_instances`
- fleet-level metadata
- convenience properties such as:
  - `n_vessels`
  - `n_ports`
  - `ports`
  - `total_containers`

Conceptually, `FleetInstance` is the fleet-level analogue of `VSRPInstance`.

### `FleetSolution`
Also defined in `core/entities.py`.

This object aggregates one `VSRPSolution` per vessel and exposes fleet-level summaries such as:

- `fleet_objective_value`
- `feasible`
- `optimal`
- `total_delayed`
- `total_misconnected`
- `total_skipped`
- `total_swapped`
- `avg_runtime_s`
- `total_runtime_s`

This is the fleet-level analogue of `VSRPSolution`.

### `FleetBenchmarkRecord`
A flat reporting entity used to simplify benchmark summary tables.

It contains:
- solver name,
- fleet scenario ID,
- number of vessels,
- objective,
- runtime,
- service metrics,
- emissions metrics,
- route/strategy validation flags where available.

---

## 2. Fleet instance construction

Fleet instance construction lives in:

- `data/fleet_instance.py`

### `build_fleet_instance()`
This function takes a list of `VesselConfig` objects and converts each one into a per-vessel canonical `VSRPInstance` using the fixed route template from `data/base_instance.py`.

The resulting list is wrapped into a `FleetInstance`.

### `build_fleet_from_delays()`
This is a convenience wrapper for the common case where the user has:

- a list of vessel delays,
- and a list of container lists, one per vessel.

It builds one `VesselConfig` per vessel and then delegates to `build_fleet_instance()`.

### Design choice
The fleet layer does **not** define a separate route model. All vessels share the same port network and distance matrix, but maintain vessel-specific:

- delay,
- containers,
- penalties,
- regulatory flags,
- metadata.

This preserves consistency with the single-vessel model.

---

## 3. Fleet solver

Fleet solving is implemented in:

- `model/fleet_solver.py`

### `FleetSolver`
This class takes any `BaseSolver` implementation as a vessel-level backend and applies it independently to each vessel instance.

Example:

```python
vessel_solver = XpressSolver()
fleet_solver = FleetSolver(vessel_solver=vessel_solver)
fleet_solution = fleet_solver.solve(fleet_instance, options=options)
```

### How it works
For each `VSRPInstance` in `fleet.vessel_instances`:

1. call the single-vessel solver,
2. obtain a canonical `VSRPSolution`,
3. collect the vessel solutions,
4. sum objective values when available.

This yields a `FleetSolution`.

### Why this is valid
This is valid because the fleet model currently assumes no coupling constraints between vessels. Therefore:

$$
\min_{x_1,\dots,x_m} \sum_{v=1}^{m} f_v(x_v)
$$

decomposes into:

$$
\min_{x_v} f_v(x_v) \qquad \forall v
$$

independently.

### Practical advantages
This design provides several software-engineering benefits:

- reuse of the single-vessel model,
- easy debugging,
- natural compatibility with all solver backends,
- no duplicate modeling stack,
- lower practical exposure to Community-license size limits.

---

## 4. Fleet cost aggregation

Fleet cost aggregation is implemented in:

- `core/fleet_costs.py`

### `FleetCostBreakdown`
This dataclass stores:

- list of per-vessel `CostBreakdown`s,
- total fuel cost,
- total port-call cost,
- total strategy penalties,
- total port penalties,
- total FuelEU penalties,
- total delay cost,
- total misconnection cost,
- total operational cost,
- total service cost,
- total fleet objective.

### `compute_fleet_cost_breakdown()`
This function applies `compute_cost_breakdown()` to each vessel and sums the results.

### `fleet_objective_gap_to_reported()`
Compares:
- recomputed fleet objective,
- reported fleet objective value from `FleetSolution`.

This mirrors the single-vessel post-solve cost-consistency check.

---

## 5. Fleet uncertainty and simulation

Fleet uncertainty lives in:

- `core/simulation.py`

### `FleetUncertaintyScenario`
This object contains:

- one `UncertaintyScenario` per vessel,
- a shared realized ETS carbon price,
- a shared realized fuel price.

The design assumes:
- vessel delays, weather factors, and port-handling multipliers may differ by vessel,
- carbon and fuel prices are market-level variables shared across the fleet episode.

### `sample_fleet_uncertainty_scenario()`
This function:
- samples one shared carbon price,
- one shared fuel price,
- and one per-vessel uncertainty scenario.

### `simulate_fleet_realized_performance()`
This function applies the existing single-vessel realized-performance simulator independently to each vessel solution and returns a list of `RealizedPerformance` objects.

### Interpretation
As with optimization, the simulation layer is additive across vessels in the current design.

---

## 6. Fleet benchmarking

Fleet benchmarking lives in:

- `experiments/fleet_benchmark.py`

### Canonical scenarios
The module defines three standard fleet scenarios:

- `Case1_Delayed`
- `Case2_PortClosure`
- `Case3_Congestion`

These mirror the scenario style introduced in the earlier enhanced notebook but are implemented through decomposed per-vessel canonical instances.

#### `Case1_Delayed`
Multiple delayed vessels with no port penalties.

#### `Case2_PortClosure`
Long Beach closure represented by a large visit penalty.

#### `Case3_Congestion`
Dutch Harbor congestion represented by a large visit penalty.

### `build_canonical_fleet()`
Builds one of these predefined scenarios.

### `fleet_solution_to_benchmark_record()`
Converts a solved `FleetSolution` into a flat `FleetBenchmarkRecord`.

### `run_fleet_benchmark()`
Runs all requested fleet scenarios across all requested solver backends and returns:

- raw fleet benchmark table,
- solver-level fleet summary table.

### `summarize_fleet_benchmark()`
Aggregates solver-level fleet statistics such as:

- feasible rate,
- optimal rate,
- average fleet objective,
- average runtime,
- average delayed/misconnected totals,
- average emissions,
- validation rates.

### Important note about partial backends
As with the single-vessel benchmark:
- Xpress returns full canonical vessel solutions,
- HiGHS and CBC currently return objective-level fleet information only, because their vessel solutions are partial canonical solutions.

Therefore some columns are only populated for Xpress.

---

## 7. Fleet CFA

Fleet-level Cost Function Approximation lives in:

- `experiments/fleet_cfa.py`

This extends the single-vessel CFA idea to a shared-theta fleet setting.

### Shared theta concept
A single theta vector:

$$
\theta : \text{port index} \to \text{tightening in hours}
$$

is shared across all vessels.

This means each destination port has one common tightening parameter applied to all vessels serving that route structure.

### Why shared theta matters
This lets vessels learn from each other.

If several vessels systematically miss at the same port, the aggregated signal increases the corresponding theta entry, encouraging more conservative optimization around that destination across the fleet.

### `initialize_fleet_theta()`
Creates an all-zero shared port-indexed theta vector.

### `apply_fleet_theta()`
Applies the same theta tightening to every vessel in the fleet by reusing the single-vessel `apply_theta_to_instance()` helper.

### `aggregate_fleet_missed_by_port()`
Sums missed-container counts by port across all vessel realized-performance records.

This produces the fleet-wide update signal.

### `train_fleet_cfa()`
For each episode:
1. sample estimated delays per vessel,
2. sample one fleet uncertainty scenario,
3. apply shared theta to all vessel optimization instances,
4. solve all vessels with `FleetSolver`,
5. simulate realized performance on the original base fleet,
6. aggregate missed-by-port signal,
7. update theta using additive, decay, or SPSA.

### `test_fleet_cfa_policy()`
Evaluates a fixed fleet theta without updating it.

### `compute_fleet_tail_risk_summary()`
Computes:
- mean,
- standard deviation,
- percentiles,
- tail mean above p95,
- FuelEU compliance rate when available,
- average realized ETS cost.

### Interpretation
As in single-vessel CFA:
- optimization objective values are **not directly comparable** across policies,
- because different theta vectors alter the optimization problem itself.

The correct comparison basis is:
- realized service cost,
- realized misses,
- realized ETS exposure,
- tail risk.

---

## 8. Fleet reporting

Fleet plotting lives in:

- `reporting/fleet_plots.py`

### `plot_fleet_scenario_comparison()`
Produces a multi-panel comparison across fleet scenarios and solver backends, including:

- fleet objective,
- service impact,
- CO2 emissions,
- runtime.

### `plot_fleet_per_vessel_breakdown()`
Shows per-vessel:
- objective,
- delayed/misconnected counts,
- emissions.

This is useful because fleet totals can hide vessel-level heterogeneity.

### `plot_fleet_benchmark_summary()`
Summarizes solver-level fleet benchmark results, including:
- average fleet objective,
- average total runtime,
- feasibility and optimality rates.

---

## 9. Fleet scripts

The main executable fleet entry points are:

### `scripts/fleet_smoke.py`
End-to-end Xpress-only smoke test for the canonical fleet scenarios.

It prints:
- per-vessel results,
- fleet summary,
- cost breakdowns,
- and generates basic fleet plots.

### `scripts/fleet_benchmark_smoke.py`
Multi-solver benchmark smoke test for the canonical fleet scenarios.

This is the main fleet-level benchmark demonstration script.

### `scripts/fleet_cfa_smoke.py`
Fleet-level CFA smoke test.

It:
- trains additive, decay, and SPSA fleet policies,
- tests fixed policies,
- exports episode-level and per-vessel tables,
- plots theta evolution.

### `scripts/fleet_license_smoke.py` or `fleet_scale_smoke.py`
If present, this type of script is intended to test practical scalability of the decomposed fleet workflow under the Xpress Community license.

Important:
- this tests the decomposed workflow,
- not a monolithic multi-vessel Xpress model.

---

## 10. Community-license interpretation for fleet workflows

A useful practical insight from the repository is:

- **decomposed fleet workflows** can scale comfortably under the Xpress Community license,
- because each vessel is solved as a separate model,
- whereas a sufficiently large **single blown-up model** can trigger explicit Community-license row/column limits.

This is an architectural benefit of the decomposed fleet design.

---

## 11. Strengths of the fleet extension

The fleet layer has several strong properties:

- reuses the canonical single-vessel architecture cleanly,
- solver-agnostic by construction,
- easy to benchmark across solvers,
- easy to debug vessel-by-vessel,
- compatible with post-solve validation, emissions, and cost recomputation,
- naturally supports shared-theta CFA policies.

---

## 12. Current limitations of the fleet extension

The fleet layer is intentionally simplified and has several important limitations.

### No coupled fleet constraints
There are no shared constraints such as:
- berth capacity,
- congestion windows,
- fleet-wide budgets,
- coordinated transshipment recovery.

### No monolithic fleet MIP
The fleet layer is decomposed rather than jointly optimized in one model.

### Open-source partial-solution limitation
HiGHS and CBC currently inherit the single-vessel partial-solution limitation, so route-level fleet KPIs are only fully available through Xpress.

### Shared route template
All vessels use the same route structure; heterogeneous services are not yet modeled.

### Shared-theta CFA simplicity
Fleet CFA currently uses a common destination-indexed tightening vector, which is interpretable and useful, but still relatively simple compared to richer adaptive policies.

---

## 13. Suggested wording when describing this layer

Good terminology:
- **fleet-level decomposed optimization**
- **multi-vessel scenario analysis**
- **fleet aggregation of independent vessel recovery problems**
- **shared-theta fleet CFA**

Avoid overstating it as:
- “full coordinated fleet optimization”
- “network-wide shipping optimization”
- “joint fleet control under shared resource constraints”

unless such coupling is actually implemented later.

---

## Summary

The fleet extension adds a clean multi-vessel layer to the repository by building on top of the canonical single-vessel VSRP architecture.

It supports:
- multi-vessel scenario construction,
- fleet-level solver benchmarking,
- fleet-level cost and emissions aggregation,
- realized fleet simulation under uncertainty,
- shared-theta CFA training and testing.

Its central design choice is **decomposition**: fleet scenarios are represented as collections of independent per-vessel VSRP instances solved separately and aggregated afterward.

This makes the fleet layer:
- easy to maintain,
- easy to benchmark,
- computationally practical,
- and architecturally consistent with the rest of the codebase.

At the same time, it should be interpreted correctly as a **decomposed fleet analysis layer**, not yet as a coupled fleet optimization model.
