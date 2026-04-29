# Core Module Documentation

## Purpose of `core/`

The `core/` package contains the **solver-agnostic foundation** of the VSRP codebase.

Its responsibilities are to:
- define canonical problem and solution entities
- build and describe the recovery network
- recompute costs from canonical solutions
- compute emissions and regulatory metrics
- simulate realized performance under uncertainty
- validate extracted solutions independently of the solver backend

This package is the architectural center of the repository. Other layers
such as `model/`, `experiments/`, and `reporting/` depend heavily on it,
but `core/` itself does not depend on solver-specific APIs.

---

## Why `core/` matters

The project is designed around two canonical shared objects:
- `VSRPInstance`
- `VSRPSolution`

These objects allow the codebase to:
- solve the same model with multiple backends
- benchmark solvers consistently
- recompute costs and emissions independently of the solver
- validate results in a solver-agnostic way
- feed tidy outputs into reporting and plots

Without `core/`, the repository would collapse back into a set of
solver-specific scripts instead of a reusable architecture.

---

## File overview

| File | Main role |
|---|---|
| `entities.py` | Canonical dataclasses for instances, solutions, outcomes, solver stats, validation, emissions |
| `network.py` | Builds the feasible route network, including skip edges and swap groups |
| `costs.py` | Recomputes operational and service costs from canonical solutions |
| `emissions.py` | Computes fuel, CO2, ETS, FuelEU, CII, and emissions summaries |
| `simulation.py` | Simulates realized performance under uncertainty for CFA and stochastic experiments |
| `validation.py` | Solver-agnostic route / strategy / timeline / structural consistency checks |

---

## 1. `core/entities.py`

### Purpose

`entities.py` defines the **canonical data model** of the repository.

It contains all of the key dataclasses used to represent:
- input data
- network edges
- solved route legs
- strategy decisions
- container outcomes
- solver statistics
- validation summaries
- emissions summaries
- experiment/benchmark records

This file is foundational because it decouples the rest of the project
from solver-specific data structures.

### Most important classes

#### `Container`
Represents one cargo demand item, including:
- origin and destination indices
- promised arrival
- delay penalty
- misconnection penalty
- optional transshipment port(s)
- optional connecting-service deadline

This is the key service-level demand object.

#### `Edge`
Represents one feasible network arc in the recovery model.

It stores:
- from/to port indices
- sailing speed
- travel time
- fuel cost
- whether the edge is part of a swap group
- which planned ports are skipped by traversing the edge

This is the key network object used by both Xpress and the open-source MIP builder.

#### `VSRPInstance`
Represents one full optimization instance.

It packages together:
- ports
- distance matrix
- generated containers
- delay level
- speed levels
- port-call profile
- penalties
- swap settings
- fuel / FuelEU settings

This is the main problem object passed into all solver backends.

#### `VSRPSolution`
Represents one canonical solved solution.

It may include:
- selected route legs
- skipped ports
- swapped ports
- strategy decisions
- timeline
- container outcomes
- solver stats
- validation result
- emissions summary

This is the main solved object consumed by experiments and reporting.

### Design significance

The canonical dataclasses are one of the strongest architectural choices
in the project. They make it possible to:
- compare different solvers fairly
- recompute diagnostics outside the solver
- treat experiments as consumers of normalized outputs instead of
  solver-specific data

---

## 2. `core/network.py`

### Purpose

`network.py` builds the **feasible routing network** used by the optimization model.

It is responsible for translating the high-level route structure into a set of
feasible edges that can be selected by the MIP.

### Main responsibilities

#### Forward edges
The module creates forward edges from port `i` to port `j` where:
- `j > i`
- no more than `max_skip` planned ports are skipped

These edges implement normal progression through the route, including omissions.

#### Swap groups
If swapping is enabled, the module creates swap groups with three structural legs:
- Leg A: predecessor of `i` to `j`
- Leg B: `j` back to `i`
- Leg C: `i` forward to successor of `j`

Each swap group is identified by a shared `swap_group_id`.

This network-level representation is what allows the optimization model to express
non-adjacent port reordering.

### Helper functions

#### `build_network()`
Constructs the full edge list.

#### `build_network_from_instance()`
Convenience wrapper that reads the relevant fields from `VSRPInstance`.

#### `get_swap_groups()` / `get_swap_group_ids()`
Utilities for grouping and indexing swap edges.

#### `get_swap_group_port_pair()`
Recovers the actual swapped port pair `(i, j)` from a swap group's reverse leg.

#### `compute_planned_arrivals()` / `compute_planned_departures()`
Build nominal timeline references used in solution reconstruction and validation.

### Design significance

This file isolates all optimization-independent route-network logic from the solver layer.
That is why both the Xpress backend and the PuLP builder can use the same network structure.

---

## 3. `core/costs.py`

### Purpose

`costs.py` recomputes the objective decomposition from a canonical solution.

This is important for:
- validating solver output
- detecting objective inconsistencies
- producing report-ready cost breakdowns

### Main outputs

#### `CostBreakdown`
A structured dataclass containing:
- fuel cost
- port-call cost
- strategy penalty cost
- port-specific penalty cost
- FuelEU penalty cost
- delay cost
- misconnection cost
- operational cost
- service cost
- weighted objective

#### `compute_cost_breakdown()`
The main entry point for post-solve cost recomputation.

### Important modeling detail

The swap penalty is charged **once per active swap group**, not once per swapped port.

This matters because:
- the canonical solution reports `PORT_SWAP` tags per swapped port
- but the optimization model treats one swap action as one group-level decision

This alignment is essential for obtaining near-zero objective recomputation gaps.

---

## 4. `core/emissions.py`

### Purpose

`emissions.py` computes environmental and regulatory metrics from route legs or full solutions.

This file is one of the most important extensions relative to the original notebook.

### Main capabilities

#### Fuel and CO2
Uses a cubic-speed fuel model to estimate:
- fuel consumption per leg
- total fuel burn
- CO2 emissions using a VLSFO emission factor

#### EU ETS
Computes ETS exposure using:
- CO2 emissions
- carbon price
- EU ETS phase-in schedule

#### FuelEU Maritime
Computes a FuelEU non-compliance penalty using an energy-based formulation.

#### CII
Computes:
- attained CII
- required CII
- A–E rating

#### EEXI scaffolding
Provides vessel-level EEXI compliance support.

### Main functions

#### `compute_leg_emission_record()`
Computes emissions and regulatory exposure for one route leg.

#### `compute_solution_leg_emissions()`
Computes leg-level emissions records for an extracted solution.

#### `compute_solution_emissions_summary()`
Main entry point for solution-level emissions summaries.

### Important assumptions

- single-fuel VLSFO assumption
- constant GHG intensity across all legs
- FuelEU modeled as a voyage-level proxy rather than annual compliance

These assumptions are reasonable for the current case-study scope but should be stated clearly.

---

## 5. `core/simulation.py`

### Purpose

`simulation.py` provides the uncertainty and realized-performance layer used by CFA and stochastic experiments.

This file is what turns a deterministic optimization model into a repeated-decision system under uncertainty.

### Main concepts

#### `UncertaintyScenario`
Represents one realized episode, including:
- realized initial delay
- port handling multipliers
- weather speed factors
- realized carbon price
- realized fuel price

#### `UncertaintyConfig`
Stores the distribution settings used to sample uncertainty.

#### `sample_uncertainty_scenario()`
Draws one realized scenario for simulation.

#### `compute_actual_route_arrivals()`
Reconstructs actual arrival times for a fixed route under a realized scenario.

#### `simulate_realized_performance()`
Evaluates a solved route under realized uncertainty and returns:
- missed containers
- service cost
- realized fuel cost
- realized ETS cost

### Design significance

This file is crucial for the CFA framework because the optimization problem is solved on an estimated state,
but evaluated on a realized state.

That separation between **planning** and **realization** is central to the policy-learning setup.

---

## 6. `core/validation.py`

### Purpose

`validation.py` provides solver-agnostic consistency checks for extracted solutions.

This is important because solver feasibility alone is not enough once solutions are:
- reconstructed into route legs
- tagged with strategies
- converted into timelines
- compared across multiple backends

### Validation layers

#### Route validation
Checks:
- non-empty route
- valid port indices
- starts at origin
- ends at sink
- adjacency continuity
- no repeated interior visits

#### Timeline validation
Checks:
- valid timeline entries
- departure not before arrival
- monotonic chronology
- consistent stored delay values

#### Strategy validation
Checks:
- skipped ports vs omission tags
- swapped ports vs swap tags
- basic swap-group route integrity

#### Container validation
Checks:
- outcome records exist for all containers
- origin/destination consistency
- transshipment consistency

#### Structural constraint residuals
Reconstructs approximate numerical residuals for:
- flow balance
- port-call consistency
- omission consistency

### Important design note

Delay-classification consistency is **not** treated as a hard numerical violation.
This is intentional because:
- the MIP delay variable uses a conservative approximation
- the reporting timeline is more explicit
- the two can differ without implying model infeasibility

This is a subtle but important modeling choice.

---

## Summary of `core/`

The `core/` package is the shared foundation of the repository.

It is responsible for:
- defining the language of the optimization problem
- representing route/network structure
- turning solver outputs into meaningful business and regulatory metrics
- validating extracted solutions independently of the solver
- enabling simulation and CFA under uncertainty

If `model/` is the optimization engine, then `core/` is the shared domain model and analytics layer.
