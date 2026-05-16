# Model Module Documentation

## Purpose of `model/`

The `model/` package contains the optimization layer of the repository.

Its role is to transform a canonical `VSRPInstance` into a canonical
`VSRPSolution` using one of several solver backends.

This package is where the mathematical optimization model is actually built
and solved.

---

## Files in `model/`

| File | Main role |
|---|---|
| `base.py` | Abstract solver interface and shared solver helpers |
| `xpress_solver.py` | Native full-feature Xpress backend |
| `highs_solver.py` | HiGHS export-based backend |
| `cbc_solver.py` | CBC export-based backend |
| `mip_builder.py` | Solver-agnostic PuLP model builder used by open-source backends |

---

## Architectural role of `model/`

The codebase is designed so that experiments do not depend directly on solver APIs.
Instead, they depend on the abstract solver interface defined in `model/base.py`.

This allows the same experiment code to run with:
- Xpress
- HiGHS
- CBC

without changing the experiment logic itself.

The `model/` package is therefore the bridge between:
- canonical problem objects from `core/` and `data/`
- concrete optimization engines

---

## 1. `model/base.py`

### Purpose

`base.py` defines the common solver interface shared by all backends.

This is the file that makes the architecture solver-agnostic.

### Main components

#### `SolveOptions`
Stores generic solve controls such as:
- time limit
- relative MIP gap target
- logging flag
- optional random seed

This gives experiments a unified way to configure different solvers.

#### `BaseSolver`
Abstract base class defining the common API:

- `solve(instance, options)`
- `is_available`

Every concrete backend must implement this interface.

#### `build_empty_solution()`
Shared helper to build a canonical empty/error solution.

This is useful when a solver:
- is unavailable
- errors during solve
- returns no feasible solution

### Why this file matters

Without `base.py`, the rest of the repository would need solver-specific branches everywhere.
With `base.py`, experiments can treat all solvers uniformly.

---

## 2. `model/xpress_solver.py`

### Purpose

`xpress_solver.py` is the **reference native backend** of the repository.

It is the most complete solver implementation and the one used for:
- full route reconstruction
- strategy extraction
- timeline reconstruction
- container-outcome extraction
- validation attachment
- emissions attachment

### Why Xpress is special in this project

The Xpress backend is not just a solver wrapper. It is the only backend that
currently reconstructs a **full canonical solution** object.

That means it returns:
- route legs
- skipped and swapped ports
- strategy tags
- timeline entries
- container delayed/misconnected outcomes
- validation summary
- emissions summary

### Main internal stages

#### Model build
The Xpress backend:
- builds the network from `core/network.py`
- creates decision variables
- adds flow, port-call, swap, strategy, delay, misconnection, and objective constraints

#### Solve
The model is solved through the native Xpress Python API.

#### Extraction
After solve, the backend reconstructs a canonical `VSRPSolution` from
the selected variable values.

### Why this file matters

This is the repository’s most important optimization implementation.
It is the benchmark reference against which the open-source backends are compared.

---

## 3. `model/mip_builder.py`

### Purpose

`mip_builder.py` contains the solver-agnostic **PuLP-based formulation builder**.

It exists so that open-source solvers can solve the same model without depending
on the native Xpress API.

### Main role

The builder creates the full VSRP mixed-integer model using PuLP and can export
it to MPS format.

This exported MPS is then read by:
- HiGHS
- CBC

### Why this file matters

This file is what makes the repository genuinely multi-solver.

Without it, HiGHS and CBC would not have an independent model-construction path.

### Formulation content

The builder includes:
- flow conservation
- port-call constraints
- swap-group constraints
- strategy classification
- delay indicators
- misconnection logic
- FuelEU penalty term when enabled
- weighted operational/service objective

### Important note

The goal is formulation-level parity with the Xpress backend, not a separate model.
That is why objective values can be meaningfully compared across solvers.

---

## 4. `model/highs_solver.py`

### Purpose

`highs_solver.py` is the HiGHS backend.

It uses:
- `mip_builder.py` to create the PuLP model
- MPS export
- the `highspy` API to solve the exported formulation

### Current output level

HiGHS currently returns a **partial canonical solution**.

That means it reports:
- objective value
- runtime
- MIP gap
- best bound
- node / iteration information where available

but does **not** yet reconstruct:
- route legs
- timeline
- container outcomes
- validation or emissions summary

### Why this file matters

It provides an open-source benchmark point against the Xpress implementation.
Even without full route extraction, it is very valuable for:
- objective consistency testing
- runtime benchmarking
- gap / bound comparisons

---

## 5. `model/cbc_solver.py`

### Purpose

`cbc_solver.py` is the CBC backend.

Like HiGHS, it uses:
- the PuLP-based formulation builder
- MPS export
- a backend-specific API (`python-mip`) to solve the exported model

### Current output level

CBC also returns a **partial canonical solution**.

It reports:
- objective value
- runtime
- MIP gap where available
- best bound where available
- node count where available

but not full route/timeline/container extraction.

### Why this file matters

It gives the repository a second open-source benchmark solver and helps confirm
that the model is not dependent on a single commercial backend.

---

## Common optimization workflow

All model backends follow the same high-level lifecycle:

1. Receive a canonical `VSRPInstance`
2. Build or export the optimization model
3. Apply solve options
4. Solve the model
5. Extract available results into a canonical `VSRPSolution`
6. Return the canonical result to the experiment or script layer

This common lifecycle is one of the main reasons the repository is easy to benchmark.

---

## Relationship to the rest of the codebase

### `model/` depends on `core/`
The model layer uses:
- canonical entities
- network generation
- validation
- emissions
- cost recomputation

### `model/` depends on `data/`
Indirectly, because instances are built in the data layer before being solved.

### `experiments/` depends on `model/`
Experiments choose which solver backend to use and pass instances into it.

---

## Why the separation between Xpress and PuLP builder matters

There are effectively two modeling paths in the repository:

### Path 1: native Xpress
- direct variable creation in Xpress
- direct solve
- full extraction

### Path 2: open-source replication
- build equivalent formulation in PuLP
- export to MPS
- solve in HiGHS or CBC
- partial extraction

This separation matters because it allows the repository to demonstrate:
- solver independence at the formulation level
- benchmark comparability
- native-backend richness vs open-source portability trade-offs

---

## Strengths of the `model/` layer

- common solver interface
- native full-feature reference backend
- open-source benchmark backends
- shared formulation logic across solvers
- clear separation between solving and reporting

---

## Current limitations of the `model/` layer

- only Xpress currently supports full canonical solution extraction
- HiGHS and CBC are benchmark-capable but not yet route-extraction-capable
- the delayed and misconnection logic uses conservative timing approximations
- FuelEU remains a single-fuel linear penalty proxy, not a full multi-fuel model

---

## Summary of `model/`

The `model/` package is the optimization engine of the repository.

It provides:
- a shared solver API
- a full native Xpress implementation
- open-source HiGHS and CBC benchmark backends
- a solver-agnostic PuLP formulation builder

This package is what turns the canonical problem objects from `core/` and `data/`
into solved optimization outputs.

