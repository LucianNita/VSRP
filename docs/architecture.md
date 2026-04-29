# Project Architecture

## Purpose

This document explains the high-level architecture of the VSRP codebase:
- what each folder is responsible for
- how data flows through the system
- how scripts, experiments, solvers, and reporting interact

The design goal of the repository is to provide a **solver-agnostic, reproducible, engineering-grade extension** of the Vessel Schedule Recovery Problem (VSRP).

---

## Core design principle

The central architectural idea is:

> **all solver backends and experiments should communicate through canonical problem and solution objects**

The two most important shared objects are:
- `VSRPInstance`
- `VSRPSolution`

These are defined in `core/entities.py` and are used across:
- instance generation
- model solving
- validation
- emissions
- benchmarking
- plotting

This allows the repository to remain modular and mostly solver-independent.

---

## Folder responsibilities

### `core/`
Shared solver-agnostic logic:
- canonical dataclasses
- network construction
- cost recomputation
- emissions and regulatory metrics
- simulation under uncertainty
- validation

This is the **common logic layer** of the codebase.

### `data/`
Problem-input generation:
- fixed case-study route and distances
- random container generation
- base instance factory

This layer constructs the inputs passed to the solvers.

### `model/`
Optimization layer:
- abstract solver interface
- native Xpress backend
- open-source HiGHS and CBC backends
- solver-agnostic PuLP MIP builder

This layer transforms `VSRPInstance` objects into `VSRPSolution` objects.

### `experiments/`
Experiment orchestration:
- benchmarking
- sensitivity analysis
- CFA training/testing
- penalty sensitivity
- stochastic ETS exposure

This layer repeatedly builds instances, solves them, and converts results into tidy tabular outputs.

### `reporting/`
Output generation:
- table saving
- figure saving
- benchmark plots
- sensitivity plots
- emissions plots
- CFA plots
- penalty sensitivity plots
- stochastic ETS plots

This layer consumes experiment outputs and writes files into `results/`.

### `scripts/`
Executable entry points:
- smoke tests
- diagnostics
- experiment runners

These are the top-level user-facing commands.

### `results/`
Generated artifacts:
- figures
- tables

This directory is the sink for experiment outputs.

---

## High-level dependency structure

```text
scripts
   ↓
experiments
   ↓
model
   ↓
core + data
   ↓
reporting
   ↓
results
```

---

## Runtime execution flow

### Example: running a smoke script

Command:

```bash
python -m scripts.smoke_test
```

Execution flow:

1. `scripts/smoke_test.py`
   - generates containers
   - builds a base instance
   - selects the Xpress solver
   - calls the solver

2. `model/xpress_solver.py`
   - builds the network
   - creates the MIP
   - solves the instance
   - reconstructs a canonical solution

3. `core/validation.py`
   - validates route, strategies, timeline, and structural consistency

4. `core/emissions.py`
   - computes fuel, CO2, ETS, FuelEU, and CII metrics

5. `core/costs.py`
   - recomputes cost breakdown and checks consistency with the reported objective

6. `scripts/smoke_test.py`
   - prints the final structured output

---

## Data flow

### Inputs

The main inputs are:
- fixed port rotation and distance matrix from `data/base_instance.py`
- randomly generated containers from `data/instance_generator.py`
- solver settings from `model/base.py`
- optional experiment parameters from `experiments/*`

### Canonical problem object

All optimization starts from:

- `VSRPInstance`

This object packages:
- ports
- distance matrix
- containers
- penalties
- speed levels
- port-call durations
- disruption parameters
- regulatory settings

### Canonical solution object

All solver outputs are normalized into:

- `VSRPSolution`

This object may contain:
- route legs
- skipped / swapped ports
- strategy decisions
- timeline
- container outcomes
- solver statistics
- validation summary
- emissions summary
- metadata

---

## Solver architecture

### Native backend

`model/xpress_solver.py` is the reference implementation:
- full model build
- full solution extraction
- validation
- emissions attachment

### Open-source backends

`model/highs_solver.py` and `model/cbc_solver.py` use:
- `model/mip_builder.py`
- MPS export
- backend-specific solve APIs

They currently return **partial canonical solutions**:
- objective
- runtime
- bounds
- gap
- availability metadata

They do **not** yet reconstruct full route/timeline/container outputs.

---

## Key architectural strengths

- canonical shared entities
- solver-agnostic core logic
- native + export-based solver support
- experiment functions returning tidy DataFrames
- reporting layer isolated from optimization layer
- smoke scripts for end-to-end verification

---

## Known architectural limitations

- HiGHS and CBC are currently partial-solution backends
- Xpress is the only backend with full route/timeline/container extraction
- some timing logic uses conservative approximation rather than full exact per-container timing variables
- regulatory modules are implemented under a single-fuel VLSFO assumption

