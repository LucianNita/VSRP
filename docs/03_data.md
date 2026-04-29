# Data Module Documentation

## Purpose of `data/`

The `data/` package is responsible for constructing the **input side** of the VSRP.

It defines:
- the fixed case-study route and distance matrix
- default speed, port-call, and penalty settings
- random container generation logic
- helper factories for reproducible instances

In architectural terms, `data/` answers the question:

> **what optimization problem instance are we solving?**

It is the layer that produces canonical `VSRPInstance` objects for the solver backends.

---

## Files in `data/`

| File | Main role |
|---|---|
| `base_instance.py` | Defines the fixed case-study route, default parameters, and instance factory |
| `instance_generator.py` | Generates random container sets and calibrates promised arrivals / penalties |

---

## Architectural role of `data/`

The repository is built around canonical `VSRPInstance` objects.
Those objects are not assembled manually inside the solver code.
Instead, they are created by the data layer.

This separation is important because it allows:
- reproducible experiments under different seeds
- sensitivity analysis on fixed demand sets
- easy reuse of the same instance across multiple solver backends
- clear separation between **problem generation** and **problem solving**

---

## 1. `data/base_instance.py`

### Purpose

`base_instance.py` defines the **fixed case-study route** and the default operational settings
used across the repository.

It is the closest thing in the codebase to a static scenario template.

### Main contents

#### `BASE_PORTS`
The nominal planned route, represented as an ordered list of ports.

This order matters because:
- the network generator builds forward edges relative to this sequence
- swap feasibility is defined relative to this sequence
- planned arrivals and departures assume this sequence

#### `BASE_DISTANCE_MATRIX_NM`
Pairwise distance matrix aligned with `BASE_PORTS`.

This matrix is used by:
- network generation
- travel-time calculation
- fuel and emissions calculation
- planned schedule reconstruction

#### `DEFAULT_SPEED_LEVELS_KNOTS`
The discrete sailing speed options used by the optimization model.

#### `DEFAULT_PORT_CALL_PROFILE`
The default port-call duration and handling-cost options.

In the current project, these are:
- STANDARD
- EXPEDITED

#### `DEFAULT_PENALTIES`
Default tactical penalty settings for:
- speed-up
- expedited port
- omission
- swap

### Main function

#### `build_base_instance()`
This is the main factory function in the file.

It builds a canonical `VSRPInstance` from:
- generated containers
- the fixed route template
- default or overridden operational settings

### Why this file matters

This file ensures that all experiments share a consistent baseline network.
Without it, every script would need to manually rebuild route, speed, penalty,
and regulation settings, which would make reproducibility much weaker.

---

## 2. `data/instance_generator.py`

### Purpose

`instance_generator.py` creates random container demand sets that are still
internally consistent with the route and timing assumptions of the model.

This file is critical because the project’s behavior depends heavily on how:
- promised arrivals are calibrated
- penalties are assigned
- transshipment containers are created

### Main design goal

The generator is not just producing random origin-destination pairs.
It is also producing **service promises** that are meant to be meaningful
under the model’s nominal timing assumptions.

That is why promised-arrival calibration is one of the most important design choices here.

---

## Promised-arrival calibration

### Why it matters

If promised arrivals are calibrated against a zero-delay baseline, then any non-zero
initial disruption immediately makes all containers structurally late.

That destroys the usefulness of:
- delay sensitivity experiments
- CFA learning signal
- realistic service-cost trade-offs

### Current approach

The generator computes nominal arrivals using:
- nominal speed
- nominal port-call duration
- **expected initial delay**

Then it adds promised slack on top.

This means container promises are calibrated to realistic disrupted operations,
not to an unrealistic zero-delay world.

This is a major improvement over a naive promise generator.

---

## Main components of `instance_generator.py`

### `cumulative_nominal_arrivals()`
Computes nominal cumulative arrival times along the route.

This helper underpins promise calibration and transshipment deadline generation.

### `generate_containers()`
The main random demand generator.

It samples:
- origin and destination
- quantity
- priority
- optional transshipment structure
- promised arrival
- penalties

### `generate_small_test_set()`
Convenience wrapper for quick debugging / smoke testing.

### `generate_transshipment_test_set()`
Convenience wrapper for generating mixed direct/transshipment demand sets.

---

## Container penalties

The generator supports two broad modes:

### Notebook-compatible mode
Delay penalty is based on the original notebook-style scaling,
and misconnection uses a fixed penalty.

This mode helps preserve comparability with the original baseline implementation.

### Stress-test mode
Delay and misconnection penalties are drawn from wider random ranges.

This mode is more useful for experimentation and robustness testing.

---

## Transshipment generation

If transshipment generation is enabled:
- a transshipment port is selected between origin and destination
- a connecting-service deadline is constructed from the nominal schedule
- the resulting container becomes subject to misconnection logic in the model

This provides a compact way to represent transshipment without explicitly modeling
a second vessel or downstream service network.

---

## Why the data layer is important

The `data/` package determines the experimental behavior of the whole project.
Even if the optimization model is correct, poor data generation can make results:
- trivial
- insensitive
- unrealistic
- impossible to interpret

Your repository avoids that by carefully calibrating promises, delays,
penalties, and optional transshipment structure.

---

## Relationship to the rest of the codebase

### `data/` -> `model/`
The data layer creates the canonical `VSRPInstance` consumed by the solvers.

### `data/` -> `experiments/`
Experiments repeatedly call the generator with different seeds or settings to
produce controlled input variation.

### `data/` -> `core/`
The data layer uses canonical dataclasses from `core/entities.py`.

---

## Summary of `data/`

The `data/` package is the instance-construction layer of the repository.

It provides:
- the fixed route template
- default speed and penalty settings
- reproducible random container generation
- calibrated service promises
- optional transshipment demand

This makes it possible to run reproducible and interpretable experiments
without embedding instance-generation logic inside the solver or experiment code.
