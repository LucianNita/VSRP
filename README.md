# VSRP Extensions & Benchmarking

**Vessel Schedule Recovery Problem**

TJL Optimization Case Study | March 2026

---

## Overview

This repository extends the FICO Xpress community notebook implementation of the
Vessel Schedule Recovery Problem (VSRP) into a fully reproducible, solver-agnostic
codebase that:

1. **Completes the baseline model** — implements Port Swapping (Level B) and full
   Transshipment Misconnection Logic omitted from the original notebook
2. **Incorporates CO2 regulatory compliance** — EU ETS phase-in, IMO CII/EEXI,
   FuelEU Maritime penalty, and green recovery trade-off analysis
3. **Benchmarks open-source solvers** — HiGHS and CBC via a PuLP-based
   solver-agnostic MIP layer, benchmarked against FICO Xpress
4. **Implements a CFA policy** — Cost Function Approximation with additive, decay,
   and SPSA update rules under full stochastic uncertainty

---

## How to Run Scripts

All scripts must be run as **modules from the project root** so that Python
resolves internal imports correctly.

**Windows (PowerShell):**

```powershell
python -m scripts.smoke_test
python -m scripts.benchmark_smoke
python -m scripts.sensitivity_smoke
# etc.
```

**Linux / macOS:**

```bash
python -m scripts.smoke_test
python -m scripts.benchmark_smoke
python -m scripts.sensitivity_smoke
# etc.
```

> **Why `-m`?** Running `python scripts/smoke_test.py` directly does not add
> the project root to `sys.path`, causing `ModuleNotFoundError` for imports
> like `from core.entities import...`. The `-m` flag runs the file as a
> module and automatically includes the project root in the path.

---

## Repository Structure

```
.
├── core/
│   ├── entities.py
│   ├── network.py
│   ├── costs.py
│   ├── emissions.py
│   ├── simulation.py
│   └── validation.py
│
├── data/
│   ├── base_instance.py
│   └── instance_generator.py
│
├── model/
│   ├── base.py
│   ├── xpress_solver.py
│   ├── highs_solver.py
│   ├── cbc_solver.py
│   └── mip_builder.py
│
├── experiments/
│   ├── benchmark.py
│   ├── sensitivity.py
│   ├── cfa.py
│   ├── penalty_sensitivity.py
│   └── stochastic_ets.py
│
├── reporting/
│   ├── export.py
│   ├── benchmark_plots.py
│   ├── benchmark_reporting.py
│   ├── sensitivity_plots.py
│   ├── emissions_plots.py
│   ├── cfa_plots.py
│   ├── penalty_sensitivity_plots.py
│   └── stochastic_ets_plots.py
│
├── scripts/
│   ├── smoke_test.py
│   ├── benchmark_smoke.py
│   ├── sensitivity_smoke.py
│   ├── cfa_smoke.py
│   ├── transshipment_smoke.py
│   ├── stochastic_ets_smoke.py
│   ├── penalty_sensitivity_smoke.py
│   ├── penalty_speedup_multiseed_smoke.py
│   ├── penalty_plot_smoke.py
│   └── debug_alpha_case.py
│
└── results/
    ├── figures/
    └── tables/
```

---

## Installation

### Requirements

- Python 3.11+
- FICO Xpress (community license sufficient for smoke tests)
- Open-source solver dependencies (optional, for benchmarking)

### Setup

```powershell
git clone https://github.com/LucianNita/VSRP.git

python -m venv.venv
.venv\Scripts\activate          # Windows PowerShell
source.venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

### `requirements.txt`

```
xpress
pulp
highspy
mip
numpy
pandas
matplotlib
openpyxl
scipy
```

> **Note:** FICO Xpress must be installed separately. The community license is
> bundled with the `xpress` Python package. HiGHS (`highspy`) and CBC (`mip`)
> are optional — the codebase degrades gracefully if they are unavailable.

---

## Quick Start

```powershell
python -m scripts.smoke_test
```

Expected output includes:

```
overall_valid            : True
strategy_consistent      : True
max_constraint_violation : 0.0
objective_recompute_abs_gap : 0.000000
```

---

## Running Experiments

All scripts write outputs to `results/figures/` and `results/tables/`.

### 1. Baseline correctness

```powershell
python -m scripts.smoke_test
```

Validates route, strategy, timeline, container outcomes, and cost consistency
for a single Xpress solve.

### 2. Transshipment logic

```powershell
python -m scripts.transshipment_smoke
```

Tests the full Li et al. (2023) transshipment misconnection logic across
Xpress, HiGHS, and CBC on a mixed direct/transshipment container set.

### 3. Solver benchmarking

```powershell
python -m scripts.benchmark_smoke
```

Runs Xpress, HiGHS, and CBC on two random instances. Reports:

- Objective value, runtime, MIP gap, best bound
- Route validity, strategy consistency, constraint violations
- Stability metrics (obj_std, gap_std, pct_feasible, pct_optimal)

### 4. Sensitivity analysis

```powershell
python -m scripts.sensitivity_smoke
```

Produces:

- Alpha sweep (0.0 to 1.0): objective, cost decomposition, CO2, strategy mix
- Delay sweep (0h to 108h): objective, emissions, service outcomes
- ETS price sweep (25 to 130 EUR/tCO2): post-solve ETS exposure
- Green recovery trade-off plots
- CII profile across delay scenarios

### 5. CFA policy

```powershell
python -m scripts.cfa_smoke
```

Trains and tests four policies (Baseline, Additive, Decay, SPSA) over
10 training and 20 test episodes under full stochastic uncertainty.
Produces policy comparison, tail risk, and ETS compliance figures.

### 6. Penalty sensitivity

```powershell
python -m scripts.penalty_sensitivity_smoke
python -m scripts.penalty_speedup_multiseed_smoke
python -m scripts.penalty_plot_smoke
```

Sweeps swap and speed-up penalties across four random seeds.
Identifies strategy activation thresholds and route stability.

### 7. Stochastic ETS exposure

```powershell
python -m scripts.stochastic_ets_smoke
```

Solves one deterministic instance, then evaluates ETS cost exposure
across 50 stochastic carbon price scenarios. Reports VaR-like tail metrics.

### 8. Diagnostic tool

```powershell
python -m scripts.debug_alpha_case
```

Prints detailed route, timeline, validation, cost, and emissions output
for a single alpha scenario. Useful for investigating specific cases.

---

## Model Description

### Problem

The VSRP determines optimal recovery actions when a liner vessel faces
disruptions. Given an initial delay, the model selects:

- Which ports to visit or skip
- Sailing speed on each leg
- Port handling duration (standard or expedited)
- Whether to reorder port visits (swap)

to minimise a weighted combination of operational and service disruption costs.

### Objective

```
min  (1 - alpha) * C_op  +  alpha * C_svc
```

where `alpha` in [0,1] controls the service/cost trade-off.

**Operational cost** includes fuel, port handling, strategy penalties
(speed-up, expedited, omission, swap), port-specific penalties, and
optionally the FuelEU non-compliance penalty.

**Service cost** includes delay penalties and misconnection penalties.

### Recovery Strategies

| Strategy | Description |
|---|---|
| Speed adjustment | Sail faster (FAST 25kn) or slower (SLOW 15kn) than nominal (20kn) |
| Expedited handling | Reduce port-call duration from 12h to 8h |
| Port omission | Skip one or more planned port calls |
| Port swapping | Reorder two port visits (non-adjacent, Brouer et al. 2013) |

### Swap Implementation

Non-adjacent port swapping is implemented using:

- Three-leg swap groups (leg A: approach, leg B: reverse, leg C: continue)
- Binary ordering variables z_ij with antisymmetry and transitivity constraints
- `max_swap_distance=2` bound to prevent degenerate routes
- Swap penalty charged once per active swap group

### Transshipment Misconnection

Full Li et al. (2023) logic: a container is misconnected if the vessel
arrives at its transshipment port after `connecting_service_deadline_h`.
Non-connecting edges are identified at model-build time and constrained
to force `o[c] = 1`.

### Emissions and Regulatory Compliance

| Module | Coverage |
|---|---|
| CO2 accounting | Cubic fuel model per leg, VLSFO emission factor 3.114 tCO2/t |
| EU ETS | Phase-in schedule (40/70/100% for 2024/2025/2026+), stochastic carbon price |
| IMO CII | Attained vs required CII, A-E rating (IMO MEPC 2024 reference lines) |
| IMO EEXI | Vessel-level compliance check (speed-limit precondition) |
| FuelEU Maritime | GHG intensity limit interpolation, energy-based penalty formula |

**Single-fuel assumption (VLSFO):** All legs burn VLSFO at constant GHG intensity
91.16 gCO2eq/MJ. The FuelEU penalty is therefore linear in total fuel burn.
The 2026 FuelEU limit is approximately 88.61 gCO2eq/MJ (interpolated between
2025=89.34 and 2030=85.69), making VLSFO always non-compliant from 2025 onward.

---

## CFA Policy

Following the Powell (2022) Cost Function Approximation framework:

```
parametric optimization  ->  simulator  ->  parameter update  ->  repeat
```

**Theta:** Per-destination promise tightening vector. Higher theta at port p
forces the MIP to treat container deadlines at p as tighter, encouraging
more aggressive recovery.

**Uncertainty sources:**

| Source | Distribution |
|---|---|
| Initial delay | Truncated normal, mean=55h, std=10h |
| Port handling multipliers | Truncated normal, mean=1.0, std=0.1, per port |
| Weather speed factors | Truncated normal, mean=1.0, std=0.05, per leg |
| Carbon price | Truncated normal, mean=65, std=15 EUR/tCO2 |
| Fuel price | Truncated normal, mean=600, std=80 USD/tonne |

**Update policies:**

| Policy | Rule |
|---|---|
| Additive | theta_p += step * misses_p |
| Decay | Increase where misses occur, decrease where none |
| SPSA | Simultaneous perturbation gradient estimate with Rademacher delta |

**Results (20 test episodes):**

| Policy | Avg missed | Avg service cost | vs Baseline |
|---|---|---|---|
| Baseline | 4.00 | $298,912 | — |
| Additive | 3.60 | $272,439 | -8.9% |
| Decay | 3.70 | $277,196 | -7.3% |
| **SPSA** | **3.00** | **$243,900** | **-18.4%** |

SPSA achieves zero variance (p90 = p95 = mean = $243,900), demonstrating
robust convergence to a consistently better policy.

---

## Benchmark Results

Two random instances, three solvers, `time_limit=60s`, `mip_gap=0.01`.

| Solver | Avg objective | Avg runtime | MIP gap | Feasible | Optimal |
|---|---|---|---|---|---|
| Xpress | $501,429 | 0.015s | 0.0% | 100% | 100% |
| HiGHS | $501,429 | 0.050s | 0.0% | 100% | 100% |
| CBC | $501,429 | 0.258s | 0.0% | 100% | 100% |

All three solvers find identical optimal solutions. Xpress is 3.3x faster
than HiGHS and 17x faster than CBC on these instances.

**Solver independence:** HiGHS and CBC use a PuLP-based MIP builder
(`model/mip_builder.py`) that is fully independent of Xpress. The formulation
is identical across all three solvers.

---

## Key Design Decisions

### Solver-agnostic architecture

All experiments use `BaseSolver` as the interface. Swapping solvers requires
only changing the solver object passed to experiment functions — no model
changes needed.

### Promised arrival calibration

Container promised arrivals are calibrated to the expected initial delay
(48h) at nominal speed (20 knots). This ensures the delay constraint
correctly classifies containers as on-time or late across the delay sweep
range (0h to 108h).

### Swap penalty charging

The swap penalty is charged once per active swap group (not once per swapped
port) in both the MIP objective and the post-solve cost recomputation.
This ensures `objective_recompute_abs_gap = 0` for all instances.

### FuelEU penalty formula

The corrected energy-based formula (Article 23, FuelEU Maritime Regulation):

```
Penalty = max(0, g - g*) / g  *  F_total  *  penalty_rate  *  FX
```

where `g` = VLSFO GHG intensity (91.16 gCO2eq/MJ), `g*` = FuelEU limit,
`F_total` = total fuel burned (tonnes), `penalty_rate` = EUR 2,400 per tonne
VLSFO equivalent, `FX` = EUR/USD conversion rate.

---

## Limitations and Future Work

| Limitation | Notes |
|---|---|
| Single-fuel (VLSFO) | Multi-fuel would require per-leg fuel-choice variables |
| Per-voyage FuelEU | Regulation applies annually; per-voyage is a standard approximation |
| Delay constraint approximation | Uses nominal speed on prior legs; big-M formulation would be exact |
| time_to_first_feasible | Unavailable on Xpress community license |
| Delay sweep flat above 48h | Structural property of this container set; more containers would show more variation |
| Port swapping rarely activates | Consistent with Brouer et al. (2013) finding that swap is rarely optimal |

---

## References

1. Brouer, B.D. et al. (2013). The Vessel Schedule Recovery Problem (VSRP).
   *European Journal of Operational Research*, 224(2), 362-374.

2. Li, S. et al. (2023). Vessel schedule recovery strategy in liner shipping
   considering expected disruption. *Ocean and Coastal Management*, 237, 106514.

3. Hu, Liu, Jin, Wang (2024). Liner disruption recovery problem with emission
   control area policies. *Transportation Research Part D*, 132, 104227.

4. Zhou et al. (2024). Strategy and Impact of Liner Shipping Schedule Recovery
   under ECA Regulation. *Journal of Marine Science and Engineering*.

5. Li, S. and Wang, T. (2025). How emissions trading system affects liner ship
   disruption recovery. *Transport Policy*, 169, 191-208.

6. Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization*.
   Wiley. [CFA framework]

7. IMO MEPC.338(76) — CII reference lines for container ships.

8. Regulation (EU) 2023/957 — EU ETS maritime phase-in schedule.

9. FuelEU Maritime Regulation — GHG intensity limits and Article 23 penalty.

