# =============================================================================
# Solver-agnostic emissions and regulatory-cost calculations for the Vessel
# Schedule Recovery Problem (VSRP).
#
# Single-fuel assumption (VLSFO):
# --------------------------------
# All sailing legs are assumed to burn Very Low Sulphur Fuel Oil (VLSFO),
# the dominant marine fuel since the IMO 2020 sulphur cap. Key properties:
#   - CO2 emission factor : 3.114 tCO2 per tonne of fuel burned
#   - GHG intensity       : 91.16 gCO2eq/MJ (well-to-wake, IMO default)
#   - Lower heating value : 40.5 MJ/kg (used in FuelEU penalty formula)
#   - Sulphur content     : <= 0.5% (IMO 2020 compliant)
#
# Under this assumption:
#   - GHG intensity is constant across all legs and speeds
#   - FuelEU penalty is linear in total fuel consumption
#   - No fuel-choice variables are required in the MIP
#
# FuelEU penalty formula (corrected):
# ------------------------------------
# The FuelEU Maritime regulation Article 23 specifies a penalty of
# EUR 2,400 per tonne of VLSFO equivalent that would need to be
# replaced by a compliant fuel to meet the GHG intensity limit.
#
# The formula is:
#
#   excess_energy_MJ = max(0, g - g*) / g * E_total
#   penalty_fuel_t   = excess_energy_MJ / LHV_VLSFO_MJ_per_t
#   penalty_EUR      = penalty_fuel_t * 2400
#
# where:
#   g       = attained GHG intensity [gCO2eq/MJ]
#   g*      = FuelEU limit [gCO2eq/MJ]
#   E_total = total energy content of fuel burned [MJ]
#   LHV     = lower heating value of VLSFO [MJ/tonne] = 40,500
#
#
# The FuelEU penalty in the objective is a per-voyage approximation.
# The regulation applies annually across all voyages; per-voyage
# approximation is standard in the literature (Hu et al. 2024,
# Zhou et al. 2024, Li & Wang 2025).
#
# Multi-fuel extensions would require per-leg fuel-choice variables
# and are outside the current scope.
#
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from core.entities import EmissionsSummary, VSRPInstance, VSRPSolution


# =============================================================================
# 1. DEFAULT PARAMETERS
# =============================================================================

DEFAULT_REFERENCE_SPEED_KNOTS: float = 20.0
DEFAULT_FUEL_BASE_CONSUMPTION_TPD: float = 100.0
DEFAULT_FUEL_PRICE_USD_PER_TONNE: float = 600.0

# VLSFO emission factor (IMO 4th GHG Study 2020)
DEFAULT_CO2_EMISSION_FACTOR_TCO2_PER_TFUEL: float = 3.114

# VLSFO lower heating value [MJ/tonne]
# Used in FuelEU penalty formula
# Source: IMO LCA guidelines, FuelEU Maritime Annex I
VLSFO_LHV_MJ_PER_TONNE: float = 40_500.0

# EU ETS phase-in schedule (Regulation EU 2023/957)
# 2024: 40% of covered emissions must surrender allowances
# 2025: 70%
# 2026+: 100%
DEFAULT_EU_ETS_PHASE_IN: dict[int, float] = {
    2024: 0.40,
    2025: 0.70,
    2026: 1.00,
}

DEFAULT_EU_ETS_CARBON_PRICE_EUR_PER_TCO2: float = 65.0

# VLSFO GHG intensity (well-to-wake, gCO2eq/MJ)
# Source: FuelEU Maritime Annex I, IMO LCA guidelines
DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ: float = 91.16

# FuelEU Maritime GHG intensity limits (gCO2eq/MJ)
# 2025: 2% reduction from 2020 baseline of 91.16
# 2030: 6% reduction
# 2035: 14.5% reduction
FUELEU_GHG_LIMITS: dict[int, float] = {
    2025: 89.34,   # 91.16 * (1 - 0.02)
    2030: 85.69,   # 91.16 * (1 - 0.06)
    2035: 77.94,   # 91.16 * (1 - 0.145)
    2040: 62.59,   # 91.16 * (1 - 0.31)
    2045: 31.91,   # 91.16 * (1 - 0.65)
    2050: 2.28,    # 91.16 * (1 - 0.80) approx
}
DEFAULT_FUELEU_GHG_LIMIT_GCO2EQ_PER_MJ: float = FUELEU_GHG_LIMITS[2025]

# FuelEU non-compliance penalty rate
# EUR 2,400 per tonne of VLSFO equivalent that exceeds the limit
# Source: FuelEU Maritime Regulation Article 23
DEFAULT_FUELEU_PENALTY_EUR_PER_TONNE_VLSFO_EQUIV: float = 2_400.0

# Representative container vessel parameters for CII
DEFAULT_VESSEL_DWT: float = 100_000.0
DEFAULT_VESSEL_GT: float = 80_000.0

# IMO CII reference lines for container ships (2024 guidelines)
# Attained CII = CO2 [g] / (DWT [t] * Distance [nm])
# Required CII = a * DWT^(-c)
# Container ship parameters (IMO MEPC.338(76) as updated)
CII_CONTAINER_A: float = 1984.0
CII_CONTAINER_C: float = 0.489

# CII rating boundaries (ratio of attained to required CII)
CII_RATING_BOUNDARIES: dict[str, float] = {
    "A": 0.85,
    "B": 0.95,
    "C": 1.05,
    "D": 1.15,
    "E": float("inf"),
}


# =============================================================================
# 2. EEXI PROFILE
# =============================================================================

@dataclass(slots=True)
class EEXIProfile:
    """
    EEXI (Energy Efficiency Existing Ship Index) compliance record.

    EEXI is a one-time technical rating that constrains the maximum
    continuous rated power (MCPP) of the vessel. For the VSRP, this
    translates to an upper bound on sailing speed, which is already
    enforced by speed_levels_knots in VSRPInstance.

    Attributes
    ----------
    attained_eexi : float
        Vessel's attained EEXI [gCO2/kW·h]. Lower is better.
    required_eexi : float
        Regulatory limit for this vessel type and size [gCO2/kW·h].
    eexi_compliant : bool
        True if attained_eexi <= required_eexi.
    reference_speed_knots : float
        Speed at which EEXI is evaluated (typically design speed).
    """
    attained_eexi: float
    required_eexi: float
    eexi_compliant: bool
    reference_speed_knots: float


def check_eexi_compliance(
    *,
    attained_eexi: float,
    required_eexi: float,
    reference_speed_knots: float = DEFAULT_REFERENCE_SPEED_KNOTS,
) -> EEXIProfile:
    """
    Check EEXI compliance for a vessel.

    In the VSRP context, EEXI compliance is a vessel-level precondition,
    not a per-voyage decision. The maximum speed in speed_levels_knots
    should already respect the EEXI-implied speed limit.
    """
    return EEXIProfile(
        attained_eexi=attained_eexi,
        required_eexi=required_eexi,
        eexi_compliant=attained_eexi <= required_eexi,
        reference_speed_knots=reference_speed_knots,
    )


# =============================================================================
# 3. CII RATING
# =============================================================================

def compute_required_cii(
    dwt: float = DEFAULT_VESSEL_DWT,
    *,
    a: float = CII_CONTAINER_A,
    c: float = CII_CONTAINER_C,
) -> float:
    """
    Compute the IMO required CII for a container vessel.

    $$
    \text{Required CII} = a \cdot \text{DWT}^{-c}
    $$

    Units: gCO2 / (DWT·nm)
    """
    return a * (dwt ** (-c))


def compute_attained_cii(
    total_co2_t: float,
    total_distance_nm: float,
    dwt: float = DEFAULT_VESSEL_DWT,
) -> float:
    """
    Compute the attained CII for one voyage.

    $$
    \text{Attained CII} = \frac{\text{CO}_2 \text{ [g]}}
                               {\text{DWT [t]} \cdot \text{Distance [nm]}}
    $$

    Note: CII is formally an annual metric. Per-voyage CII is used here
    as a proxy for regulatory exposure, consistent with Hu et al. (2024).
    """
    if total_distance_nm <= 0 or dwt <= 0:
        return float("inf")
    co2_g = total_co2_t * 1e6
    return co2_g / (dwt * total_distance_nm)


def assign_cii_rating(
    attained_cii: float,
    required_cii: float,
) -> str:
    """
    Assign an IMO CII rating (A–E) based on attained vs required CII.

    Rating boundaries:
      A : attained <= 0.85 * required  (>= 15% better than required)
      B : attained <= 0.95 * required  (>= 5% better)
      C : attained <= 1.05 * required  (within ±5%)
      D : attained <= 1.15 * required  (>= 15% worse)
      E : attained >  1.15 * required
    """
    if required_cii <= 0:
        return "UNKNOWN"

    ratio = attained_cii / required_cii

    if ratio <= CII_RATING_BOUNDARIES["A"]:
        return "A"
    elif ratio <= CII_RATING_BOUNDARIES["B"]:
        return "B"
    elif ratio <= CII_RATING_BOUNDARIES["C"]:
        return "C"
    elif ratio <= CII_RATING_BOUNDARIES["D"]:
        return "D"
    return "E"


def compute_cii_rating(
    total_co2_t: float,
    total_distance_nm: float,
    *,
    dwt: float = DEFAULT_VESSEL_DWT,
) -> dict:
    """
    Compute full CII assessment for one voyage.

    Returns
    -------
    dict with keys:
      attained_cii  : float  [gCO2 / (DWT·nm)]
      required_cii  : float  [gCO2 / (DWT·nm)]
      cii_ratio     : float  attained / required
      cii_rating    : str    A / B / C / D / E
      cii_compliant : bool   rating in {A, B, C}
    """
    required = compute_required_cii(dwt)
    attained = compute_attained_cii(total_co2_t, total_distance_nm, dwt)
    rating = assign_cii_rating(attained, required)

    return {
        "attained_cii": attained,
        "required_cii": required,
        "cii_ratio": attained / required if required > 0 else float("inf"),
        "cii_rating": rating,
        "cii_compliant": rating in {"A", "B", "C"},
    }


# =============================================================================
# 4. FUELEU MARITIME
# =============================================================================

def get_fueleu_limit(year: int) -> float:
    """
    Return the FuelEU GHG intensity limit for a given year.

    Interpolates linearly between defined milestone years.
    Before 2025: no FuelEU obligation (returns VLSFO baseline).
    After 2050: returns the 2050 limit.
    """
    if year < 2025:
        return DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ

    milestones = sorted(FUELEU_GHG_LIMITS.keys())

    if year >= milestones[-1]:
        return FUELEU_GHG_LIMITS[milestones[-1]]

    for i in range(len(milestones) - 1):
        y0, y1 = milestones[i], milestones[i + 1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            return (
                FUELEU_GHG_LIMITS[y0] * (1 - t)
                + FUELEU_GHG_LIMITS[y1] * t
            )

    return DEFAULT_FUELEU_GHG_LIMIT_GCO2EQ_PER_MJ


def compute_fueleu_penalty_usd(
    total_fuel_t: float,
    ghg_intensity_gco2eq_per_mj: float,
    *,
    year: int = 2026,
    penalty_eur_per_tonne_vlsfo_equiv: float = (
        DEFAULT_FUELEU_PENALTY_EUR_PER_TONNE_VLSFO_EQUIV
    ),
    eur_to_usd_rate: float = 1.08,
    lhv_mj_per_tonne: float = VLSFO_LHV_MJ_PER_TONNE,
) -> float:
    """
    Compute FuelEU Maritime non-compliance penalty for one voyage.

    The FuelEU regulation Article 23 specifies EUR 2,400 per tonne of
    VLSFO equivalent that would need to be replaced by a compliant fuel
    to meet the GHG intensity limit.

    Step 1: Total energy content of fuel burned
        E_total [MJ] = total_fuel_t [t] * 1000 [kg/t] * lhv [MJ/kg]
                     = total_fuel_t * lhv_mj_per_tonne

    Step 2: Fraction of energy that exceeds the limit
        excess_fraction = max(0, g - g*) / g
        where g  = attained GHG intensity [gCO2eq/MJ]
              g* = FuelEU limit [gCO2eq/MJ]

    Step 3: Excess energy [MJ]
        E_excess = excess_fraction * E_total

    Step 4: VLSFO equivalent of excess energy [tonnes]
        F_excess = E_excess / lhv_mj_per_tonne

    Step 5: Penalty [EUR]
        penalty_EUR = F_excess * penalty_eur_per_tonne_vlsfo_equiv

    Step 6: Convert to USD
        penalty_USD = penalty_EUR * eur_to_usd_rate

    Under the single-fuel VLSFO assumption, GHG intensity is constant
    at 91.16 gCO2eq/MJ. For 2026, the limit is ~88.61 gCO2eq/MJ
    (interpolated between 2025=89.34 and 2030=85.69).

    Example for 205 tonnes of fuel:
        E_total       = 205,000 kg * 40.5 MJ/kg = 8,302,500 MJ
        excess_frac   = (91.16 - 88.61) / 91.16 = 0.02797
        E_excess      = 0.02797 * 8,302,500 = 232,221 MJ
        F_excess      = 232,221 / 40,500 = 5.734 tonnes VLSFO equiv
        penalty_EUR   = 5.734 * 2,400 = 13,762 EUR
        penalty_USD   = 13,762 * 1.08 = ~14,863 USD

    Parameters
    ----------
    total_fuel_t : float
        Total fuel consumed on the voyage [tonnes]
    ghg_intensity_gco2eq_per_mj : float
        Fuel GHG intensity [gCO2eq/MJ]
    year : int
        Regulatory year (determines applicable limit)
    penalty_eur_per_tonne_vlsfo_equiv : float
        EUR penalty per tonne of VLSFO equivalent above the limit
    eur_to_usd_rate : float
        EUR/USD conversion rate
    lhv_mj_per_tonne : float
        Lower heating value of VLSFO [MJ/tonne]

    Returns
    -------
    float
        FuelEU penalty in USD
    """
    if year < 2025:
        return 0.0

    if total_fuel_t <= 0:
        return 0.0

    limit = get_fueleu_limit(year)
    excess_ghg = max(0.0, ghg_intensity_gco2eq_per_mj - limit)

    if excess_ghg <= 0:
        return 0.0

    # Step 1: Total energy content
    e_total_mj = total_fuel_t * lhv_mj_per_tonne

    # Step 2: Fraction of energy exceeding the limit
    if ghg_intensity_gco2eq_per_mj <= 0:
        return 0.0
    excess_fraction = excess_ghg / ghg_intensity_gco2eq_per_mj

    # Step 3: Excess energy
    e_excess_mj = excess_fraction * e_total_mj

    # Step 4: VLSFO equivalent of excess energy
    f_excess_t = e_excess_mj / lhv_mj_per_tonne

    # Step 5 & 6: Penalty in EUR then USD
    penalty_eur = f_excess_t * penalty_eur_per_tonne_vlsfo_equiv
    return penalty_eur * eur_to_usd_rate

def compute_fueleu_penalty_per_fuel_tonne_usd(
    ghg_intensity_gco2eq_per_mj: float,
    *,
    year: int = 2026,
    penalty_eur_per_tonne_vlsfo_equiv: float = (
        DEFAULT_FUELEU_PENALTY_EUR_PER_TONNE_VLSFO_EQUIV
    ),
    eur_to_usd_rate: float = 1.08,
) -> float:
    """
    Compute FuelEU penalty in USD per tonne of fuel burned.

    Under the single-fuel VLSFO assumption, the FuelEU penalty is linear
    in total fuel consumption, so the per-voyage penalty can be written as:

    $$
    \text{Penalty USD} =
    \text{fuel}_t \cdot
    \left(
      \frac{\max(0, g - g^*)}{g}
      \cdot
      \text{penalty}_{EUR/t}
      \cdot
      \text{FX}
    \right)
    $$

    where:
    - $$g$$ is attained GHG intensity
    - $$g^*$$ is the FuelEU limit
    - $$\text{penalty}_{EUR/t}$$ is EUR 2400 per tonne VLSFO equivalent
    - $$\text{FX}$$ converts EUR to USD

    Returns
    -------
    float
        USD penalty per tonne of fuel burned.
    """
    if year < 2025:
        return 0.0

    if ghg_intensity_gco2eq_per_mj <= 0:
        return 0.0

    limit = get_fueleu_limit(year)
    excess_ghg = max(0.0, ghg_intensity_gco2eq_per_mj - limit)

    if excess_ghg <= 0:
        return 0.0

    excess_fraction = excess_ghg / ghg_intensity_gco2eq_per_mj
    penalty_eur_per_fuel_tonne = (
        excess_fraction * penalty_eur_per_tonne_vlsfo_equiv
    )
    return penalty_eur_per_fuel_tonne * eur_to_usd_rate

# =============================================================================
# 5. BASIC HELPERS
# =============================================================================

def fuel_consumption_tonnes(
    distance_nm: float,
    speed_knots: float,
    *,
    fuel_base_consumption_tpd: float = DEFAULT_FUEL_BASE_CONSUMPTION_TPD,
    reference_speed_knots: float = DEFAULT_REFERENCE_SPEED_KNOTS,
) -> float:
    """
    Compute fuel consumption for one sailing leg (cubic-speed model).

    $$
    F = F_0 \cdot \left(\frac{v}{v_0}\right)^3 \cdot \frac{d}{v \cdot 24}
    $$
    """
    if speed_knots <= 0:
        raise ValueError("speed_knots must be strictly positive")
    if distance_nm < 0:
        raise ValueError("distance_nm must be non-negative")

    travel_time_h = distance_nm / speed_knots
    daily_burn_tpd = fuel_base_consumption_tpd * (
        speed_knots / reference_speed_knots
    ) ** 3
    return daily_burn_tpd * travel_time_h / 24.0


def co2_from_fuel_tonnes(
    fuel_tonnes: float,
    *,
    co2_emission_factor_tco2_per_tfuel: float = (
        DEFAULT_CO2_EMISSION_FACTOR_TCO2_PER_TFUEL
    ),
) -> float:
    """
    Convert fuel burn to CO2 emissions.
    """
    return fuel_tonnes * co2_emission_factor_tco2_per_tfuel


def ets_cost_eur(
    co2_tonnes: float,
    *,
    year: int = 2026,
    carbon_price_eur_per_tco2: float = DEFAULT_EU_ETS_CARBON_PRICE_EUR_PER_TCO2,
    phase_in_schedule: dict[int, float] | None = None,
) -> float:
    """
    Compute EU ETS cost for a given quantity of CO2 emissions.

    Phase-in schedule (Regulation EU 2023/957):
      2024: 40% of covered emissions
      2025: 70%
      2026+: 100%
    """
    phase_in_schedule = phase_in_schedule or DEFAULT_EU_ETS_PHASE_IN
    phase_in_fraction = phase_in_schedule.get(year, 1.0)
    return co2_tonnes * phase_in_fraction * carbon_price_eur_per_tco2


def ghg_intensity_gco2eq_per_mj(
    *,
    fuel_type_intensity_gco2eq_per_mj: float = (
        DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ
    ),
) -> float:
    """
    Return fuel GHG intensity.

    Under the single-fuel (VLSFO) assumption, intensity is constant.
    This function exists as a hook for future multi-fuel extensions.
    """
    return fuel_type_intensity_gco2eq_per_mj


# =============================================================================
# 6. LEG-LEVEL ENTITY AND COMPUTATION
# =============================================================================

@dataclass(slots=True)
class LegEmissionRecord:
    """
    Detailed emissions and regulatory-cost record for one route leg.
    """
    from_port_idx: int
    to_port_idx: int
    distance_nm: float
    speed_knots: float

    fuel_tonnes: float
    co2_tonnes: float
    ets_cost_eur: float

    ghg_intensity_gco2eq_per_mj: float
    fueleu_compliant: bool
    fueleu_penalty_usd: float


def compute_leg_emission_record(
    instance: VSRPInstance,
    from_port_idx: int,
    to_port_idx: int,
    speed_knots: float,
    *,
    year: int = 2026,
    fuel_base_consumption_tpd: float = DEFAULT_FUEL_BASE_CONSUMPTION_TPD,
    reference_speed_knots: float = DEFAULT_REFERENCE_SPEED_KNOTS,
    co2_emission_factor_tco2_per_tfuel: float = (
        DEFAULT_CO2_EMISSION_FACTOR_TCO2_PER_TFUEL
    ),
    carbon_price_eur_per_tco2: float = DEFAULT_EU_ETS_CARBON_PRICE_EUR_PER_TCO2,
    phase_in_schedule: dict[int, float] | None = None,
    fuel_type_intensity_gco2eq_per_mj: float = (
        DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ
    ),
    include_fueleu_penalty: bool = True,
) -> LegEmissionRecord:
    """
    Compute full emissions record for one selected sailing leg.

    Parameters
    ----------
    include_fueleu_penalty : bool
        When False, fueleu_penalty_usd is set to 0.0 regardless of
        GHG intensity vs limit comparison.
    """
    distance_nm = instance.distance_matrix_nm[from_port_idx][to_port_idx]

    fuel_t = fuel_consumption_tonnes(
        distance_nm=distance_nm,
        speed_knots=speed_knots,
        fuel_base_consumption_tpd=fuel_base_consumption_tpd,
        reference_speed_knots=reference_speed_knots,
    )
    co2_t = co2_from_fuel_tonnes(
        fuel_tonnes=fuel_t,
        co2_emission_factor_tco2_per_tfuel=co2_emission_factor_tco2_per_tfuel,
    )
    ets_eur = ets_cost_eur(
        co2_tonnes=co2_t,
        year=year,
        carbon_price_eur_per_tco2=carbon_price_eur_per_tco2,
        phase_in_schedule=phase_in_schedule,
    )
    ghg_intensity = ghg_intensity_gco2eq_per_mj(
        fuel_type_intensity_gco2eq_per_mj=fuel_type_intensity_gco2eq_per_mj
    )
    fueleu_limit = get_fueleu_limit(year)
    fueleu_ok = ghg_intensity <= fueleu_limit

    # Compute FuelEU penalty only when flag is active
    if include_fueleu_penalty:
        fueleu_penalty = compute_fueleu_penalty_usd(
            total_fuel_t=fuel_t,
            ghg_intensity_gco2eq_per_mj=ghg_intensity,
            year=year,
        )
    else:
        fueleu_penalty = 0.0

    return LegEmissionRecord(
        from_port_idx=from_port_idx,
        to_port_idx=to_port_idx,
        distance_nm=distance_nm,
        speed_knots=speed_knots,
        fuel_tonnes=fuel_t,
        co2_tonnes=co2_t,
        ets_cost_eur=ets_eur,
        ghg_intensity_gco2eq_per_mj=ghg_intensity,
        fueleu_compliant=fueleu_ok,
        fueleu_penalty_usd=fueleu_penalty,
    )


def compute_solution_leg_emissions(
    instance: VSRPInstance,
    solution: VSRPSolution,
    *,
    year: int = 2026,
    include_fueleu_penalty: bool = True,
) -> list[LegEmissionRecord]:
    """
    Compute leg-level emissions records for the selected route.
    """
    records: list[LegEmissionRecord] = []

    for leg in solution.route_legs:
        records.append(
            compute_leg_emission_record(
                instance=instance,
                from_port_idx=leg.from_port_idx,
                to_port_idx=leg.to_port_idx,
                speed_knots=leg.speed_knots,
                year=year,
                include_fueleu_penalty=include_fueleu_penalty,
            )
        )

    return records


# =============================================================================
# 7. SOLUTION-LEVEL SUMMARY
# =============================================================================

def compute_solution_emissions_summary(
    instance: VSRPInstance,
    solution: VSRPSolution,
    *,
    year: int = 2026,
    dwt: float = DEFAULT_VESSEL_DWT,
) -> EmissionsSummary:
    """
    Compute a full solution-level emissions summary.
    """
    include_fueleu = getattr(instance, "include_fueleu_penalty", False)

    leg_records = compute_solution_leg_emissions(
        instance=instance,
        solution=solution,
        year=year,
        include_fueleu_penalty=include_fueleu,
    )

    fueleu_limit = get_fueleu_limit(year)

    if not leg_records:
        ghg_intensity = DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ
        return EmissionsSummary(
            total_fuel_t=0.0,
            total_co2_t=0.0,
            total_ets_eur=0.0,
            total_ets_usd=0.0,
            total_fueleu_penalty_usd=0.0,
            avg_ghg_gco2eq_per_mj=ghg_intensity,
            fueleu_compliant=ghg_intensity <= fueleu_limit,
            fueleu_limit_gco2eq_per_mj=fueleu_limit,
            eexi_compliant=None,
            cii_rating=None,
            attained_cii=None,
            required_cii=None,
        )

    total_fuel_t = sum(r.fuel_tonnes for r in leg_records)
    total_co2_t = sum(r.co2_tonnes for r in leg_records)
    total_ets_eur = sum(r.ets_cost_eur for r in leg_records)
    total_ets_usd = total_ets_eur * 1.08

    total_fueleu_penalty_usd = (
        sum(r.fueleu_penalty_usd for r in leg_records)
        if include_fueleu
        else 0.0
    )

    if total_fuel_t > 1e-12:
        avg_ghg = (
            sum(
                r.ghg_intensity_gco2eq_per_mj * r.fuel_tonnes
                for r in leg_records
            )
            / total_fuel_t
        )
    else:
        avg_ghg = DEFAULT_VLSFO_GHG_INTENSITY_GCO2EQ_PER_MJ

    fueleu_ok = avg_ghg <= fueleu_limit

    # CII assessment
    total_distance_nm = sum(r.distance_nm for r in leg_records)
    cii_result = compute_cii_rating(
        total_co2_t=total_co2_t,
        total_distance_nm=total_distance_nm,
        dwt=dwt,
    )

    return EmissionsSummary(
        total_fuel_t=total_fuel_t,
        total_co2_t=total_co2_t,
        total_ets_eur=total_ets_eur,
        total_ets_usd=total_ets_usd,
        total_fueleu_penalty_usd=total_fueleu_penalty_usd,
        avg_ghg_gco2eq_per_mj=avg_ghg,
        fueleu_compliant=fueleu_ok,
        fueleu_limit_gco2eq_per_mj=fueleu_limit,
        eexi_compliant=None,
        cii_rating=cii_result["cii_rating"],
        attained_cii=cii_result["attained_cii"],
        required_cii=cii_result["required_cii"],
    )


# =============================================================================
# 8. REPORTING HELPERS
# =============================================================================

def leg_emissions_to_dataframe(
    instance: VSRPInstance,
    leg_records: list[LegEmissionRecord],
) -> pd.DataFrame:
    """
    Convert leg-level emissions records to a tidy DataFrame.
    """
    rows = []
    for r in leg_records:
        rows.append({
            "from_port_idx": r.from_port_idx,
            "from_port": instance.port_name(r.from_port_idx),
            "to_port_idx": r.to_port_idx,
            "to_port": instance.port_name(r.to_port_idx),
            "distance_nm": r.distance_nm,
            "speed_knots": r.speed_knots,
            "fuel_tonnes": round(r.fuel_tonnes, 4),
            "co2_tonnes": round(r.co2_tonnes, 4),
            "ets_cost_eur": round(r.ets_cost_eur, 4),
            "ghg_intensity_gco2eq_per_mj": round(
                r.ghg_intensity_gco2eq_per_mj, 4
            ),
            "fueleu_compliant": r.fueleu_compliant,
            "fueleu_penalty_usd": round(r.fueleu_penalty_usd, 4),
        })

    return pd.DataFrame(rows)


def summarize_emissions_to_dict(summary: EmissionsSummary) -> dict:
    """
    Convert EmissionsSummary into a plain dictionary for reporting.
    """
    return {
        "total_fuel_t": summary.total_fuel_t,
        "total_co2_t": summary.total_co2_t,
        "total_ets_eur": summary.total_ets_eur,
        "total_ets_usd": summary.total_ets_usd,
        "total_fueleu_penalty_usd": summary.total_fueleu_penalty_usd,
        "avg_ghg_gco2eq_per_mj": summary.avg_ghg_gco2eq_per_mj,
        "fueleu_compliant": summary.fueleu_compliant,
        "fueleu_limit_gco2eq_per_mj": summary.fueleu_limit_gco2eq_per_mj,
        "eexi_compliant": summary.eexi_compliant,
        "cii_rating": summary.cii_rating,
        "attained_cii": summary.attained_cii,
        "required_cii": summary.required_cii,
    }