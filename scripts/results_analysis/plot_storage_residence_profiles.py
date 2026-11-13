"""Aggregate LIFO-inferred residence-time profiles across model cases.

Run from the project root (next to _plots.py)::

    pixi run python -m scripts.results_analysis.plot_storage_residence_profiles --formulation FX
    pixi run python -m scripts.results_analysis.plot_storage_residence_profiles --formulation FX --country NLD
    pixi run python -m scripts.results_analysis.plot_storage_residence_profiles --formulation FX1 --country NLD

--formulation FX matches FX1, FX2, FX4, ...; an exact code such as FX1
matches that experiment alone. All matching .nc cases in --input-dir have
EQUAL weight; unlike throughput pooling, each eligible case is normalised by
its own attributable discharged energy before averaging residence-bin shares.

Technologies not installed in any case are omitted from the FIGURE (retained
in detailed CSV only when their family is defined). The Selected column counts
cases with installed capacity, including selected-but-idle stores; the residence
profile uses only cases with attributable known-age discharge. Mean cycles/yr
is conditional on installed capacity, including idle installed cases.

Battery 2h/4h/8h, CAES, SC, SA, DGF and LRC are considered; --include-hydro
additionally includes pre-existing hydrostorage as an explicitly FIXED
benchmark (not a new investment). Hydro requires a compatible storage SoC and
flow-in/flow-out energy balance; incompatible hydro is warned and omitted.

Residence times follow the original LIFO ledger: charge before withdrawal
within a timestep, energy-flow efficiencies/losses included, cyclic dispatch
replayed to resolve prehistory. This is an accounting convention, NOT physical
mixing of hydrogen, electricity or water. Remaining unknown-age withdrawn
energy is omitted from residence bin shares and audited. For each case,
individual physical technology variants are evaluated before family grouping.

The detailed CSVs and summary CSV are always exported (unlike the source
script, where CSV writes were commented out). PDF by default; optional SVG.

The companion plot_storage_operations_case.py reuses these same LIFO methods
for a country/weather-specific SoC time series and LIFO age CDF.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

try:
    from ._plots import model_case, storage_family
except ImportError:
    from _plots import model_case, storage_family

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "capacity_expansion_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "residence_profiles"
MM = 1/25.4
HOURS_PER_YEAR = 24 * 365.2425
FONT_SIZE = 9
PLASMA = plt.get_cmap("plasma")
FAMILIES = (
    "battery_2h", "battery_4h", "battery_8h", "compressed_air_storage",
    "salt_cavern", "aquifer", "depleted_gas_field", "lined_rock_cavern",
    "hydrostorage",
)
LABELS = {
    "battery_2h": "Battery 2h", "battery_4h": "Battery 4h",
    "battery_8h": "Battery 8h", "compressed_air_storage": "CAES",
    "salt_cavern": "SC", "aquifer": "SA", "depleted_gas_field": "DGF",
    "lined_rock_cavern": "LRC", "hydrostorage": "Hydro (existing)",
}
HYDRO_IDS = {
    "hydrostorage", "hydro_storage", "hydro_reservoir", "hydro_reservoir_storage",
    "reservoir_hydro", "pumped_hydro", "pumped_hydro_storage", "pumped_storage",
}
# Bins 0--6 h, 6--24 h, 1--7 d, 7--30 d, 30--90 d, 90--365 d, >365 d.
DEFAULT_RESIDENCE_BINS_DAYS = (1, 7, 30, 90, 365, 730)

@dataclass
class Store:
    tech: str
    family: str
    capacity: float  # MWh or the consistent model energy unit
    time: pd.DatetimeIndex
    soc: np.ndarray  # End-of-step stored energy
    charge: np.ndarray  # Energy entering the inventory, after efficiency
    discharge: np.ndarray  # Energy removed from inventory, before efficiency
    loss: np.ndarray  # Fractional leakage per hour
    dt: np.ndarray  # Timestep resolution in hours
    cyclic: bool
    storage_initial: float
    injection_cap: float = float("nan")  # MW
    withdrawal_cap: float = float("nan")  # MW
    design_epr: float = float("nan")  # Fixed design EPR, hours
    design_wtir: float = float("nan")  # Fixed design WTIR, dimensionless


@dataclass
class LifoResult:
    ages_hours: np.ndarray
    discharged_energy: np.ndarray
    unattributed_energy: float
    total_discharge: float
    max_balance_error: float
    replayed_periods: int = 1
    cyclic_age_converged: bool = False
    unknown_inventory_at_end: float = 0.0


def _family(tech: str, *, include_hydro: bool = False) -> str | None:
    identity = tech.lower()
    if re.fullmatch(r"battery_(?:2|4|8)h", identity):
        return identity
    if identity in ("compressed_air_storage", "caes", "compressed_air"):
        return "compressed_air_storage"
    if include_hydro and identity in HYDRO_IDS:
        return "hydrostorage"
    family = storage_family(identity)
    return family if family in FAMILIES else None


def _sel_tech(array, tech: str):
    if "techs" in array.dims:
        array = array.sel(techs=tech)
    if "nodes" in array.dims and array.sizes["nodes"] > 1:
        raise ValueError(f"{tech}: multiple nodes; process physical nodes separately")
    return array


def _scalar(array, tech: str) -> float:
    array = _sel_tech(array, tech)
    data = np.asarray(array.values, dtype=float).ravel()
    finite = data[np.isfinite(data)]
    if not len(finite):
        return float("nan")
    if len(finite) > 1 and not np.allclose(finite, finite[0]):
        raise ValueError(f"{tech}: expected one installed scalar, got {finite}")
    return float(finite[0])


def _time_parameter(inputs, name: str, tech: str, n: int, default: float) -> np.ndarray:
    if name not in inputs:
        return np.full(n, default)
    array = _sel_tech(inputs[name], tech)
    if "timesteps" in array.dims:
        other = [d for d in array.dims if d != "timesteps"]
        if other:
            array = array.squeeze(drop=True)
        vals = np.asarray(array.transpose("timesteps").values, dtype=float)
        if len(vals) != n:
            raise ValueError(f"{name}: unexpected number of timesteps")
    else:
        value = _scalar(array, tech)
        vals = np.full(n, value)
    return np.where(np.isfinite(vals), vals, default)


def _series(array, tech: str, *, sum_other: bool = False) -> np.ndarray:
    array = _sel_tech(array, tech)
    if "timesteps" not in array.dims:
        raise ValueError(f"{tech}: {array.name} lacks timesteps")
    if not sum_other:
        array = array.squeeze(drop=True)  # A single-country case has one node.
    other = [d for d in array.dims if d != "timesteps"]
    if other:
        if not sum_other:
            raise ValueError(f"{tech}: multiple independent dimensions in {array.name}")
        array = array.fillna(0).sum(dim=other)
    return np.asarray(array.transpose("timesteps").values, dtype=float)


def _inventory_flow(results, inputs, tech: str, direction: str) -> np.ndarray:
    # Some Calliope exports expose the already efficiency-adjusted expression.
    adjusted = f"flow_{direction}_inc_eff"
    if adjusted in results:
        return _series(results[adjusted], tech, sum_other=True)
    key = f"flow_{direction}"
    if key not in results:
        raise ValueError(f"Missing {key}; LIFO residence time requires both flows")
    flow = _sel_tech(results[key], tech).fillna(0)
    efficiency_key = f"flow_{direction}_eff"
    if efficiency_key in inputs:
        eff = _sel_tech(inputs[efficiency_key], tech).fillna(1)
        if bool(((eff <= 0) & (flow > 0)).any()):
            raise ValueError(f"Non-positive {efficiency_key} with positive flow for {tech}")
        flow = flow * eff if direction == "in" else flow / eff
    other = [d for d in flow.dims if d != "timesteps"]
    if other:
        flow = flow.sum(dim=other)
    return np.asarray(flow.transpose("timesteps").values, dtype=float)


def _installed_cap(results, tech: str, names: tuple[str, ...]) -> float:
    for name in names:
        if name in results:
            return _scalar(results[name], tech)
    return float("nan")


def _design_parameter(inputs, results, tech: str, names: tuple[str, ...]) -> float:
    for dataset in (inputs, results):
        for name in names:
            if name in dataset:
                value = _scalar(dataset[name], tech)
                if math.isfinite(value) and value > 0:
                    return value
    return float("nan")


def _load_store(results, inputs, tech: str, family: str) -> Store | None:
    cap = _scalar(results["storage_cap"], tech)
    if not math.isfinite(cap) or cap <= 1e-7:
        return None
    time = pd.DatetimeIndex(pd.to_datetime(results.coords["timesteps"].values))
    if not time.is_monotonic_increasing or time.has_duplicates:
        raise ValueError("LIFO needs strictly increasing chronological timesteps")
    n = len(time)
    soc = _series(results["storage"], tech)
    charge = _inventory_flow(results, inputs, tech, "in")
    discharge = _inventory_flow(results, inputs, tech, "out")
    if any(len(v) != n for v in (soc, charge, discharge)):
        raise ValueError(f"{tech}: inconsistent timestep lengths")
    if not all(np.isfinite(v).all() for v in (soc, charge, discharge)):
        raise ValueError(f"{tech}: NaN/inf in storage operation")
    if min(float(soc.min()), float(charge.min()), float(discharge.min())) < -max(1e-5, 1e-7 * cap):
        raise ValueError(f"{tech}: materially negative storage or flows")
    soc, charge, discharge = (np.maximum(v, 0) for v in (soc, charge, discharge))
    loss = _time_parameter(inputs, "storage_loss", tech, n, 0.0)
    if (loss < 0).any() or (loss >= 1).any():
        raise ValueError(f"{tech}: storage loss must lie in [0, 1)")
    if "timestep_resolution" in inputs:
        dt = _time_parameter(inputs, "timestep_resolution", tech, n, 1.0)
    elif n > 1:
        differences = np.diff(time.asi8) / 3.6e12
        dt = np.r_[differences, differences[-1]]
    else:
        dt = np.array([1.0])
    if (dt <= 0).any() or not np.isfinite(dt).all():
        raise ValueError("All timestep resolutions must be positive and finite")
    cyclic = bool(_time_parameter(inputs, "cyclic_storage", tech, n, 1.0)[0])
    storage_initial = _time_parameter(inputs, "storage_initial", tech, n, 0.0)[0]
    return Store(
        tech, family, cap, time, soc, charge, discharge, loss, dt,
        cyclic, storage_initial,
        _installed_cap(results, tech, ("flow_in_cap", "flow_cap_in")),
        _installed_cap(results, tech, ("flow_out_cap", "flow_cap_out")),
        _design_parameter(inputs, results, tech, ("EPR", "epr", "EPR_hours")),
        _design_parameter(inputs, results, tech, ("WTIR", "wtir")),
    )


def load_stores(path: Path, *, allow_empty: bool = False,
                include_hydro: bool = False) -> list[Store]:
    import xarray as xr

    with xr.open_dataset(path, group="results") as results:
        try:
            inputs_context = xr.open_dataset(path, group="inputs")
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read the 'inputs' NetCDF group: {exc}") from exc
        with inputs_context as inputs:
            required = ("storage", "storage_cap", "flow_in", "flow_out")
            missing = [name for name in required if name not in results]
            if missing:
                raise ValueError(f"Missing required Calliope results: {missing}")
            stores = []
            for tech in map(str, results.coords["techs"].values):
                family = _family(tech, include_hydro=include_hydro)
                if family:
                    try:
                        store = _load_store(results, inputs, tech, family)
                    except (ValueError, KeyError) as exc:
                        # A fixed hydro model might not expose a two-sided
                        # energy store; do not manufacture LIFO residence ages.
                        if family != "hydrostorage":
                            raise
                        warnings.warn(f"{path.name}: skipping hydro {tech}: {exc}",
                                      stacklevel=2)
                        continue
                    if store is not None:
                        stores.append(store)
    if not stores and not allow_empty:
        raise ValueError("No installed compatible storage capacity in this case")
    return stores


def lifo_residence(store: Store, *, max_cycles: int = 8,
                   unknown_tolerance: float = 1e-6) -> LifoResult:
    """Run a LIFO ledger, repeating cyclic dispatch to resolve boundary ages.

    The returned statistics use ONLY the final simulated period. A previous
    period's withdrawals are never counted twice. For noncyclic technologies,
    only the actual modeled period is evaluated.
    """
    if max_cycles < 1:
        raise ValueError("max_cycles must be at least one")
    if not 0 <= unknown_tolerance < 1:
        raise ValueError("unknown_tolerance must be in [0, 1)")
    # Inventories are tracked in a lazily scaled unit so leakage costs O(1)
    # per timestep, rather than looping over every historic injection cohort.
    initial = float(store.soc[-1]) if store.cyclic else store.storage_initial * store.capacity
    stack: list[tuple[float | None, float]] = [(None, initial)] if initial > 0 else []
    scale, total_base = 1.0, initial
    worst_error = 0.0
    tol = max(1e-2, 1e-6 * store.capacity)
    n = len(store.time)
    offsets = np.r_[0.0, np.cumsum(store.dt[:-1])]
    period_hours = float(store.dt.sum())
    previous_result: LifoResult | None = None

    for cycle in range(max_cycles if store.cyclic else 1):
        ages, energies = [], []
        unknown, total_out = 0.0, 0.0
        for i in range(n):
            if cycle > 0 or i > 0 or store.cyclic:
                previous = (i - 1) % n
                scale *= (1 - store.loss[previous]) ** store.dt[previous]
            if scale < 1e-80:
                stack = [(date, amount * scale) for date, amount in stack]
                total_base *= scale
                scale = 1.0

            time_hours = cycle * period_hours + offsets[i]
            charge_base = store.charge[i] / scale
            if charge_base > 0:
                stack.append((time_hours, charge_base))
                total_base += charge_base
            withdraw_base = store.discharge[i] / scale
            total_out += store.discharge[i]
            if withdraw_base > total_base + tol / scale:
                raise ValueError(
                    f"{store.tech}: withdrawal exceeds LIFO inventory "
                    f"at period {cycle + 1}, step {i}"
                )
            tiny = max(1e-10, 1e-12 * store.capacity) / scale
            while withdraw_base > tiny and stack:
                date, amount = stack.pop()
                taken = min(amount, withdraw_base)
                amount -= taken
                withdraw_base -= taken
                total_base -= taken
                if amount > tiny:
                    stack.append((date, amount))
                else:
                    total_base -= amount  # Discard only floating-point dust.
                if date is None:
                    unknown += taken * scale
                else:
                    ages.append(max(time_hours - date, 0.0))
                    energies.append(taken * scale)
            if withdraw_base > tol / scale:
                raise ValueError(
                    f"{store.tech}: insufficient stock at period {cycle + 1}, step {i}"
                )
            actual = total_base * scale
            error = abs(actual - store.soc[i])
            worst_error = max(worst_error, error)
            if error > tol:
                raise ValueError(
                    f"{store.tech}: inventory residual {error:.4g} at "
                    f"period {cycle + 1}, timestep {i} (tolerance {tol:.4g}). "
                    "Check flow efficiencies, loss and custom storage math."
                )

        initial_stock_remaining = sum(amount for date, amount in stack if date is None) * scale
        result = LifoResult(
            np.asarray(ages, float), np.asarray(energies, float),
            unknown, total_out, worst_error, cycle + 1,
            False, initial_stock_remaining,
        )
        if not store.cyclic:
            return result
        unknown_share = unknown / total_out if total_out > 0 else 0.0
        if previous_result is not None:
            previous_share = (previous_result.unattributed_energy / previous_result.total_discharge
                              if previous_result.total_discharge > 0 else 0.0)
            if unknown_share <= unknown_tolerance and previous_share <= unknown_tolerance:
                if _lifo_distributions_stable(previous_result, result, float(store.dt.min())):
                    result.cyclic_age_converged = True
                    return result
        previous_result = result

    warnings.warn(
        f"{store.tech}: cyclic LIFO ages did not converge after {max_cycles} "
        f"repeated periods; remaining unknown-age discharge is "
        f"{(previous_result.unattributed_energy / previous_result.total_discharge):.2%}"
        if previous_result.total_discharge > 0 else
        f"{store.tech}: cyclic LIFO ages did not converge after {max_cycles} periods",
        stacklevel=2,
    )
    return previous_result


def _lifo_distributions_stable(a: LifoResult, b: LifoResult, min_step_hours: float) -> bool:
    """Require the entire energy-weighted CDF and its mean to stabilize."""
    a_total, b_total = a.discharged_energy.sum(), b.discharged_energy.sum()
    if a_total <= 0 or b_total <= 0:
        return bool(a_total <= 0 and b_total <= 0)

    # Quantize only sub-timestep numerical noise before comparing step CDFs.
    # This avoids treating near-identical floating timestamps as different ages.
    resolution = max(1e-8, min_step_hours * 1e-6)
    a_ages = np.rint(a.ages_hours / resolution).astype(np.int64)
    b_ages = np.rint(b.ages_hours / resolution).astype(np.int64)
    a_order, b_order = np.argsort(a_ages), np.argsort(b_ages)
    a_ages, b_ages = a_ages[a_order], b_ages[b_order]
    a_cdf = np.cumsum(a.discharged_energy[a_order]) / a_total
    b_cdf = np.cumsum(b.discharged_energy[b_order]) / b_total
    grid = np.union1d(a_ages, b_ages)
    a_idx = np.searchsorted(a_ages, grid, side="right") - 1
    b_idx = np.searchsorted(b_ages, grid, side="right") - 1
    a_y = np.where(a_idx >= 0, a_cdf[np.maximum(a_idx, 0)], 0.0)
    b_y = np.where(b_idx >= 0, b_cdf[np.maximum(b_idx, 0)], 0.0)
    if np.max(np.abs(a_y - b_y)) > 1e-4:
        return False

    a_mean = float(np.average(a.ages_hours, weights=a.discharged_energy))
    b_mean = float(np.average(b.ages_hours, weights=b.discharged_energy))
    return math.isclose(a_mean, b_mean, rel_tol=1e-6,
                        abs_tol=max(1e-6, min_step_hours * 1e-3))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not len(values) or weights.sum() <= 0:
        return float("nan")
    indices = np.argsort(values)
    cumsum = np.cumsum(weights[indices])
    return float(values[indices[np.searchsorted(cumsum, quantile * cumsum[-1])]])


def _read_design_csv(path: Path | None, case: dict) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    design = pd.read_csv(path)
    if "case" in design:
        design = design.loc[design["case"].astype(str) == str(case["case"])].copy()
    elif {"country", "experiment", "weather"}.issubset(design.columns):
        for name in ("country", "experiment", "weather"):
            design = design.loc[
                design[name].astype(str).str.lower() == str(case[name]).lower()
            ]
    else:
        raise ValueError("Design CSV must contain 'case' or country/experiment/weather")
    if "technology" not in design:
        raise ValueError("Design CSV needs a technology column")
    design["family"] = design["technology"].astype(str).map(_family)
    return design.loc[design["family"].isin(FAMILIES)]


def _design_metrics(family: str, group: list[Store], design: pd.DataFrame) -> tuple[float, float]:
    rows = design.loc[design["family"] == family] if not design.empty else design
    if not rows.empty:
        # Prefer the physical component capacities; ratio averages are invalid.
        if {"storage_MWh", "withdrawal_MW", "injection_MW"}.issubset(rows.columns):
            energy = pd.to_numeric(rows["storage_MWh"], errors="coerce").sum()
            out = pd.to_numeric(rows["withdrawal_MW"], errors="coerce").sum()
            power_in = pd.to_numeric(rows["injection_MW"], errors="coerce").sum()
            if energy > 0 and out > 0 and power_in > 0:
                return energy / out, out / power_in
        if len(rows) == 1 and {"EPR_hours", "WTIR"}.issubset(rows.columns):
            return float(rows.iloc[0]["EPR_hours"]), float(rows.iloc[0]["WTIR"])
    # Parameter EPR/WTIR may encode a fixed design even when separate
    # directional capacity variables are not exported by Calliope.
    capacity = sum(s.capacity for s in group)
    out = [s.withdrawal_cap if math.isfinite(s.withdrawal_cap) and s.withdrawal_cap > 0
           else s.capacity / s.design_epr if math.isfinite(s.design_epr) and s.design_epr > 0
           else float("nan") for s in group]
    power_in = [s.injection_cap if math.isfinite(s.injection_cap) and s.injection_cap > 0
                else out[i] / s.design_wtir
                if math.isfinite(s.design_wtir) and s.design_wtir > 0
                else float("nan") for i, s in enumerate(group)]
    if not all(math.isfinite(v) and v > 0 for v in [*out, *power_in]):
        return float("nan"), float("nan")
    return capacity / sum(out), sum(out) / sum(power_in)


def summarise(stores: list[Store], case: dict, design: pd.DataFrame, *,
              lifo_max_cycles: int = 8):
    families = [f for f in FAMILIES if any(s.family == f for s in stores)]
    fixed_palette = PLASMA(np.linspace(0.08, 0.91, len(FAMILIES) - 1))
    colours = {fam: fixed_palette[FAMILIES.index(fam)] for fam in families
               if fam != "hydrostorage"}
    if "hydrostorage" in families:
        colours["hydrostorage"] = "0.40"
    grouped, summaries = {}, []
    for fam in families:
        members = [s for s in stores if s.family == fam]
        results = [lifo_residence(s, max_cycles=lifo_max_cycles) for s in members]
        cyclic_results = [r for s, r in zip(members, results) if s.cyclic]
        ages = np.concatenate([r.ages_hours for r in results])
        energy = np.concatenate([r.discharged_energy for r in results])
        cap = sum(s.capacity for s in members)
        attributable = float(energy.sum())
        all_discharge = sum(r.total_discharge for r in results)
        unknown = sum(r.unattributed_energy for r in results)
        epr, wtir = _design_metrics(fam, members, design)
        mean_term = float(np.average(ages, weights=energy)) if attributable > 0 else float("nan")
        grouped[fam] = (members, ages, energy)
        summaries.append({
            "country": case["country"], "experiment": case["experiment"],
            "weather": case["weather"], "technology": fam,
            "variant_count": len(members), "installed_storage_capacity": cap,
            "discharged_energy": all_discharge,
            "attributed_discharge": attributable,
            "unknown_initial_discharge": unknown,
            "unknown_fraction_of_discharge": unknown / all_discharge if all_discharge > 0 else 0,
            "unknown_initial_inventory_at_end": sum(r.unknown_inventory_at_end for r in results),
            "lifo_replayed_periods": max(r.replayed_periods for r in results),
            "lifo_cyclic_age_converged": (
                all(r.cyclic_age_converged for r in cyclic_results)
                if cyclic_results else pd.NA
            ),
            "attributed_capacity_turnovers": attributable / cap,
            "mean_term_hours": mean_term,
            "median_term_hours": weighted_quantile(ages, energy, 0.5),
            "p90_term_hours": weighted_quantile(ages, energy, 0.9),
            "design_EPR_hours": epr, "design_WTIR": wtir,
            "mean_term_over_EPR": mean_term / epr if math.isfinite(epr) and epr > 0 else float("nan"),
            "max_inventory_balance_error": max(r.max_balance_error for r in results),
        })
    return grouped, pd.DataFrame(summaries), colours


def _residence_bin_keys(bins_days: tuple[float, ...]) -> list[str]:
    def number(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    bounds = (0.0, *bins_days)
    return ([f"share_{number(a)}_to_{number(b)}d"
             for a, b in zip(bounds[:-1], bounds[1:])]
            + [f"share_over_{number(bounds[-1])}d"])


def _residence_bin_labels(bins_days: tuple[float, ...]) -> list[str]:
    def number(value: float) -> str:
        return f"{value:g}"

    bounds = (0.0, *bins_days)
    def label(a: float, b: float) -> str:
        if b <= 1:
            return f"{24*a:g}–{24*b:g} h"
        return f"{a:g}–{b:g} d"
    return ([label(a, b) for a,b in zip(bounds[:-1],bounds[1:])]
            + [f">{number(bounds[-1])} d"])


def residence_bin_shares(ages_hours: np.ndarray, energies: np.ndarray,
                         bins_days: tuple[float, ...]) -> np.ndarray:
    """Discharged-energy shares by LIFO age (one scenario, one family)."""
    if len(ages_hours) != len(energies):
        raise ValueError("Residence ages and energies have different lengths")
    if not len(energies) or float(energies.sum()) <= 0:
        return np.full(len(bins_days) + 1, np.nan)
    edges = np.r_[0.0, bins_days, np.inf]
    hist, _ = np.histogram(ages_hours / 24, bins=edges, weights=energies)
    total = float(energies.sum())
    if not math.isclose(float(hist.sum()), total, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError("Residence-time binning omitted discharged energy")
    return hist / total


def summarise_weather_profiles(cases: list[tuple[Path, dict]], *,
                               design_csv: Path | None,
                               lifo_max_cycles: int,
                               bins_days: tuple[float, ...],
                               include_hydro: bool = False,
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One aggregate profile per family; each weather case carries equal weight.

    The denominator of Selected is *all* matching weather cases. A family is
    selected if it has positive installed storage capacity. Residence shares
    are conditional on having attributable discharged energy; cycles per year
    are averaged over installed cases, including installed but idle cases.
    """
    keys = _residence_bin_keys(bins_days)
    detail_rows = []
    for path, case in cases:
        print(f"Reading {path.name}")
        stores = load_stores(path, allow_empty=True, include_hydro=include_hydro)
        if stores:
            design = _read_design_csv(design_csv, case)
            grouped, summary, _ = summarise(
                stores, case, design, lifo_max_cycles=lifo_max_cycles,
            )
            summary = summary.set_index("technology")
        else:
            grouped, summary = {}, pd.DataFrame()
        for family in FAMILIES:
            if family == "hydrostorage" and not include_hydro:
                continue
            row = {
                "country": case["country"], "experiment": case["experiment"],
                "weather": case["weather"], "technology": family,
                "selected": family in grouped, "has_residence_profile": False,
                "installed_storage_capacity": 0.0, "discharged_energy": 0.0,
                "attributed_discharge": 0.0,
                "known_age_discharge_fraction": float("nan"),
                "equivalent_full_discharge_cycles_per_year": float("nan"),
                "lifo_cyclic_age_converged": pd.NA,
            }
            shares = np.full(len(keys), np.nan)
            if family in grouped:
                members, ages, energy = grouped[family]
                family_result = summary.loc[family]
                capacity = float(family_result["installed_storage_capacity"])
                discharged = float(family_result["discharged_energy"])
                known = float(family_result["attributed_discharge"])
                # All variants represent the same modelled weather duration.
                durations = [float(s.dt.sum()) for s in members]
                if not np.allclose(durations, durations[0]):
                    raise ValueError(f"{family}: variants have different model durations")
                years = durations[0] / HOURS_PER_YEAR
                if years <= 0 or capacity <= 0:
                    raise ValueError(f"{family}: invalid duration or installed capacity")
                shares = residence_bin_shares(ages, energy, bins_days)
                row.update({
                    "has_residence_profile": bool(known > 0),
                    "installed_storage_capacity": capacity,
                    "discharged_energy": discharged,
                    "attributed_discharge": known,
                    "known_age_discharge_fraction": (
                        known / discharged if discharged > 0 else float("nan")
                    ),
                    "equivalent_full_discharge_cycles_per_year": (
                        discharged / capacity / years
                    ),
                    "lifo_cyclic_age_converged": family_result["lifo_cyclic_age_converged"],
                })
            row.update(zip(keys, shares))
            detail_rows.append(row)

    detail = pd.DataFrame(detail_rows)
    aggregates = []
    for family in FAMILIES:
        if family == "hydrostorage" and not include_hydro:
            continue
        members = detail.loc[detail["technology"] == family]
        selected = members.loc[members["selected"]]
        profiled = members.loc[members["has_residence_profile"]]
        if len(selected) and len(selected) != len(profiled):
            warnings.warn(
                f"{LABELS[family]}: selected in {len(selected)} weather cases, "
                f"but has attributable LIFO throughput in only {len(profiled)}. "
                "The residence profile uses the attributable cases; cycle "
                "averages include all selected cases.", stacklevel=2,
            )
        agg = {
            "technology": family,
            "selected_scenarios": len(selected),
            "total_weather_scenarios": len(members),
            "profile_scenarios": len(profiled),
            "mean_equivalent_full_discharge_cycles_per_year": (
                float(selected["equivalent_full_discharge_cycles_per_year"].mean())
                if len(selected) else float("nan")
            ),
            "mean_installed_storage_capacity_when_selected": (
                float(selected["installed_storage_capacity"].mean())
                if len(selected) else float("nan")
            ),
        }
        for key in keys:
            # Equal weight per weather case, after each case's own throughput
            # normalisation; not a capacity- or discharged-energy-weighted pool.
            agg[key] = float(profiled[key].mean()) if len(profiled) else float("nan")
        aggregates.append(agg)
    return pd.DataFrame(aggregates), detail


def plot_weather_profiles(aggregate: pd.DataFrame, bins_days: tuple[float, ...],
                          path: Path, *, svg: bool = False) -> None:
    """Compact Elsevier two-column residence composition figure, installed only."""
    selected = aggregate.loc[aggregate["selected_scenarios"] > 0].copy()
    if selected.empty:
        raise ValueError("No technologies installed in the selected model results")
    keys = _residence_bin_keys(bins_days)
    labels = _residence_bin_labels(bins_days)
    colours = PLASMA(np.linspace(0.08, 0.96, len(keys)))
    n = len(selected)
    # Height adapts to number of installed families; journal figure stays compact.
    height_mm = max(74, 43 + n * 9.8)
    with plt.rc_context({
        "font.size": FONT_SIZE, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    }):
        fig, ax = plt.subplots(figsize=(190*MM, height_mm*MM))
        fig.subplots_adjust(left=0.125, right=0.85, top=0.99, bottom=0.22)
        y_coords = np.arange(n)[::-1]
        transform = blended_transform_factory(ax.transAxes, ax.transData)
        for y, (_, row) in zip(y_coords, selected.iterrows()):
            shares = row[keys].to_numpy(dtype=float)
            if np.isfinite(shares).all():
                left = 0.0
                for share, colour in zip(shares, colours):
                    if share > 0:
                        ax.barh(y, share, left=left, height=0.66, color=colour,
                                linewidth=0.25, edgecolor="white", zorder=3)
                    left += share
            else:
                ax.text(0.5, y, "No attributable discharge", color="0.5",
                        fontsize=8, ha="center", va="center")
            
            cycles = row["mean_equivalent_full_discharge_cycles_per_year"]
            ax.text(1.1, y, f"{cycles:.2f}" if np.isfinite(cycles) else "—",
                    transform=transform, ha="center", va="center", clip_on=False)
        header = y_coords[0]+0.8
        
        ax.text(1.1, header, "Cycles/yr", transform=transform,
                ha="center", va="center", fontsize=8, clip_on=False)
        ax.set_yticks(y_coords, [LABELS[f] for f in selected["technology"]])
        ax.tick_params(axis="y", length=0, pad=8, labelsize=9)
        ax.set_ylim(-0.65, header+0.40)
        ax.set_xlim(0, 1)
        ax.set_xticks((0, .25, .50, .75, 1))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Share of withdrawn energy", fontsize=9)
        ax.grid(axis="x", color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["left","top","right"]].set_visible(False)
        ax.spines["bottom"].set_color("0.55")
        handles = [Patch(facecolor=c, label=l) for c,l in zip(colours,labels)]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.49, 0.035), ncol=max(4, len(handles)),
                   frameon=False, fontsize=8.5, columnspacing=1.25,
                   handlelength=1.25, handletextpad=0.4, labelspacing=0.7)
        fig.savefig(path)
        if svg:
            fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)


def _weather_match(requested: str | None, observed: str) -> bool:
    if requested is None:
        return True
    requested, observed = requested.lower(), observed.lower()
    if requested == observed or requested == observed.removeprefix("weather_"):
        return True
    match = re.fullmatch(r"(\d{4})-(\d{4})", requested)
    if match:
        start, end = map(int, match.groups())
        return observed == f"weather_{start}_{end - start}y"
    if re.fullmatch(r"\d{4}", requested):
        return bool(re.fullmatch(rf"weather_{requested}_\d+y", observed))
    return observed == f"weather_{requested}"


def matching_cases(input_dir: Path, country: str | None, experiment: str | None,
                   weather: str | None, *, list_only: bool = False):
    cases = []
    for path in sorted(input_dir.expanduser().glob("*.nc")):
        if path.name.endswith(".pending.nc"):
            continue
        try:
            case = model_case(path)
            if case:
                cases.append((path, case))
        except Exception:
            continue
    if not cases:
        raise ValueError(f"No model result cases found in {input_dir}")
    if list_only:
        for path, case in cases:
            print(case["country"], case["experiment"], case["weather"], path.name)
        return []
    matches = [
        (path, case) for path, case in cases
        if (country is None or case["country"].upper() == country.upper())
        and (experiment is None or case["experiment"].upper() == experiment.upper())
        and _weather_match(weather, case["weather"])
    ]
    return matches


def select_case(input_dir: Path, country: str | None, experiment: str | None,
                weather: str | None, list_only: bool):
    matches = matching_cases(input_dir, country, experiment, weather,
                             list_only=list_only)
    if list_only:
        return None
    if len(matches) != 1:
        sample = ", ".join(f"{c['country']}/{c['experiment']}/{c['weather']}"
                           for _, c in matches[:10])
        raise ValueError(f"Selection matched {len(matches)} cases (expected one): {sample}. "
                         "Use --list-cases or refine --country/--experiment/--weather")
    return matches[0]


def parse_residence_bins(spec: str) -> tuple[float, ...]:
    try:
        values = tuple(float(s.strip()) for s in spec.split(","))
    except ValueError as exc:
        raise ValueError("Residence bins must be comma-separated numbers of days") from exc
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Residence-bin boundaries must be finite and positive")
    if not all(b > a for a, b in zip(values, values[1:])):
        raise ValueError("Residence-bin boundaries must be strictly increasing")
    return values


def _matches_formulation(experiment: str, formulation: str) -> bool:
    """FX -> FX1/FX2/FX4; FX1 -> FX1 only. F never includes FX."""
    exp, requested = experiment.upper(), formulation.upper()
    if requested in {"B", "P", "F", "FX"}:
        return re.fullmatch(re.escape(requested) + r"\d+[A-Z]?", exp) is not None
    return exp == requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--formulation", default="FX",
                        help="Family FX/B/P/F, or exact experiment FX1/FX2 etc (default FX)")
    parser.add_argument("--country", help="Optional ISO3 country filter, e.g. NLD")
    parser.add_argument("--weather", help="Optional sample filter e.g. 2018-2021")
    parser.add_argument("--horizon", help="Optional horizon e.g. 3y")
    parser.add_argument("--residence-bins-days", default=",".join(map(str,DEFAULT_RESIDENCE_BINS_DAYS)),
                        help="Comma-separated residence bin upper boundaries in days")
    parser.add_argument("--include-hydro", action="store_true",
                        help="Show installed FIXED hydrostorage as a benchmark, if SoC/flows supported")
    parser.add_argument("--lifo-max-cycles", type=int, default=8)
    parser.add_argument("--design-csv", type=Path)
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.lifo_max_cycles < 1:
        parser.error("--lifo-max-cycles must be at least 1")
    bins = parse_residence_bins(args.residence_bins_days)
    cases = matching_cases(args.input_dir, args.country, None, args.weather)
    if args.horizon:
        cases = [(p,c) for p,c in cases
                 if c["weather"].lower().endswith("_"+args.horizon.lower())]
    cases = [(p,c) for p,c in cases
             if _matches_formulation(c["experiment"],args.formulation)]
    if args.list_cases:
        for p,c in cases:
            print(c["country"], c["experiment"], c["weather"], p.name)
        return
    if not cases:
        parser.error("No cases match the given formulation/country/weather/horizon filters")
    identifiers = [(c["country"],c["experiment"],c["weather"]) for _,c in cases]
    if len(set(identifiers)) != len(identifiers):
        parser.error("Duplicate country/experiment/weather cases found")
    print(f"Selected {len(cases)} model cases (equal weight per model case):")
    for p,_ in cases: print(" ",p.name,flush=True)
    aggregate, detail = summarise_weather_profiles(
        cases, design_csv=args.design_csv, lifo_max_cycles=args.lifo_max_cycles,
        bins_days=bins, include_hydro=args.include_hydro,
    )
    tag = args.formulation.upper()
    if args.country:
        tag += "_"+args.country.upper()
    if args.horizon:
        tag += "_"+args.horizon
    if args.weather:
        tag += "_"+re.sub(r"[^A-Za-z0-9_-]", "_", args.weather)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    prefix = args.output_dir/f"storage_residence_profile_{tag}"
    summary_path = prefix.with_suffix(".csv")
    detail_path = args.output_dir/f"storage_residence_profile_cases_{tag}.csv"
    pdf_path = prefix.with_suffix(".pdf")
    aggregate.to_csv(summary_path,index=False)
    detail.to_csv(detail_path,index=False)
    plot_weather_profiles(aggregate,bins,pdf_path,svg=args.svg)
    print("Saved",pdf_path)
    print("Saved",summary_path)
    print("Saved",detail_path)
    if args.svg: print("Saved",pdf_path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
