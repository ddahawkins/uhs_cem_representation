"""Long-duration storage operation and LIFO residence-time diagnostics.

Run alongside ``_plots.py`` from the project root::

    pixi run python -m scripts.results_analysis.plot_storage_operations `
        --country NLD --experiment FX1 --weather 2015-2018

The weather selector also accepts ``2015``, ``2015_3y`` and
``weather_2015_3y``; use ``--list-cases`` to inspect available combinations.

Outputs (PDF only) are (1) SoC / installed energy capacity through time and
(2) an energy-weighted LIFO residence-time CDF: cumulative percentage of
withdrawn energy with known age. An audit CSV contains residence-time
statistics and capacity turnovers.

``--aggregate-weather`` instead analyses every matching weather window for a
country/experiment and writes one 100%-stacked residence-time profile per
technology. Within each weather case, residence-bin shares are normalised by
the technology's known-age discharged throughput. The shares are then averaged
with *equal weight per weather case* (not pooled by capacity or throughput).
The aggregate also reports how often the technology was installed and its
mean annual equivalent full discharge cycles when installed. It writes both
an aggregate and a per-case audit CSV. Example::

    pixi run python -m scripts.results_analysis.plot_storage_operations `
        --country NLD --experiment B1 --aggregate-weather

The LIFO calculation treats each installed concrete technology separately,
then aggregates by CAES / geological family. It uses *energy per timestep*
flows (not power), applies flow efficiencies and storage loss, and verifies
the reconstructed inventory against Calliope's end-of-step ``storage``.

For cyclic storage, the dispatch window is replayed as a periodic prehistory
until the LIFO withdrawal-age distribution stabilizes. Any remaining unknown
initial-inventory withdrawals are excluded and explicitly reported. Repeating
the dispatch is an additional assumption: cyclic storage alone fixes only the
boundary quantity, not its composition or age. Injection is processed before
withdrawal within a timestep (zero age for same-timestep recycling). These
are accounting assumptions, not a physical model of hydrogen mixing.

Optional ``--design-csv`` accepts the storage comparison CSV with columns
``case, technology, storage_MWh, injection_MW, withdrawal_MW, EPR_hours, WTIR``
to supply installed design EPR / WTIR when unavailable from model outputs.
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

import matplotlib.dates as mdates
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
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / "results_2h_no_coupling_bounds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
FAMILIES = ("compressed_air_storage", "lined_rock_cavern", "salt_cavern",
            "aquifer", "depleted_gas_field")
SOC_LONG_DURATION = ("salt_cavern", "aquifer", "depleted_gas_field")
SOC_MEDIUM_DURATION = ("compressed_air_storage", "lined_rock_cavern")
LABELS = {
    "compressed_air_storage": "CAES", "lined_rock_cavern": "LRC",
    "salt_cavern": "SC", "aquifer": "SA", "depleted_gas_field": "DGF",
}
FONT_SIZE = 10
PLASMA = plt.get_cmap("plasma")
DEFAULT_RESIDENCE_BINS_DAYS = (1.0, 7.0, 30.0, 90.0, 365.0)
HOURS_PER_YEAR = 24 * 365.2425


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


def _family(tech: str) -> str | None:
    identity = tech.lower()
    if identity in ("compressed_air_storage", "caes", "compressed_air"):
        return "compressed_air_storage"
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


def load_stores(path: Path, *, allow_empty: bool = False) -> list[Store]:
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
                family = _family(tech)
                if family:
                    store = _load_store(results, inputs, tech, family)
                    if store is not None:
                        stores.append(store)
    if not stores and not allow_empty:
        raise ValueError("No installed CAES/UHS storage capacity in this case")
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
    palette = np.linspace(0.12, 0.85, max(2, len(families)))
    colours = {fam: PLASMA(palette[i]) for i, fam in enumerate(families)}
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


def plot_soc(grouped: dict, colours: dict, path: Path) -> None:
    with plt.rc_context({"font.size": FONT_SIZE, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(
            2, 1, sharex=True, figsize=(10.5, 5.9),
            gridspec_kw={"height_ratios": (2, 1)},
        )
        fig.subplots_adjust(left=0.105, right=0.98, top=0.90,
                            bottom=0.13, hspace=0.36)
        for ax, families, group_label in (
            (axes[0], SOC_LONG_DURATION, "Long-duration"),
            (axes[1], SOC_MEDIUM_DURATION, "Medium-duration"),
        ):
            for fam in families:
                if fam not in grouped:
                    continue
                members, _, _ = grouped[fam]
                cap = sum(s.capacity for s in members)
                time = members[0].time
                if any(not s.time.equals(time) for s in members):
                    raise ValueError(f"{fam}: variants do not share the same timestamps")
                normalized = sum((s.soc for s in members), np.zeros(len(time))) / cap
                ax.plot(time, normalized, color=colours[fam], linewidth=1.15,
                        label=LABELS[fam], rasterized=False)
            ax.set_ylim(-0.02, 1.04)
            ax.grid(axis="y", color="0.85", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(0.99, 1.02, group_label, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=FONT_SIZE,
                    clip_on=False)
            if ax.lines:
                ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005),
                          ncol=len(ax.lines), frameon=False, handlelength=2.0)
            else:
                ax.text(0.5, 0.5, "No installed storage in this group",
                        ha="center", va="center", transform=ax.transAxes)

        fig.supylabel("State of charge / installed capacity",
                     x=0.02, fontsize=FONT_SIZE)
        axes[-1].set_xlabel("Time")
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        axes[-1].xaxis.set_major_locator(locator)
        axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.savefig(path)
        plt.close(fig)


def _metric(value: float, unit: str = "") -> str:
    return f"{value:.3g}{unit}" if math.isfinite(value) else "n/a"


def plot_residence(grouped: dict, summary: pd.DataFrame, colours: dict, path: Path) -> None:
    with plt.rc_context({"font.size": FONT_SIZE, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(10.5, 4.75))
        fig.subplots_adjust(left=0.10, right=0.69, top=0.91, bottom=0.16)
        maximum_days = 1.0
        for fam, (members, ages, energies) in grouped.items():
            if not len(ages) or float(energies.sum()) <= 0:
                continue
            indices = np.argsort(ages, kind="stable")
            sorted_days = ages[indices] / 24
            # Normalize against discharged throughput with known residence time,
            # not energy capacity. Unknown-age initial withdrawals cannot be
            # assigned a residence time and remain reported in the audit CSV.
            totals = np.cumsum(energies[indices]) / float(energies.sum())
            ax.step(np.r_[0, sorted_days], np.r_[0, totals], where="post",
                    color=colours[fam], linewidth=1.5, label=LABELS[fam])
            maximum_days = max(maximum_days, float(sorted_days[-1]))
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(0, maximum_days * 1.12)
        ax.set_xlabel("LIFO residence time before withdrawal (days)")
        ax.set_ylabel("Cumulative share of known-age discharge")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis="both", color="0.86", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.has_data():
            ax.legend(loc="upper left", ncol=2, frameon=False)
        else:
            ax.text(0.5, 0.5, "No discharge with known residence time",
                    ha="center", va="center", transform=ax.transAxes)

        # EPR is a discharge-duration benchmark (hours), WTIR a power ratio.
        # Neither is another estimate of actual residence time.
        for i, row in summary.reset_index(drop=True).iterrows():
            fam = str(row["technology"])
            y = 0.93 - i * 0.18
            fig.text(0.735, y, LABELS[fam], color=colours[fam],
                     fontsize=FONT_SIZE, fontweight="bold")
            fig.text(0.735, y - 0.045,
                     f"Mean { _metric(row['mean_term_hours'] / 24, ' d') }"
                     f"  |  P50 { _metric(row['median_term_hours'] / 24, ' d') }")
            fig.text(0.735, y - 0.090,
                     f"EPR { _metric(row['design_EPR_hours'], ' h') }"
                     f"  |  WTIR { _metric(row['design_WTIR']) }")
            if row["unknown_fraction_of_discharge"] > 0.01:
                fig.text(0.735, y - 0.128,
                         f"Initial age unknown: {row['unknown_fraction_of_discharge']:.0%}",
                         fontsize=FONT_SIZE)
        fig.savefig(path)
        plt.close(fig)


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
    return ([f"{number(a)}–{number(b)} d"
             for a, b in zip(bounds[:-1], bounds[1:])]
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
                               bins_days: tuple[float, ...]
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
        stores = load_stores(path, allow_empty=True)
        if stores:
            design = _read_design_csv(design_csv, case)
            grouped, summary, _ = summarise(
                stores, case, design, lifo_max_cycles=lifo_max_cycles,
            )
            summary = summary.set_index("technology")
        else:
            grouped, summary = {}, pd.DataFrame()
        for family in FAMILIES:
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
                          path: Path) -> None:
    """One 100%-stacked weather-average residence profile per technology."""
    keys = _residence_bin_keys(bins_days)
    labels = _residence_bin_labels(bins_days)
    colours = [PLASMA(x) for x in np.linspace(0.10, 0.92, len(keys))]
    n = len(aggregate)
    with plt.rc_context({"font.size": FONT_SIZE, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(10.5, max(4.1, 0.53 * n + 1.6)))
        fig.subplots_adjust(left=0.13, right=0.69, top=0.83, bottom=0.26)
        coords = np.arange(n)[::-1]
        position = blended_transform_factory(ax.transAxes, ax.transData)
        for y, (_, row) in zip(coords, aggregate.iterrows()):
            fractions = row[keys].to_numpy(dtype=float)
            if np.isfinite(fractions).all():
                start = 0.0
                for share, colour in zip(fractions, colours):
                    if share > 0:
                        ax.barh(y, share, left=start, height=0.64,
                                color=colour, edgecolor="white", linewidth=0.45)
                    start += share
            else:
                message = ("Not installed" if row["selected_scenarios"] == 0
                           else "No attributable discharge")
                ax.text(0.5, y, message, ha="center", va="center", color="0.55")
            selected = f"{int(row['selected_scenarios'])}/{int(row['total_weather_scenarios'])}"
            ax.text(1.095, y, selected, ha="center", va="center",
                    transform=position, clip_on=False)
            cycles = row["mean_equivalent_full_discharge_cycles_per_year"]
            ax.text(1.39, y, f"{cycles:.2f}" if math.isfinite(cycles) else "—",
                    ha="center", va="center", transform=position, clip_on=False)

        header_y = coords[0] + 0.9
        ax.text(1.095, header_y, "Selected", ha="center", va="center",
                transform=position, clip_on=False)
        ax.text(1.39, header_y, "Cycles/yr", ha="center", va="center",
                transform=position, clip_on=False)
        ax.set_yticks(coords, [LABELS[f] for f in aggregate["technology"]])
        ax.tick_params(axis="y", length=0, pad=9)
        ax.set_ylim(-0.70, header_y + 0.38)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Mean share of withdrawn energy")
        ax.grid(axis="x", color="0.86", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.legend(
            handles=[Patch(facecolor=c, label=l) for c, l in zip(colours, labels)],
            loc="lower center", bbox_to_anchor=(0.46, 0.025),
            ncol=min(3, len(keys)), frameon=False, columnspacing=1.6,
        )
        fig.savefig(path)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--country", help="ISO3 country, e.g. NLD")
    parser.add_argument("--experiment", help="Representation, e.g. FX1")
    parser.add_argument("--weather", help="Weather window, e.g. 2015-2018")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--aggregate-weather", action="store_true",
                        help="Average residence-time composition equally across all "
                             "matching weather scenarios (requires country and experiment)")
    parser.add_argument("--residence-bins-days", default="1,7,30,90,365",
                        help="Comma-separated residence-bin upper bounds in days "
                             "for aggregate plot (default 1,7,30,90,365)")
    parser.add_argument("--design-csv", type=Path,
                        help="Optional installed capacity / EPR / WTIR CSV")
    parser.add_argument("--lifo-max-cycles", type=int, default=8,
                        help="Maximum periodic LIFO replays for cyclic storage (default 8)")
    args = parser.parse_args()
    if args.aggregate_weather and not args.list_cases:
        if not args.country or not args.experiment:
            parser.error("--aggregate-weather requires --country and --experiment")
        try:
            bins_days = parse_residence_bins(args.residence_bins_days)
            cases = matching_cases(args.input_dir, args.country,
                                   args.experiment, args.weather)
            if not cases:
                raise ValueError("No matching weather cases found; use --list-cases")
            identifiers = [(c["country"], c["experiment"], c["weather"])
                           for _, c in cases]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("Multiple result files have the same country, "
                                 "experiment and weather window; resolve duplicates")
            aggregate, detail = summarise_weather_profiles(
                cases, design_csv=args.design_csv,
                lifo_max_cycles=args.lifo_max_cycles, bins_days=bins_days,
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"{args.country.upper()}_{args.experiment.upper()}"
            if args.weather:
                suffix += "_" + re.sub(r"[^a-zA-Z0-9_-]", "_", args.weather)
            figure = args.output_dir / f"storage_residence_profile_{suffix}.pdf"
            aggregate_csv = args.output_dir / f"storage_residence_profile_{suffix}.csv"
            case_csv = args.output_dir / f"storage_residence_profile_cases_{suffix}.csv"
            plot_weather_profiles(aggregate, bins_days, figure)
            # aggregate.to_csv(aggregate_csv, index=False)
            # detail.to_csv(case_csv, index=False)
        except ValueError as exc:
            parser.error(str(exc))
        for output in (figure, aggregate_csv, case_csv):
            print(f"Saved {output}")
        return
    try:
        selected = select_case(args.input_dir, args.country, args.experiment,
                               args.weather, args.list_cases)
        if selected is None:
            return
        path, case = selected
        print(f"Reading {path.name}")
        stores = load_stores(path)
        design = _read_design_csv(args.design_csv, case)
        grouped, summary, colours = summarise(
            stores, case, design, lifo_max_cycles=args.lifo_max_cycles,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (summary["design_EPR_hours"].isna() | summary["design_WTIR"].isna()).any():
        warnings.warn("Some installed EPR/WTIR values are unavailable; "
                      "provide --design-csv to add them", stacklevel=1)

    suffix = f"{case['country']}_{case['experiment']}_{case['weather']}".replace("/", "_")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    soc_pdf = args.output_dir / f"storage_soc_{suffix}.pdf"
    term_pdf = args.output_dir / f"storage_lifo_term_{suffix}.pdf"
    csv = args.output_dir / f"storage_lifo_summary_{suffix}.csv"
    plot_soc(grouped, colours, soc_pdf)
    plot_residence(grouped, summary, colours, term_pdf)
    # summary.to_csv(csv, index=False)
    for output in (soc_pdf, term_pdf, csv):
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
