"""Long-duration storage operation and LIFO residence-time diagnostics.

Run alongside ``_plots.py`` from the project root::

    pixi run python -m scripts.results_analysis.plot_storage_operations `
        --country NLD --experiment FX1 --weather 2015-2018

The weather selector also accepts ``2015``, ``2015_3y`` and
``weather_2015_3y``; use ``--list-cases`` to inspect available combinations.

Outputs (PDF only) are (1) SoC / installed energy capacity through time and
(2) cumulative energy withdrawn by LIFO residence time, measured in equivalent
storage-capacity turnovers. An audit CSV contains residence-time statistics.

The LIFO calculation treats each installed concrete technology separately,
then aggregates by CAES / geological family. It uses *energy per timestep*
flows (not power), applies flow efficiencies and storage loss, and verifies
the reconstructed inventory against Calliope's end-of-step ``storage``.

Initial inventory has unknown age, especially under cyclic_storage: its
withdrawals are excluded from known-age statistics and reported separately.
Injection is processed before withdrawal within a timestep (zero age for
same-timestep recycling). These are accounting assumptions, not physical
claims about mixing or molecule-specific hydrogen residence times.

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
LABELS = {
    "compressed_air_storage": "CAES", "lined_rock_cavern": "LRC",
    "salt_cavern": "SC", "aquifer": "SA", "depleted_gas_field": "DGF",
}
FONT_SIZE = 10
PLASMA = plt.get_cmap("plasma")


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


def load_stores(path: Path) -> list[Store]:
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
    if not stores:
        raise ValueError("No installed CAES/UHS storage capacity in this case")
    return stores


def lifo_residence(store: Store) -> LifoResult:
    """Energy-tagged LIFO stack; unknown-age initial stock remains untagged."""
    # Inventories are tracked in a lazily scaled unit so leakage costs O(1)
    # per timestep, rather than looping over every historic injection cohort.
    initial = float(store.soc[-1]) if store.cyclic else store.storage_initial * store.capacity
    stack: list[tuple[int | None, float]] = [(None, initial)] if initial > 0 else []
    scale, total_base = 1.0, initial
    ages, energies = [], []
    unknown, total_out, worst_error = 0.0, 0.0, 0.0
    tol = max(1e-2, 1e-6 * store.capacity)
    for i in range(len(store.time)):
        if i > 0 or store.cyclic:
            previous = (i - 1) % len(store.time)
            scale *= (1 - store.loss[previous]) ** store.dt[previous]
        if scale < 1e-80:
            stack = [(date, amount * scale) for date, amount in stack]
            total_base *= scale
            scale = 1.0

        charge_base = store.charge[i] / scale
        if charge_base > 0:
            stack.append((i, charge_base))
            total_base += charge_base
        withdraw_base = store.discharge[i] / scale
        total_out += store.discharge[i]
        if withdraw_base > total_base + tol / scale:
            raise ValueError(f"{store.tech}: withdrawal exceeds LIFO inventory at step {i}")
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
                age = (store.time[i] - store.time[date]).total_seconds() / 3600
                ages.append(max(age, 0.0))
                energies.append(taken * scale)
        if withdraw_base > tol / scale:
            raise ValueError(f"{store.tech}: insufficient stock at timestep {i}")
        # For tiny solver-residual withdrawals, do not let floating-point
        # leakage in the ledger accumulate across a multi-year run.
        actual = total_base * scale
        error = abs(actual - store.soc[i])
        worst_error = max(worst_error, error)
        if error > tol:
            raise ValueError(
                f"{store.tech}: inventory residual {error:.4g} at timestep {i} "
                f"(tolerance {tol:.4g}). Check flow efficiencies, storage loss, "
                "timestep resolution, and whether the model uses custom storage math."
            )
    return LifoResult(np.asarray(ages, float), np.asarray(energies, float),
                      unknown, total_out, worst_error)


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


def summarise(stores: list[Store], case: dict, design: pd.DataFrame):
    families = [f for f in FAMILIES if any(s.family == f for s in stores)]
    palette = np.linspace(0.12, 0.85, max(2, len(families)))
    colours = {fam: PLASMA(palette[i]) for i, fam in enumerate(families)}
    grouped, summaries = {}, []
    for fam in families:
        members = [s for s in stores if s.family == fam]
        results = [lifo_residence(s) for s in members]
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
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.18)
        for fam, (members, _, _) in grouped.items():
            cap = sum(s.capacity for s in members)
            time = members[0].time
            if any(not s.time.equals(time) for s in members):
                raise ValueError(f"{fam}: variants do not share the same timestamps")
            normalized = sum((s.soc for s in members), np.zeros(len(time))) / cap
            ax.plot(time, normalized, color=colours[fam], linewidth=1.15,
                    label=LABELS[fam], rasterized=False)
        ax.set_ylim(-0.02, 1.04)
        ax.set_ylabel("State of charge / installed capacity")
        ax.set_xlabel("Time")
        ax.grid(axis="y", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.legend(loc="upper center", ncol=len(grouped), frameon=False,
                   bbox_to_anchor=(0.55, 0.995))
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
            totals = np.cumsum(energies[indices]) / sum(s.capacity for s in members)
            ax.step(np.r_[0, sorted_days], np.r_[0, totals], where="post",
                    color=colours[fam], linewidth=1.5, label=LABELS[fam])
            maximum_days = max(maximum_days, float(sorted_days[-1]))
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(0, maximum_days * 1.12)
        ax.set_xlabel("LIFO residence time before withdrawal (days)")
        ax.set_ylabel("Cumulative withdrawn energy / installed capacity")
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


def select_case(input_dir: Path, country: str | None, experiment: str | None,
                weather: str | None, list_only: bool):
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
        return None
    matches = [
        (path, case) for path, case in cases
        if (country is None or case["country"].upper() == country.upper())
        and (experiment is None or case["experiment"].upper() == experiment.upper())
        and _weather_match(weather, case["weather"])
    ]
    if len(matches) != 1:
        sample = ", ".join(f"{c['country']}/{c['experiment']}/{c['weather']}"
                           for _, c in matches[:10])
        raise ValueError(f"Selection matched {len(matches)} cases (expected one): {sample}. "
                         "Use --list-cases or refine --country/--experiment/--weather")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--country", help="ISO3 country, e.g. NLD")
    parser.add_argument("--experiment", help="Representation, e.g. FX1")
    parser.add_argument("--weather", help="Weather window, e.g. 2015-2018")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--design-csv", type=Path,
                        help="Optional installed capacity / EPR / WTIR CSV")
    args = parser.parse_args()
    try:
        selected = select_case(args.input_dir, args.country, args.experiment,
                               args.weather, args.list_cases)
        if selected is None:
            return
        path, case = selected
        print(f"Reading {path.name}")
        stores = load_stores(path)
        design = _read_design_csv(args.design_csv, case)
        grouped, summary, colours = summarise(stores, case, design)
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
