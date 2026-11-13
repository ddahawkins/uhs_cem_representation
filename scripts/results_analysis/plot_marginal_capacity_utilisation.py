"""Weather-averaged marginal utilisation of storage power and energy capacity.

Place this script alongside ``plot_storage_operation.py`` and ``_plots.py``.
From the project root, run e.g.::

    pixi run python -m scripts.results_analysis.plot_marginal_capacity_utilisation `
        --countries NLD DEU GBR

The default compares B1 and FX1 on the *same weather windows* in each country.
It generates one injection/withdrawal PDF and one storage-energy PDF per
country, plus per-case, aggregate-curve and selection/audit CSVs. Add
``--include-caes`` to show CAES alongside the four geological technologies.

Power capacity is divided into equal nameplate-percentile increments (default
100, i.e. 1%). During each timestep, dispatched energy is converted to average
MW by dividing by the timestep duration. Every increment is credited with its
fraction actually dispatched. Results are equivalent full-load hours/year.

Energy capacity has two different diagnostics:
  * Occupancy: fraction of time each marginal MWh was occupied, measured above
    the minimum observed SoC to avoid an arbitrary constant inventory offset.
  * Discharge crossings: equivalent full discharge cycles/year through each
    marginal MWh, attributing withdrawals to the layer between end-of-step SoC
    and end-of-step SoC plus efficiency-adjusted withdrawal. This follows the
    same injection-before-withdrawal accounting convention as the LIFO tool.
    It is not a physical model of layered hydrogen. Simultaneous flows can
    produce an assumed intermediate inventory above installed capacity; the
    fraction of discharge represented by the profile is audited in the CSV.

For different weather cases, each curve is calculated after dividing by that
case's OWN installed capacity; the resulting curves are then averaged equally
over selected cases with usable capacity data. Capacity-weighted pooling across
weather cases is deliberately avoided. A selected-but-idle case contributes a
zero power-utilisation curve. No investment means an unavailable curve (NaN),
not zero utilisation; the selection count is shown explicitly.

Calliope output must include the results/inputs NetCDF groups supported by
``plot_storage_operation.py``. It supplies the established storage-family and
capacity extraction rules. For a B1 bundled design only, missing directional
power capacities may be inferred from installed storage cap, fixed EPR and
WTIR; inference sources are recorded. For other configurations, missing power
capacities are never fabricated. Supply ``--design-csv`` if needed, using the
existing storage-comparison CSV with ``case``, ``technology``, ``injection_MW``
and ``withdrawal_MW`` columns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "capacity_expansion_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "capacity_utilisation"
HOURS_PER_YEAR = 365.2425 * 24
EPS = 1e-7

FAMILIES = (
    "salt_cavern", "depleted_gas_field", "aquifer", "lined_rock_cavern",
)
FAMILY_LABELS = {
    "salt_cavern": "SC", "depleted_gas_field": "DGF",
    "aquifer": "SA", "lined_rock_cavern": "LRC",
    "compressed_air_storage": "CAES",
}
EXPERIMENT_COLOURS = {
    "B1": plt.get_cmap("plasma")(0.12),
    "FX1": plt.get_cmap("plasma")(0.72),
    "FX2": plt.get_cmap("plasma")(0.48),
    "FX4": plt.get_cmap("plasma")(0.90),
}
METRICS = (
    "injection_full_load_hours_per_year",
    "withdrawal_full_load_hours_per_year",
    "storage_occupancy_percent",
    "storage_discharge_crossings_per_year",
)
CASE_PATTERN = re.compile(
    r"^(?P<country>[A-Z]{3})_(?P<experiment>[A-Z][A-Z0-9]*)_"
    r"(?P<weather>weather_\d{4}_\d+y)\.nc$", re.IGNORECASE,
)


def discover_cases(input_dir: Path, countries: list[str] | None,
                   experiments: list[str], common_weather: bool) -> list[tuple[Path, dict]]:
    """Select cases while avoiding an unbalanced B1/FX1 weather comparison."""
    wanted = {c.upper() for c in countries} if countries else None
    exp_set = set(experiments)
    all_cases: dict[tuple[str, str, str], Path] = {}
    for path in sorted(input_dir.expanduser().glob("*.nc")):
        match = CASE_PATTERN.fullmatch(path.name)
        if not match or path.name.endswith(".pending.nc"):
            continue
        case = {k: v.upper() if k != "weather" else v.lower()
                for k, v in match.groupdict().items()}
        if (wanted is not None and case["country"] not in wanted) or case["experiment"] not in exp_set:
            continue
        key = (case["country"], case["experiment"], case["weather"])
        if key in all_cases:
            raise ValueError(f"Duplicate country/experiment/weather results: {key}")
        all_cases[key] = path

    if not all_cases:
        raise ValueError(f"No matching .nc files found in {input_dir}")
    if common_weather:
        allowed: dict[str, set[str]] = {}
        for country in sorted({k[0] for k in all_cases}):
            weather_by_experiment = [
                {k[2] for k in all_cases if k[0] == country and k[1] == exp}
                for exp in experiments
            ]
            allowed[country] = set.intersection(*weather_by_experiment)
            if not allowed[country]:
                warnings.warn(f"{country}: no weather windows common to {experiments}; skipped",
                              stacklevel=2)
        removed = sum(k[2] not in allowed[k[0]] for k in all_cases)
        if removed:
            print(f"Ignoring {removed} unmatched country-weather experiment files")
        all_cases = {k: path for k, path in all_cases.items()
                     if k[2] in allowed[k[0]]}
    if not all_cases:
        raise ValueError("No comparable weather cases remain; try --unpaired-weather")
    return [(path, dict(zip(("country", "experiment", "weather"), key)))
            for key, path in sorted(all_cases.items())]


def marginal_power_hours(flow_energy: np.ndarray, dt_hours: np.ndarray,
                         installed_mw: float, bins: int) -> np.ndarray:
    """Exact partial-slice full-load hours/year; conserves total dispatched energy."""
    if not (math.isfinite(installed_mw) and installed_mw > EPS):
        return np.full(bins, np.nan)
    flow_energy = np.asarray(flow_energy, dtype=float)
    dt_hours = np.asarray(dt_hours, dtype=float)
    if flow_energy.shape != dt_hours.shape or np.any(dt_hours <= 0):
        raise ValueError("Power dispatch and timestep durations do not align")
    if not (np.isfinite(flow_energy).all() and np.isfinite(dt_hours).all()):
        raise ValueError("Non-finite dispatch or timestep lengths")
    if np.min(flow_energy) < -max(EPS, installed_mw * 1e-5):
        raise ValueError("Materially negative power dispatch")
    power = np.maximum(flow_energy, 0) / dt_hours
    if np.max(power) > installed_mw * 1.02 + EPS:
        raise ValueError(f"Peak dispatch {np.max(power):g} exceeds installed "
                         f"power capacity {installed_mw:g} by more than 2%; "
                         "check timestep units/capacity extraction")
    boundaries = np.arange(bins, dtype=float) * (installed_mw / bins)
    width = installed_mw / bins
    # Power in a layer ranges from zero to its rated increment in each hour.
    fraction_dispatched = np.clip(
        (power[:, None] - boundaries[None, :]) / width, 0, 1
    )
    years = dt_hours.sum() / HOURS_PER_YEAR
    return (fraction_dispatched * dt_hours[:, None]).sum(axis=0) / years


def marginal_storage_profiles(stores: list, bins: int
                              ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Energy-capacity-weighted occupancy and discharge-through-layer profiles."""
    installed_energy = sum(s.capacity for s in stores)
    occupancy = np.zeros(bins)
    crossings = np.zeros(bins)
    accounted_discharge = 0.0
    total_discharge = 0.0
    used_energy_swing = 0.0
    years_list = [s.dt.sum() / HOURS_PER_YEAR for s in stores]
    if not np.allclose(years_list, years_list[0]):
        raise ValueError("Geological variants have unequal weather horizons")
    for store in stores:
        cap = store.capacity
        level = np.maximum(store.soc - float(store.soc.min()), 0)
        if level.max() > cap * 1.01 + EPS:
            raise ValueError(f"{store.tech}: SoC excursion exceeds installed energy")
        used_energy_swing += float(level.max())
        width = cap / bins
        thresholds = np.arange(bins, dtype=float)[None, :] * width
        # Marginal fraction of each energy increment occupied at step end.
        filled = np.clip((level[:, None] - thresholds) / width, 0, 1)
        occ = (filled * store.dt[:, None]).sum(axis=0) / store.dt.sum()
        # Assume charging precedes withdrawal; allocate withdrawals to the
        # marginal layers above end-of-step SoC, clipping at nameplate capacity.
        before_discharge = level + store.discharge
        before_filled = np.clip((before_discharge[:, None] - thresholds) / width, 0, 1)
        discharged_fraction = np.maximum(before_filled - filled, 0)
        discharged_in_layers = discharged_fraction.sum(axis=0) * width
        coverage = float(discharged_in_layers.sum())
        accounted_discharge += coverage
        total_discharge += float(store.discharge.sum())
        occupancy += (cap / installed_energy) * occ
        crossings += (cap / installed_energy) * (
            discharged_in_layers / width / years_list[0]
        )
    coverage_fraction = (accounted_discharge / total_discharge
                         if total_discharge > EPS else 1.0)
    return (occupancy * 100, crossings,
            used_energy_swing / installed_energy, coverage_fraction)


def _valid_capacity(value: float) -> bool:
    return math.isfinite(value) and value > EPS


def _family_power_capacity(stores: list, experiment: str, design: pd.DataFrame,
                           family: str, direction: str) -> tuple[float, str]:
    """Use explicit power decisions, the design CSV, or B1 fixed ratios only."""
    raw = [s.injection_cap if direction == "injection" else s.withdrawal_cap
           for s in stores]
    if all(_valid_capacity(p) for p in raw):
        return sum(raw), "result_capacity"
    if not design.empty and direction + "_MW" in design.columns:
        rows = design.loc[design["family"] == family]
        if not rows.empty:
            values = pd.to_numeric(rows[direction + "_MW"], errors="coerce")
            if values.notna().all() and values.sum() > EPS:
                return float(values.sum()), "design_csv"
    if experiment.upper() == "B1":
        values = []
        for store in stores:
            out = (store.withdrawal_cap if _valid_capacity(store.withdrawal_cap)
                   else store.capacity / store.design_epr
                   if _valid_capacity(store.design_epr) else float("nan"))
            if direction == "withdrawal":
                values.append(out)
            else:
                injection = (store.injection_cap if _valid_capacity(store.injection_cap)
                             else out / store.design_wtir
                             if _valid_capacity(out) and _valid_capacity(store.design_wtir)
                             else float("nan"))
                values.append(injection)
        if all(_valid_capacity(p) for p in values):
            return sum(values), "B1_fixed_ratio_inference"
    return float("nan"), "missing_directional_capacity"


def read_case(path: Path, case: dict, families: tuple[str, ...], bins: int,
              design_csv: Path | None) -> tuple[list[dict], list[dict]]:
    """Read storage operation using the same loaders as the LIFO plot."""
    try:
        from .plot_storage_operations import (  # type: ignore
            _read_design_csv,
            _series,
            load_stores,
            model_case,
        )
    except ImportError:
        from plot_storage_operations import (
            _read_design_csv,
            _series,
            load_stores,
            model_case,
        )
    import xarray as xr

    stores = load_stores(path, allow_empty=True)
    indexed = defaultdict(list)
    for store in stores:
        indexed[store.family].append(store)
    design = (_read_design_csv(design_csv, model_case(path))
              if design_csv is not None else pd.DataFrame())
    dispatch: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if stores:
        with xr.open_dataset(path, group="results") as results:
            for store in stores:
                dispatch[store.tech] = (
                    _series(results["flow_in"], store.tech, sum_other=True),
                    _series(results["flow_out"], store.tech, sum_other=True),
                )

    profiles: list[dict] = []
    audit_rows: list[dict] = []
    centre_pct = (np.arange(bins) + 0.5) * 100 / bins
    for family in families:
        members = indexed[family]
        selected = bool(members)
        installed_energy = sum(s.capacity for s in members)
        power_cap, source = {}, {}
        curves = {name: np.full(bins, np.nan) for name in METRICS}
        diagnostic = {"soc_excursion_fraction": float("nan"),
                      "storage_discharge_profile_coverage": float("nan")}
        years = float(members[0].dt.sum() / HOURS_PER_YEAR) if selected else float("nan")
        if selected:
            durations = [float(m.dt.sum()) for m in members]
            if not np.allclose(durations, durations[0]):
                raise ValueError(f"{family}: variants have different timestep durations")
            for direction in ("injection", "withdrawal"):
                cap, cap_source = _family_power_capacity(
                    members, case["experiment"], design, family, direction
                )
                power_cap[direction], source[direction] = cap, cap_source
                if _valid_capacity(cap):
                    raw_energy = np.sum(
                        [dispatch[s.tech][0 if direction == "injection" else 1]
                         for s in members], axis=0,
                    )
                    try:
                        curves[f"{direction}_full_load_hours_per_year"] = (
                            marginal_power_hours(raw_energy, members[0].dt, cap, bins)
                        )
                    except ValueError as exc:
                        warnings.warn(f"{path.name} {family} {direction}: {exc}", stacklevel=2)
                        source[direction] = f"invalid_power_data: {exc}"
                else:
                    warnings.warn(f"{path.name} {family}: missing {direction} "
                                  "capacity (see audit CSV)", stacklevel=2)

            occ, crossings, excursion, coverage = marginal_storage_profiles(members, bins)
            curves["storage_occupancy_percent"] = occ
            curves["storage_discharge_crossings_per_year"] = crossings
            diagnostic.update(soc_excursion_fraction=excursion,
                              storage_discharge_profile_coverage=coverage)
            if coverage < 0.99:
                warnings.warn(
                    f"{path.name} {family}: only {coverage:.1%} of storage "
                    "withdrawals are assigned to energy-capacity layers by "
                    "the assumed within-timestep operation order", stacklevel=2,
                )
        for metric, values in curves.items():
            for index, (capacity_pct, value) in enumerate(zip(centre_pct, values)):
                profiles.append({**case, "technology": family, "metric": metric,
                                 "slice_index": index + 1,
                                 "capacity_percentile": capacity_pct,
                                 "value": float(value), "selected": selected})
        audit_rows.append({**case, "technology": family, "selected": selected,
                           "variant_count": len(members), "simulation_years": years,
                           "installed_storage_energy": installed_energy,
                           "installed_injection_power": power_cap.get("injection", float("nan")),
                           "installed_withdrawal_power": power_cap.get("withdrawal", float("nan")),
                           "injection_capacity_source": source.get("injection", "not_selected"),
                           "withdrawal_capacity_source": source.get("withdrawal", "not_selected"),
                           "injection_profile_available": bool(np.isfinite(curves[METRICS[0]]).all()),
                           "withdrawal_profile_available": bool(np.isfinite(curves[METRICS[1]]).all()),
                           "mean_full_discharge_cycles_per_year": (
                               sum(float(s.discharge.sum()) for s in members) / installed_energy / years
                               if selected else float("nan")
                           ),
                           **diagnostic})
    return profiles, audit_rows


def aggregate_curves(per_case: pd.DataFrame) -> pd.DataFrame:
    """Equal-weather-weight arithmetic mean; NaN cases are not zero curves."""
    return (per_case.groupby(
        ["country", "experiment", "technology", "metric", "slice_index",
         "capacity_percentile"], as_index=False,
    ).agg(mean_value=("value", "mean"),
          profiled_weather_scenarios=("value", "count")))


def selection_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (country, experiment, technology), group in audit.groupby(
        ["country", "experiment", "technology"], sort=True
    ):
        selected = group.loc[group["selected"]]
        rows.append({
            "country": country, "experiment": experiment, "technology": technology,
            "selected_scenarios": len(selected), "total_weather_scenarios": len(group),
            "injection_profile_scenarios": int(selected["injection_profile_available"].sum()),
            "withdrawal_profile_scenarios": int(selected["withdrawal_profile_available"].sum()),
            "mean_energy_capacity_when_selected": selected["installed_storage_energy"].mean(),
            "mean_injection_capacity_when_selected": selected["installed_injection_power"].mean(),
            "mean_withdrawal_capacity_when_selected": selected["installed_withdrawal_power"].mean(),
            "mean_storage_soc_excursion_fraction": selected["soc_excursion_fraction"].mean(),
            "mean_full_discharge_cycles_per_year": selected["mean_full_discharge_cycles_per_year"].mean(),
            "mean_energy_discharge_profile_coverage": selected["storage_discharge_profile_coverage"].mean(),
        })
    return pd.DataFrame(rows)


def _colours(experiments: list[str]) -> dict[str, tuple]:
    fallbacks = plt.get_cmap("plasma")(
        np.linspace(0.1, 0.9, len(experiments))
    )
    return {exp: EXPERIMENT_COLOURS.get(exp, fallbacks[i])
            for i, exp in enumerate(experiments)}


def draw_country(average: pd.DataFrame, selection: pd.DataFrame, country: str,
                 experiments: list[str], families: tuple[str, ...],
                 output: Path, *, energy: bool) -> None:
    """Scientific 4/5-by-2 panels: two experimental curves per technology."""
    if energy:
        metrics = METRICS[2:]
        titles = ("SoC occupancy", "Discharge through energy layer")
        labels = ("Mean occupied time (%)", "Discharge crossings (cycles/yr)")
    else:
        metrics = METRICS[:2]
        titles = ("Injection power", "Withdrawal power")
        labels = ("Marginal full-load hours/yr", "Marginal full-load hours/yr")

    colours = _colours(experiments)
    with plt.rc_context({"font.size": 9.5, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(len(families), 2, figsize=(7.2, 1.78 * len(families) + 1.45),
                                 sharex=True, squeeze=False)
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.085, top=0.835,
                            wspace=0.29, hspace=0.34)
        for j, title in enumerate(titles):
            axes[0, j].set_title(title, fontsize=10.5, fontweight="bold", pad=12)
        for i, family in enumerate(families):
            summary = selection.loc[(selection["country"] == country)
                                    & (selection["technology"] == family)]
            summary_by_exp = summary.set_index("experiment") if not summary.empty else None
            count_labels = []
            for exp in experiments:
                if summary_by_exp is None or exp not in summary_by_exp.index:
                    count_labels.append(f"{exp} 0/0")
                else:
                    row = summary_by_exp.loc[exp]
                    count_labels.append(
                        f"{exp} {int(row['selected_scenarios'])}/{int(row['total_weather_scenarios'])}"
                    )
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                maximum = 0.0
                has_line = False
                for exp in experiments:
                    part = average.loc[(average["country"] == country)
                                       & (average["experiment"] == exp)
                                       & (average["technology"] == family)
                                       & (average["metric"] == metric)]
                    if part.empty:
                        continue
                    part = part.sort_values("capacity_percentile")
                    values = part["mean_value"].to_numpy(dtype=float)
                    if not np.isfinite(values).any():
                        continue
                    has_line = True
                    maximum = max(maximum, float(np.nanmax(values)))
                    ax.plot(part["capacity_percentile"], values,
                            color=colours[exp], linestyle="-" if exp == "B1" else "--",
                            linewidth=1.75, solid_capstyle="round", label=exp)
                if not has_line:
                    ax.text(0.5, 0.50, "Not installed / no usable capacity",
                            color="0.53", ha="center", va="center", fontsize=8.2,
                            transform=ax.transAxes)
                ax.set_xlim(0, 100)
                ax.set_xticks((0, 25, 50, 75, 100))
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
                ax.grid(axis="both", color="0.89", linewidth=0.55)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                if energy and j == 0:
                    ax.set_ylim(0, 102)
                else:
                    ax.set_ylim(0, max(1.0, maximum * 1.12))
                if j == 0:
                    ax.set_ylabel(f"{FAMILY_LABELS[family]}\n{labels[j]}")
                    ax.text(0.98, 0.98, "  |  ".join(count_labels), ha="right",
                            va="top", fontsize=7.1, color="0.37",
                            transform=ax.transAxes)
                elif i == 0:
                    ax.set_ylabel(labels[j], fontsize=8.7)
                if i == len(families) - 1:
                    ax.set_xlabel("Increment of installed capacity")

        fig.legend(
            handles=[Line2D([], [], color=colours[exp], linewidth=2,
                            linestyle="-" if exp == "B1" else "--", label=exp)
                     for exp in experiments],
            loc="upper center", bbox_to_anchor=(0.55, 0.94),
            ncol=len(experiments), frameon=False, columnspacing=2.5,
        )
        fig.text(0.51, 0.985, f"{country}: marginal {'energy' if energy else 'power'} capacity utilisation",
                 ha="center", va="top", fontsize=11, fontweight="bold")
        fig.savefig(output)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="*", help="ISO3 countries; default all")
    parser.add_argument("--experiments", nargs="+", default=["B1", "FX1"],
                        help="Configurations to compare (default B1 FX1)")
    parser.add_argument("--include-caes", action="store_true",
                        help="Add CAES as a fifth technology row")
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of marginal increments; default 100 (1%% each)")
    parser.add_argument("--unpaired-weather", action="store_true",
                        help="Allow different weather-window sets for each experiment")
    parser.add_argument("--design-csv", type=Path,
                        help="Optional installed component capacities CSV from earlier analysis")
    args = parser.parse_args()
    experiments = [s.upper() for s in args.experiments]
    if len(set(experiments)) != len(experiments):
        parser.error("Duplicate experiments are not allowed")
    if not 1 <= args.bins <= 1000:
        parser.error("--bins must be in the range 1..1000")
    families = FAMILIES + (("compressed_air_storage",) if args.include_caes else ())
    cases = discover_cases(args.input_dir, args.countries, experiments,
                           common_weather=not args.unpaired_weather)
    print(f"Processing {len(cases)} country / experiment / weather cases")
    profile_rows: list[dict] = []
    audit_rows: list[dict] = []
    for path, case in cases:
        print(f"Reading {path.name}", flush=True)
        try:
            profile, audit = read_case(path, case, families, args.bins, args.design_csv)
        except Exception as exc:
            raise RuntimeError(f"Cannot process {path}: {exc}") from exc
        profile_rows.extend(profile)
        audit_rows.extend(audit)
    per_case = pd.DataFrame(profile_rows)
    audit = pd.DataFrame(audit_rows)
    average = aggregate_curves(per_case)
    selection = selection_summary(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "storage_marginal_capacity"
    tables = {
        f"{prefix}_cases.csv": per_case,
        f"{prefix}_weather_averages.csv": average,
        f"{prefix}_selection.csv": selection,
        f"{prefix}_audit.csv": audit,
    }
    for filename, frame in tables.items():
        destination = args.output_dir / filename
        # frame.to_csv(destination, index=False)
        print(f"Saved {destination}")
    for country in sorted(average["country"].unique()):
        for energy, suffix in ((False, "power"), (True, "energy")):
            destination = args.output_dir / f"{prefix}_{suffix}_{country}.pdf"
            draw_country(average, selection, country, experiments, families,
                         destination, energy=energy)
            print(f"Saved {destination}")


if __name__ == "__main__":
    main()
