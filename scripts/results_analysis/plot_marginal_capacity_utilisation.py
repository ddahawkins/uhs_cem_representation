"""Compare Dutch depleted-gas-field marginal power utilisation across formulations.

Place beside scripts/results_analysis/plot_storage_operations.py, and run from
project root:

    pixi run python -m scripts.results_analysis.plot_marginal_capacity_utilisation_DGF

Default: NLD, three-year weather horizons, B1/P1/F1/FX1. 

    pixi run python -m scripts.results_analysis.plot_marginal_capacity_utilisation `
        --experiments B1 P1 F1 FX1 

Only DGF injection/withdrawal capacity is analysed. No energy-capacity metrics
are calculated. The resulting figure is 190 x 76 mm (Elsevier two-column width).

Each selected weather case has its *own* installed directional power capacity
split into equal marginal capacity increments. Its equivalent full-load hours
per year are calculated for each increment, then averaged equally across
weather cases with a usable profile. Scenarios without DGF investment are
excluded from the mean, not represented as zero. Scenarios with DGF selected
but no dispatch correctly contribute a zero curve. The legend shows DGF
selection counts (selected weather windows / matching weather windows).

Pairing defaults to common weather windows across ALL selected formulations.
Input filenames must follow NLD_B1_weather_2009_3y.nc, etc. This script uses
the established plot_storage_operations.load_stores and _series extractors and
keeps the original B1-only fixed-ratio fallback for missing power capacities.

Outputs: vector PDF, per-case profiles, averaged profiles, selection and audit
CSVs. Add --svg for a vector copy suitable for PowerPoint.
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
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "capacity_expansion_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "capacity_utilisation" / "dgf_nld"

COUNTRY = "NLD"                        # Intentionally fixed to the Dutch case study.
FAMILY = "depleted_gas_field"          # Intentionally excludes all other technologies.
DEFAULT_EXPERIMENTS = ("B1", "P1", "F1", "FX1")
EXPERIMENT_LABELS = {"B1": "B", "P1": "P", "F1": "F", "FX1": "FX",
                     "FX2": "FX2", "FX4": "FX4"}
EXPERIMENT_COLOURS = {             # Stable Plasma assignment for the 1-sample series.
    "B1": plt.get_cmap("plasma")(0.12),
    "P1": plt.get_cmap("plasma")(0.34),
    "F1": plt.get_cmap("plasma")(0.57),
    "FX1": plt.get_cmap("plasma")(0.79),
    "FX2": plt.get_cmap("plasma")(0.88),
    "FX4": plt.get_cmap("plasma")(0.95),
}
EXPERIMENT_STYLES = {"B1": "-", "P1": "-", "F1": "-",
                     "FX1": "-", }
METRICS = ("injection_full_load_hours_per_year",
           "withdrawal_full_load_hours_per_year")
HOURS_PER_YEAR = 365.2425 * 24
EPS = 1e-7
MM_TO_IN = 1 / 25.4
FIGURE_SIZE_MM = (190, 76)
CASE_PATTERN = re.compile(
    r"^(?P<country>[A-Z]{3})_(?P<experiment>(?:B|P|F|FX)\d+[A-Z]?)_"
    r"(?P<weather>weather_\d{4}_(?P<horizon>\d+y))\.nc$", re.IGNORECASE,
)


def discover_cases(input_dir: Path, experiments: list[str], horizon: str,
                   common_weather: bool) -> list[tuple[Path, dict]]:
    """Read NLD-only case filenames, selecting matched weather samples by default."""
    all_cases: dict[tuple[str, str], Path] = {}
    for path in sorted(input_dir.expanduser().glob("*.nc")):
        if path.name.endswith(".pending.nc"):
            continue
        match = CASE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        group = match.groupdict()
        if group["country"].upper() != COUNTRY or group["horizon"].lower() != horizon.lower():
            continue
        experiment = group["experiment"].upper()
        if experiment not in experiments:
            continue
        key = (experiment, group["weather"].lower())
        if key in all_cases:
            raise ValueError(f"Duplicate result case {key}: {path}")
        all_cases[key] = path
    if not all_cases:
        raise FileNotFoundError(
            f"No NLD results for {experiments} and horizon={horizon} in {input_dir}")

    absent = sorted(set(experiments) - {k[0] for k in all_cases})
    if absent:
        raise ValueError(f"Missing NLD NetCDF files for experiment(s): {', '.join(absent)}")

    if common_weather:
        samples = [set(w for e, w in all_cases if e == experiment)
                   for experiment in experiments]
        shared = set.intersection(*samples)
        if not shared:
            raise ValueError(
                f"No common {horizon} NLD weather samples across {experiments}; "
                "use --unpaired-weather only if unequal sample sets are intended")
        removed = sum(weather not in shared for (_, weather) in all_cases)
        if removed:
            warnings.warn(f"Excluded {removed} unmatched weather/experiment cases", stacklevel=2)
        all_cases = {key: path for key, path in all_cases.items() if key[1] in shared}
        print(f"Using {len(shared)} matched Dutch weather scenarios")

    return [(path, {"country": COUNTRY, "experiment": experiment, "weather": weather})
            for (experiment, weather), path in sorted(all_cases.items())]


def marginal_power_hours(flow_energy: np.ndarray, dt_hours: np.ndarray,
                         installed_mw: float, bins: int) -> np.ndarray:
    """Original fractional marginal-slice calculation, as equivalent FLH/year."""
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
                         f"power capacity {installed_mw:g} by >2%; check unit conventions")
    boundaries = np.arange(bins, dtype=float) * (installed_mw / bins)
    width = installed_mw / bins
    fraction = np.clip((power[:, None] - boundaries[None, :]) / width, 0, 1)
    years = dt_hours.sum() / HOURS_PER_YEAR
    return (fraction * dt_hours[:, None]).sum(axis=0) / years


def _valid_capacity(value: float) -> bool:
    return math.isfinite(value) and value > EPS


def _family_power_capacity(stores: list, experiment: str, design: pd.DataFrame,
                           direction: str) -> tuple[float, str]:
    """Preserve installed decision, CSV, or B1 fixed-ratio fallback precedence."""
    raw = [s.injection_cap if direction == "injection" else s.withdrawal_cap
           for s in stores]
    if all(_valid_capacity(p) for p in raw):
        return sum(raw), "result_capacity"
    if not design.empty and direction + "_MW" in design.columns:
        rows = design.loc[design["family"] == FAMILY]
        if not rows.empty:
            values = pd.to_numeric(rows[direction + "_MW"], errors="coerce")
            if values.notna().all() and values.sum() > EPS:
                return float(values.sum()), "design_csv"
    if experiment.upper() == "B1":
        values = []
        for s in stores:
            out = (s.withdrawal_cap if _valid_capacity(s.withdrawal_cap)
                   else s.capacity / s.design_epr
                   if _valid_capacity(s.design_epr) else float("nan"))
            if direction == "withdrawal":
                values.append(out)
            else:
                inlet = (s.injection_cap if _valid_capacity(s.injection_cap)
                         else out / s.design_wtir
                         if _valid_capacity(out) and _valid_capacity(s.design_wtir)
                         else float("nan"))
                values.append(inlet)
        if all(_valid_capacity(p) for p in values):
            return sum(values), "B1_fixed_ratio_inference"
    return float("nan"), "missing_directional_capacity"


def read_case(path: Path, case: dict, bins: int,
              design_csv: Path | None) -> tuple[list[dict], dict]:
    """Extract DGF only, ignoring the SoC/energy-capacity calculations."""
    try:
        from .plot_storage_operations import (
            _read_design_csv, _series, load_stores, model_case,
        )
    except ImportError:
        from plot_storage_operations import (
            _read_design_csv, _series, load_stores, model_case,
        )
    import xarray as xr

    stores = [s for s in load_stores(path, allow_empty=True) if s.family == FAMILY]
    selected = bool(stores)
    design = (_read_design_csv(design_csv, model_case(path))
              if design_csv is not None else pd.DataFrame())
    dt = None
    flows = {}
    if selected:
        dt = stores[0].dt
        if not all(np.allclose(s.dt, dt) for s in stores):
            raise ValueError(f"DGF variants have different timesteps: {path.name}")
        with xr.open_dataset(path, group="results") as results:
            for store in stores:
                flows[store.tech] = (
                    _series(results["flow_in"], store.tech, sum_other=True),
                    _series(results["flow_out"], store.tech, sum_other=True),
                )

    centre_pct = (np.arange(bins) + 0.5) * 100 / bins
    curves = {metric: np.full(bins, np.nan) for metric in METRICS}
    capacities = {}
    sources = {}
    if selected:
        for direction, metric, flow_index in (("injection", METRICS[0], 0),
                                              ("withdrawal", METRICS[1], 1)):
            capacity, source = _family_power_capacity(
                stores, case["experiment"], design, direction)
            capacities[direction] = capacity
            sources[direction] = source
            if _valid_capacity(capacity):
                aggregate_dispatch = np.sum([flows[s.tech][flow_index]
                                              for s in stores], axis=0)
                try:
                    curves[metric] = marginal_power_hours(
                        aggregate_dispatch, dt, capacity, bins)
                except ValueError as exc:
                    warnings.warn(f"{path.name} {FAMILY} {direction}: {exc}", stacklevel=2)
                    sources[direction] = f"invalid_power_data: {exc}"
            else:
                warnings.warn(f"{path.name} DGF: missing {direction} capacity; "
                              "see audit CSV", stacklevel=2)

    rows = [{**case, "technology": FAMILY, "metric": metric,
             "slice_index": index + 1, "capacity_percentile": centre_pct[index],
             "value": float(values[index]), "selected": selected}
            for metric, values in curves.items() for index in range(bins)]
    audit = {
        **case, "technology": FAMILY, "selected": selected,
        "variant_count": len(stores),
        "installed_storage_energy": sum(s.capacity for s in stores),
        "installed_injection_power": capacities.get("injection", float("nan")),
        "installed_withdrawal_power": capacities.get("withdrawal", float("nan")),
        "injection_capacity_source": sources.get("injection", "not_selected"),
        "withdrawal_capacity_source": sources.get("withdrawal", "not_selected"),
        "injection_profile_available": bool(np.isfinite(curves[METRICS[0]]).all()),
        "withdrawal_profile_available": bool(np.isfinite(curves[METRICS[1]]).all()),
    }
    return rows, audit


def aggregate_curves(per_case: pd.DataFrame) -> pd.DataFrame:
    """Equal-weather arithmetic mean of per-case, per-capacity-fraction curves."""
    return (per_case.groupby(
        ["country", "experiment", "technology", "metric",
         "slice_index", "capacity_percentile"], as_index=False,
    ).agg(mean_value=("value", "mean"),
          profiled_weather_scenarios=("value", "count")))


def selection_summary(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in audit.groupby("experiment", sort=False):
        selected = group.loc[group["selected"]]
        rows.append({
            "country": COUNTRY, "technology": FAMILY, "experiment": experiment,
            "selected_scenarios": len(selected),
            "total_weather_scenarios": len(group),
            "injection_profile_scenarios": int(selected["injection_profile_available"].sum()),
            "withdrawal_profile_scenarios": int(selected["withdrawal_profile_available"].sum()),
            "mean_energy_capacity_when_selected": selected["installed_storage_energy"].mean(),
            "mean_injection_capacity_when_selected": selected["installed_injection_power"].mean(),
            "mean_withdrawal_capacity_when_selected": selected["installed_withdrawal_power"].mean(),
        })
    return pd.DataFrame(rows)


def _colour(experiment: str, rank: int, n_experiments: int):
    if experiment in EXPERIMENT_COLOURS:
        return EXPERIMENT_COLOURS[experiment]
    return plt.get_cmap("plasma")(0.12 + 0.73 * rank / max(1, n_experiments - 1))


def draw_figure(average: pd.DataFrame, selection: pd.DataFrame,
                experiments: list[str], output: Path,
                *, size_mm: tuple[float, float], svg: bool = False) -> None:
    """One compact full-width 1x2 figure: DGF injection and withdrawal."""
    metrics = METRICS
    titles = ("(a) Injection", "(b) Withdrawal")
    selected_map = selection.set_index("experiment")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.transparent": False,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(size_mm[0] * MM_TO_IN,
                                                   size_mm[1] * MM_TO_IN),
                                 sharex=True)
        fig.subplots_adjust(left=0.1, right=.97, bottom=0.175,
                            top=0.8, wspace=0.25)
        colours = {experiment: _colour(experiment, i, len(experiments))
                   for i, experiment in enumerate(experiments)}
        legend_handles = []
        for experiment in experiments:
            label = EXPERIMENT_LABELS.get(experiment, experiment)
            if experiment in selected_map.index:
                counts = selected_map.loc[experiment]
                #label += f" ({int(counts['selected_scenarios'])}/{int(counts['total_weather_scenarios'])})"
            legend_handles.append(Line2D([], [], color=colours[experiment], linewidth=2.1,
                                         linestyle=EXPERIMENT_STYLES.get(experiment, "-"),
                                         label=label))
        fig.legend(handles=legend_handles, ncol=len(experiments), frameon=False,
                   loc="upper center", bbox_to_anchor=(0.54, 0.99),
                   handlelength=2.4, columnspacing=2.3)

        for ax, metric, title in zip(axes, metrics, titles):
            peak = 0.0
            for experiment in experiments:
                part = average.loc[(average["experiment"] == experiment)
                                   & (average["metric"] == metric)]
                if part.empty:
                    continue
                part = part.sort_values("capacity_percentile")
                values = part["mean_value"].to_numpy(dtype=float)
                if not np.isfinite(values).any():
                    continue
                peak = max(peak, float(np.nanmax(values)))
                ax.plot(part["capacity_percentile"].to_numpy(dtype=float), values,
                        color=colours[experiment], linewidth=2.0,
                        linestyle=EXPERIMENT_STYLES.get(experiment, "-"),
                        solid_capstyle="round", label=experiment, zorder=3)
            if peak <= 0:
                ax.text(.50, .52, "No usable DGF power profiles",
                        fontsize=9, color="0.45", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_title(title, loc="left", fontweight="bold", pad=9)
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.set_ylim(bottom=0, top=max(1.0, peak * 1.10))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=3))
            ax.grid(axis="y", color="0.88", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("0.45")
            ax.tick_params(axis="both", width=0.7, length=3)
        fig.supxlabel("Increment of installed capacity (%)",
                      fontsize=9, y=0.03)
        fig.supylabel("Marginal full-load hours/year", fontsize=9, x=0.005)
        fig.savefig(output)
        if svg:
            fig.savefig(output.with_suffix(".svg"))
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiments", nargs="+", default=list(DEFAULT_EXPERIMENTS),
                        help="NLD formulations to compare (default: B1 P1 F1 FX1)")
    parser.add_argument("--horizon", default="3y", help="Weather horizon (default 3y)")
    parser.add_argument("--bins", type=int, default=100,
                        help="Marginal capacity slices (default 100 = 1%% each)")
    parser.add_argument("--unpaired-weather", action="store_true",
                        help="Permit unequal weather samples across formulations")
    parser.add_argument("--design-csv", type=Path,
                        help="Optional installed DGF power-capacity CSV")
    parser.add_argument("--width-mm", type=float, default=FIGURE_SIZE_MM[0])
    parser.add_argument("--height-mm", type=float, default=FIGURE_SIZE_MM[1])
    parser.add_argument("--svg", action="store_true", help="Also save PowerPoint-friendly SVG")
    args = parser.parse_args()

    experiments = [str(e).upper() for e in args.experiments]
    if len(set(experiments)) != len(experiments):
        parser.error("Duplicate formulations supplied")
    if not 1 <= args.bins <= 1000:
        parser.error("--bins must be between 1 and 1000")
    if min(args.width_mm, args.height_mm) <= 0:
        parser.error("Figure dimensions must be positive")

    cases = discover_cases(args.input_dir, experiments, args.horizon,
                           common_weather=not args.unpaired_weather)
    print(f"Processing {len(cases)} {COUNTRY}/{FAMILY} cases", flush=True)
    profiles, audits = [], []
    for path, case in cases:
        print(f"Reading {path.name}", flush=True)
        try:
            rows, audit = read_case(path, case, args.bins, args.design_csv)
        except Exception as exc:
            raise RuntimeError(f"Failed to process {path.name}: {exc}") from exc
        profiles.extend(rows)
        audits.append(audit)
    per_case = pd.DataFrame.from_records(profiles)
    audit = pd.DataFrame.from_records(audits)
    average = aggregate_curves(per_case)
    selection = selection_summary(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "storage_marginal_power_DGF_NLD"
    for suffix, table in {
        "_per_case.csv": per_case,
        "_weather_means.csv": average,
        "_selection.csv": selection,
        "_audit.csv": audit,
    }.items():
        path = args.output_dir / (prefix + suffix)
        table.to_csv(path, index=False)
        print(f"Saved {path}")
    output = args.output_dir / (prefix + ".pdf")
    draw_figure(average, selection, experiments, output,
                size_mm=(args.width_mm, args.height_mm), svg=args.svg)
    print(f"Saved {output}")
    if args.svg:
        print(f"Saved {output.with_suffix('.svg')}")
    print("Selection and usable profile counts:")
    print(selection[["experiment", "selected_scenarios", "total_weather_scenarios",
                     "injection_profile_scenarios", "withdrawal_profile_scenarios"]].to_string(index=False))


if __name__ == "__main__":
    main()
