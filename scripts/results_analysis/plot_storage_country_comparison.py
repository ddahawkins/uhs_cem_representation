"""Compare B1 and FX1 storage investments across countries and weather windows.

Run from the project root::

    pixi run python -m scripts.results_analysis.plot_storage_country_comparison `
    --countries NLD ESP

Default input:  hpc/results_2h/*.nc
Default output: results/figures/storage_capacity_by_country.pdf

This is an energy-capacity comparison (TWh), not a cost or withdrawal-power
comparison. Batteries (2/4/8 h), CAES and all four UHS families are included.
Pumped hydro is added automatically if it appears in the model results.

Each weather scenario is one marker, with NO weather-specific horizontal
jitter. B1 and FX1 have fixed offsets within each technology category, and a
black horizontal segment shows the median over the available weather cases.
Individual geological variants are summed into their family for each run.

An installed capacity of zero is plotted as zero when the technology is
present in the model's results. Technologies absent altogether are marked
unavailable; a missing country/experiment/weather file is never a zero.
"""

from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    from ._plots import capacity_for_tech, model_case, storage_family
except ImportError:  # Allow running the file directly beside _plots.py.
    from _plots import capacity_for_tech, model_case, storage_family


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / "results_2h_no_coupling_bounds_no_lrc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

EXPERIMENTS = ("B1", "FX1")

TECHNOLOGY_ORDER = (
    "battery_2h",
    "battery_4h",
    "battery_8h",
    "compressed_air_storage",
    "lined_rock_cavern",
    "salt_cavern",
    "aquifer",
    "depleted_gas_field",
)
TECHNOLOGY_SHORT = {
    "battery_2h": "Bat\n2h",
    "battery_4h": "Bat\n4h",
    "battery_8h": "Bat\n8h",
    "compressed_air_storage": "CAES",
    "lined_rock_cavern": "LRC",
    "salt_cavern": "SC",
    "aquifer": "SA",
    "depleted_gas_field": "DGF",
    "pumped_hydro_storage": "PHS",
}
PHS_IDS = {"pumped_hydro", "pumped_hydro_storage", "pumped_storage"}
WEATHER_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h", "<", ">")
PLASMA = plt.get_cmap("plasma")
# Retain positions in the established B1, P1, F1, FX1, FX2, FX4 progression.
EXPERIMENT_COLORS = {"B1": PLASMA(0.10), "FX1": PLASMA(0.58)}


def _weather_key(weather: str) -> tuple[int, str]:
    match = re.fullmatch(r"weather_(\d{4})_(.+)", weather.lower())
    return (int(match.group(1)), match.group(2)) if match else (9999, weather)


def _weather_label(weather: str) -> str:
    match = re.fullmatch(r"weather_(\d{4})_(.+)", weather.lower())
    return f"{match.group(1)} ({match.group(2)})" if match else weather


def _technology_family(tech_id: str) -> str | None:
    """Translate an installed concrete technology ID to a plot category."""
    identity = tech_id.lower()
    if identity in TECHNOLOGY_ORDER:
        return identity
    if identity in PHS_IDS:
        return "pumped_hydro_storage"
    return storage_family(identity)


def read_case_capacities(path: Path) -> pd.DataFrame:
    """Read every storage category, distinguishing genuine zero from absent."""
    import xarray as xr

    details = model_case(path)
    if details["experiment"].upper() not in EXPERIMENTS:
        return pd.DataFrame()

    with xr.open_dataset(path, group="results") as results:
        if "techs" not in results.coords or "storage_cap" not in results:
            raise ValueError(f"Missing storage results in {path.name}")

        capacity_by_family: dict[str, float] = {}
        for raw_id in results.coords["techs"].values:
            tech_id = str(raw_id)
            family = _technology_family(tech_id)
            if family is None:
                continue
            installed = capacity_for_tech(results, "storage_cap", tech_id)
            if installed < -1e-5:
                raise ValueError(f"Negative storage capacity for {tech_id}: {installed}")
            capacity_by_family[family] = (
                capacity_by_family.get(family, 0.0) + max(0.0, installed)
            )

    all_categories = (*TECHNOLOGY_ORDER, "pumped_hydro_storage")
    return pd.DataFrame([
        {
            "country": details["country"].upper(),
            "experiment": details["experiment"].upper(),
            "weather": details["weather"].lower(),
            "case": details["case"],
            "technology": tech,
            "available": tech in capacity_by_family,
            "capacity_TWh": (
                capacity_by_family[tech] / 1e6
                if tech in capacity_by_family else float("nan")
            ),
        }
        for tech in all_categories
    ])


def load_comparison(
    input_dir: Path,
    countries: list[str] | None = None,
    *,
    common_weather: bool = False,
) -> pd.DataFrame:
    """Load B1/FX1 runs, keeping missing experiments distinct from zero MW."""
    input_dir = input_dir.expanduser()
    files = sorted(
        path for path in input_dir.glob("*.nc")
        if not path.name.endswith(".pending.nc")
    )
    if not files:
        raise FileNotFoundError(f"No .nc results found in {input_dir}")

    requested_countries = {c.upper() for c in countries} if countries else None
    all_frames: list[pd.DataFrame] = []
    for path in files:
        try:
            case = model_case(path)
            if case["experiment"].upper() not in EXPERIMENTS:
                continue
            if requested_countries is not None and case["country"].upper() not in requested_countries:
                continue
            print(f"Reading {path.name}", flush=True)
            frame = read_case_capacities(path)
            if not frame.empty:
                all_frames.append(frame)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {type(exc).__name__}: {exc}", stacklevel=2)

    if not all_frames:
        raise ValueError("No readable B1 or FX1 .nc files were found")

    combined = pd.concat(all_frames, ignore_index=True)
    case_counts = combined[["country", "experiment", "weather", "case"]].drop_duplicates()
    if case_counts.duplicated(["country", "experiment", "weather"]).any():
        raise ValueError("Duplicate country/experiment/weather result files")
    cases = case_counts[["country", "experiment", "weather"]]

    detected = set(combined["country"])
    if requested_countries is not None and requested_countries - detected:
        warnings.warn(
            f"No readable B1/FX1 results for {sorted(requested_countries - detected)}",
            stacklevel=2,
        )

    coverage = {
        (country, experiment): set(cases.loc[
            (cases["country"] == country) & (cases["experiment"] == experiment),
            "weather",
        ])
        for country in sorted(detected) for experiment in EXPERIMENTS
    }
    missing_groups = [key for key, weather in coverage.items() if not weather]
    if missing_groups:
        warnings.warn(f"Missing country/experiment groups: {missing_groups}", stacklevel=2)
    nonempty = [weather for weather in coverage.values() if weather]
    if common_weather:
        if missing_groups:
            raise ValueError("--common-weather requires B1 and FX1 for every country")
        shared = set.intersection(*nonempty)
        if not shared:
            raise ValueError("No common weather windows across all country/experiment groups")
        combined = combined.loc[combined["weather"].isin(shared)].copy()
        print(f"Using {len(shared)} shared weather windows: {sorted(shared, key=_weather_key)}")
    elif len({frozenset(weather) for weather in nonempty}) > 1:
        warnings.warn(
            "Weather coverage differs between countries/representations. "
            "Medians use available runs; use --common-weather for a paired comparison.",
            stacklevel=2,
        )

    # PHS is optional; do not reserve an empty x-category when no file offers it.
    if not combined.loc[
        combined["technology"] == "pumped_hydro_storage", "available",
    ].any():
        combined = combined.loc[
            combined["technology"] != "pumped_hydro_storage"
        ].copy()
    return combined


def plot_comparison(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    country_order: list[str] | None = None,
    y_scale: str = "linear",
    png: bool = False,
) -> tuple[Path, ...]:
    """Country -> technology -> representation, with aligned weather markers."""
    if frame.empty:
        raise ValueError("No storage capacity results to plot")
    if y_scale not in {"linear", "symlog"}:
        raise ValueError("y_scale must be 'linear' or 'symlog'")
    if (frame.loc[frame["available"], "capacity_TWh"] < 0).any():
        raise ValueError("Installed storage capacities must be non-negative")

    countries = (
        [country for country in country_order if country in set(frame["country"])]
        if country_order else sorted(frame["country"].unique())
    )
    technologies = [
        tech for tech in (*TECHNOLOGY_ORDER, "pumped_hydro_storage")
        if tech in set(frame["technology"])
    ]
    weather_order = sorted(set(frame["weather"]), key=_weather_key)
    weather_shapes = {
        weather: WEATHER_MARKERS[index % len(WEATHER_MARKERS)]
        for index, weather in enumerate(weather_order)
    }

    n_tech = len(technologies)
    country_span = n_tech + 1.0
    fig_width = max(12.4, 0.78 * n_tech * len(countries) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.27, top=0.81)

    positions: dict[tuple[str, str], float] = {}
    tick_positions, tick_labels = [], []
    for c_index, country in enumerate(countries):
        start = c_index * country_span
        stop = start + n_tech - 1
        if c_index % 2:
            ax.axvspan(start - 0.5, stop + 0.5, color="0.97", zorder=0)
        if c_index:
            ax.axvline(start - 1.0, color="0.75", linewidth=0.8, zorder=0)
        for t_index, tech in enumerate(technologies):
            x = start + t_index
            positions[(country, tech)] = x
            tick_positions.append(x)
            tick_labels.append(TECHNOLOGY_SHORT[tech])
        midpoint = (start + stop) / 2
        ax.text(midpoint, -0.20, country, ha="center", va="top",
                fontsize=11, weight="semibold",
                transform=ax.get_xaxis_transform(), clip_on=False)
        ax.plot([start - 0.35, stop + 0.35], [-0.14, -0.14],
                color="0.65", linewidth=0.75,
                transform=ax.get_xaxis_transform(), clip_on=False)

    marker_count = 0
    median_count = 0
    for country in countries:
        for tech in technologies:
            x_centre = positions[(country, tech)]
            for experiment, offset in (("B1", -0.19), ("FX1", 0.19)):
                slot = frame.loc[
                    (frame["country"] == country)
                    & (frame["technology"] == tech)
                    & (frame["experiment"] == experiment)
                    & frame["available"]
                ]
                x = x_centre + offset
                if slot.empty:
                    # No technology in any available run; do not invent zero.
                    continue
                for _, row in slot.iterrows():
                    ax.scatter(
                        x, float(row["capacity_TWh"]),
                        marker=weather_shapes[str(row["weather"])],
                        color=EXPERIMENT_COLORS[experiment],
                        s=48, alpha=0.85, edgecolors="0.25",
                        linewidths=0.45, zorder=4,
                    )
                    marker_count += 1

                median = float(slot["capacity_TWh"].median())
                ax.plot([x - 0.12, x + 0.12], [median, median],
                        color="black", linewidth=2.3,
                        solid_capstyle="butt", zorder=6)
                median_count += 1

    ax.set_xticks(tick_positions, tick_labels)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=9)
    ax.tick_params(axis="y", labelsize=10, colors="0.25")
    ax.set_xlim(-0.65, (len(countries) - 1) * country_span + n_tech - 0.35)
    ax.set_ylabel("Installed storage energy capacity (TWh)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.55")
    ax.grid(axis="y", color="0.85", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if y_scale == "symlog":
        ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
        ax.set_ylim(bottom=0)
    else:
        maximum = float(frame["capacity_TWh"].max())
        maximum = maximum if math.isfinite(maximum) and maximum > 0 else 0.01
        ax.set_ylim(-maximum * 0.04, maximum * 1.10)

    rep_handles = [
        Line2D([], [], color="none", linestyle="none", marker="o", markersize=7,
               markerfacecolor=EXPERIMENT_COLORS[experiment],
               markeredgecolor="0.3", label=experiment)
        for experiment in EXPERIMENTS
    ]
    weather_handles = [
        Line2D([], [], color="none", linestyle="none", marker=weather_shapes[weather],
               markerfacecolor="0.6", markeredgecolor="0.3", markersize=7,
               label=_weather_label(weather))
        for weather in weather_order
    ]
    fig.text(0.08, 0.945, "Representation", fontsize=10,
             weight="semibold", va="center")
    fig.legend(handles=rep_handles, loc="center left", bbox_to_anchor=(0.22, 0.945),
               ncol=2, frameon=False, fontsize=10,
               handlelength=1.0, columnspacing=1.3)
    fig.text(0.08, 0.885, "Weather window", fontsize=10,
             weight="semibold", va="center")
    fig.legend(handles=weather_handles, loc="center left",
               bbox_to_anchor=(0.22, 0.885),
               ncol=min(8, len(weather_handles)), frameon=False, fontsize=9,
               handlelength=1.0, columnspacing=1.25)
    # An explicit key for the black summary bars.
    fig.text(0.83, 0.945, "Black bars: median", fontsize=9,
             color="0.2", va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    outputs = [output_path]
    if png:
        png_path = output_path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        outputs.append(png_path)
    plt.close(fig)
    print(
        f"Plotted {marker_count} weather/representation/category values and "
        f"{median_count} median bars across {len(countries)} countries."
    )
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="+", default=None,
                        help="Optional ISO3 country selection and display order")
    parser.add_argument("--common-weather", action="store_true",
                        help="Compare only weather windows found for every country and B1/FX1")
    parser.add_argument("--y-scale", choices=("linear", "symlog"), default="linear",
                        help="Optional symlog scale for batteries and larger UHS on one axis")
    parser.add_argument("--png", action="store_true",
                        help="Also save a 300-dpi PNG copy (PDF is always written)")
    args = parser.parse_args()

    frame = load_comparison(
        args.input_dir, countries=args.countries, common_weather=args.common_weather,
    )
    countries = [c.upper() for c in args.countries] if args.countries else None
    output_path = args.output_dir / "storage_capacity_by_country.pdf"
    for path in plot_comparison(
        frame, output_path, country_order=countries,
        y_scale=args.y_scale, png=args.png,
    ):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
