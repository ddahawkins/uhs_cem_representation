"""Compare B1 and FX1 storage portfolios within each weather window.

Run from the project root::

    pixi run python -m scripts.results_analysis.storage_portfolio_b1_fx1

Or select countries and export PNG copies::

    pixi run python -m scripts.results_analysis.storage_portfolio_b1_fx1 \
        --countries NLD ESP --png

Reads hpc/results_2h/*.nc and saves one PDF per country to results/figures/.
Each weather window has adjacent B1 and FX1 stacked bars; there is no averaging
across weather windows. Colours distinguish storage technology families.

The plotted quantity is total modelled storage_cap in TWh. It is *installed*
energy capacity, not necessarily *new* investment: existing or minimum-required
capacity can be part of storage_cap. All concrete UHS constructs are summed by
family. Batteries, CAES, pumped hydro, and other invested storage technologies
are included. Unmatched B1/FX1 weather windows are excluded with a warning,
never replaced with zero-height bars.

Requires the existing scripts/results_analysis/_plots.py helper.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from ._plots import capacity_for_tech, model_case, storage_family
except ImportError:  # Support running directly beside _plots.py.
    from _plots import capacity_for_tech, model_case, storage_family


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / "results_2h_no_coupling_bounds_no_lrc"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

EXPERIMENTS = ("B1", "FX1")
TECHNOLOGY_ORDER = (
    "battery_2h", "battery_4h", "battery_8h",
    "compressed_air_storage", "pumped_hydro_storage",
    "lined_rock_cavern", "salt_cavern", "aquifer", "depleted_gas_field",
)
TECH_LABELS = {
    "battery_2h": "Battery 2h",
    "battery_4h": "Battery 4h",
    "battery_8h": "Battery 8h",
    "compressed_air_storage": "CAES",
    "pumped_hydro_storage": "PHS",
    "lined_rock_cavern": "LRC",
    "salt_cavern": "SC",
    "aquifer": "SA",
    "depleted_gas_field": "DGF",
}
PHS_IDS = {"pumped_hydro", "pumped_hydro_storage", "pumped_storage"}
PLASMA = plt.get_cmap("plasma")
# Consistent technology colours across countries, even if a country omits a
# technology completely. Keep colours inside the 0.0–0.9 plasma interval.
TECH_COLORS = dict(zip(
    TECHNOLOGY_ORDER,
    PLASMA(np.linspace(0.06, 0.90, len(TECHNOLOGY_ORDER))),
))


def _storage_family(tech_id: str) -> str | None:
    """Match known storage techs, including batteries omitted by _plots."""
    tech_id = tech_id.lower()
    if re.fullmatch(r"battery_\d+h", tech_id):
        return tech_id
    if tech_id in PHS_IDS:
        return "pumped_hydro_storage"
    return storage_family(tech_id)


def _weather_key(weather: str) -> tuple[int, str]:
    match = re.fullmatch(r"weather_(\d{4})_(.+)", weather.lower())
    return (int(match.group(1)), match.group(2)) if match else (9999, weather)


def _weather_label(weather: str) -> str:
    match = re.fullmatch(r"weather_(\d{4})_(.+)", weather.lower())
    if match:
        return f"{match.group(1)} ({match.group(2)})"
    return weather


def read_portfolio(path: Path) -> dict[str, object]:
    """Read one model's complete installed storage energy portfolio in TWh."""
    import xarray as xr

    case = model_case(path)
    if case["experiment"].upper() not in EXPERIMENTS:
        raise ValueError(f"Not a B1/FX1 result: {path.name}")

    capacities: dict[str, float] = {}
    with xr.open_dataset(path, group="results") as ds:
        if "techs" not in ds.coords or "storage_cap" not in ds:
            raise ValueError(f"No storage_cap with techs coordinate in {path.name}")
        for raw_id in ds.coords["techs"].values:
            tech_id = str(raw_id)
            capacity_mwh = capacity_for_tech(ds, "storage_cap", tech_id)
            if capacity_mwh < -1e-5:
                raise ValueError(f"Negative capacity for {tech_id} in {path.name}")
            if capacity_mwh < 0:
                capacity_mwh = 0.0  # Clamp insignificant numerical noise.

            family = _storage_family(tech_id)
            if family is None:
                if capacity_mwh <= 1e-6:
                    continue
                # Do not silently omit any other technology with storage_cap.
                family = f"other:{tech_id}"
                warnings.warn(
                    f"Unrecognised storage technology {tech_id!r} in {path.name}; "
                    "plotting it separately.", stacklevel=2,
                )
            capacities[family] = capacities.get(family, 0.0) + capacity_mwh / 1e6

    return {
        "country": case["country"].upper(),
        "experiment": case["experiment"].upper(),
        "weather": case["weather"].lower(),
        "case": case["case"],
        "capacities": capacities,
    }


def load_portfolios(
    input_dir: Path, *, countries: list[str] | None = None,
) -> list[dict[str, object]]:
    """Collect B1/FX1 case portfolios, rejecting duplicate weather results."""
    files = sorted(
        p for p in input_dir.expanduser().glob("*.nc")
        if not p.name.endswith(".pending.nc")
    )
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {input_dir}")

    requested = {country.upper() for country in countries} if countries else None
    portfolios: list[dict[str, object]] = []
    observed: set[tuple[str, str, str]] = set()
    for path in files:
        # Quick prefilter avoids opening the many P1/F1/FX2/etc. experiments.
        if not re.match(r"^[A-Za-z]{3}_(?:B1|FX1)_", path.stem, re.I):
            continue
        try:
            case = model_case(path)
            if requested and case["country"].upper() not in requested:
                continue
            print(f"Reading {path.name}", flush=True)
            portfolio = read_portfolio(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {exc}", stacklevel=2)
            continue
        key = (
            str(portfolio["country"]), str(portfolio["experiment"]),
            str(portfolio["weather"]),
        )
        if key in observed:
            raise ValueError(f"Duplicate country/experiment/weather result: {key}")
        observed.add(key)
        portfolios.append(portfolio)

    if not portfolios:
        raise ValueError("No readable B1 or FX1 NetCDF results found")
    if requested:
        found = {str(portfolio["country"]) for portfolio in portfolios}
        if missing := requested - found:
            warnings.warn(f"No B1/FX1 results for {sorted(missing)}", stacklevel=2)
    return portfolios


def _paired_weather(
    country_cases: list[dict[str, object]], country: str,
) -> tuple[list[str], dict[tuple[str, str], dict[str, object]]]:
    cases = {
        (str(case["weather"]), str(case["experiment"])): case
        for case in country_cases
    }
    b1 = {weather for weather, experiment in cases if experiment == "B1"}
    fx1 = {weather for weather, experiment in cases if experiment == "FX1"}
    if unpaired := sorted(b1.symmetric_difference(fx1), key=_weather_key):
        warnings.warn(
            f"{country}: skipping weather windows without both B1 and FX1: "
            f"{', '.join(unpaired)}", stacklevel=2,
        )
    shared = sorted(b1.intersection(fx1), key=_weather_key)
    if not shared:
        warnings.warn(f"{country}: no paired B1/FX1 windows; no figure", stacklevel=2)
    return shared, cases


def plot_country(
    country: str,
    country_cases: list[dict[str, object]],
    output_dir: Path,
    *,
    png: bool = False,
) -> tuple[Path, ...]:
    """One country; adjacent B1/FX1 bars within each weather group."""
    weather_order, case_lookup = _paired_weather(country_cases, country)
    if not weather_order:
        return ()

    present = {
        tech for case in country_cases for tech in case["capacities"]
    }
    technologies = [tech for tech in TECHNOLOGY_ORDER if tech in present]
    technologies.extend(sorted(present - set(TECHNOLOGY_ORDER)))
    colors = {tech: TECH_COLORS.get(tech, PLASMA(0.90)) for tech in technologies}

    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig_width = max(8.1, 1.55 * len(weather_order) + 1.6)
        fig, ax = plt.subplots(figsize=(fig_width, 5.25))
        fig.subplots_adjust(left=0.10, right=0.975, bottom=0.225, top=0.725)

        bar_width = 0.32
        offset = 0.19
        totals: dict[tuple[str, str], float] = {}
        max_total = 0.0

        for i, weather in enumerate(weather_order):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="0.975", zorder=0)
            for experiment, side in (("B1", -1), ("FX1", +1)):
                portfolio = case_lookup[(weather, experiment)]
                capacity_by_tech = portfolio["capacities"]
                x_pos = i + side * offset
                bottom = 0.0
                for tech in technologies:
                    height = float(capacity_by_tech.get(tech, 0.0))
                    if height <= 0:
                        continue
                    ax.bar(
                        x_pos, height, bottom=bottom, width=bar_width,
                        facecolor=colors[tech], edgecolor="white",
                        linewidth=0.30, zorder=3,
                    )
                    bottom += height
                totals[(weather, experiment)] = bottom
                max_total = max(max_total, bottom)

        # Small total labels help distinguish changes in overall investments.
        max_total = max(max_total, 1e-6)
        for i, weather in enumerate(weather_order):
            for experiment, side in (("B1", -1), ("FX1", +1)):
                total = totals[(weather, experiment)]
                if total <= 0:
                    continue
                label = f"{total:.2f}" if total < 1 else f"{total:.1f}"
                ax.text(
                    i + side * offset, total + max_total * 0.018, label,
                    fontsize=8, ha="center", va="bottom", color="0.25",
                )

        # The representations are the inner x-axis; weather windows are outer.
        tick_positions = [
            i + side * offset
            for i in range(len(weather_order)) for side in (-1, +1)
        ]
        ax.set_xticks(tick_positions, ["B1", "FX1"] * len(weather_order))
        ax.tick_params(axis="x", length=0, pad=5, labelsize=10)
        for i, weather in enumerate(weather_order):
            ax.text(
                i, -0.16, _weather_label(weather),
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, clip_on=False,
            )
            if i < len(weather_order) - 1:
                ax.axvline(i + 0.5, color="0.82", linewidth=0.8, zorder=1)

        ax.set_xlim(-0.55, len(weather_order) - 0.45)
        ax.set_ylim(0, max_total * 1.18)
        ax.set_ylabel("Installed storage energy capacity (TWh)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("0.55")
        ax.grid(axis="y", color="0.85", linewidth=0.65, alpha=0.65)
        ax.set_axisbelow(True)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[tech], edgecolor="none",
                          label=TECH_LABELS.get(tech, tech.replace("other:", "").replace("_", " ")))
            for tech in technologies
        ]
        if legend_handles:
            # Matplotlib fills the legend down columns. Reorder the handles so
            # readers see batteries -> CAES/PHS -> UHS from left to right.
            ncols = min(5, len(legend_handles))
            rows = [legend_handles[i:i + ncols]
                    for i in range(0, len(legend_handles), ncols)]
            row_major = [row[j] for j in range(ncols) for row in rows
                         if j < len(row)]
            fig.legend(
                handles=row_major, loc="upper center",
                bbox_to_anchor=(0.54, 0.95), ncol=ncols,
                frameon=False, fontsize=10, handlelength=1.25,
                columnspacing=1.4, labelspacing=0.8,
            )
        fig.text(0.10, 0.745, country, fontsize=11, weight="semibold")

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"storage_portfolio_{country}_b1_vs_fx1.pdf"
        fig.savefig(path)
        outputs = [path]
        if png:
            png_path = path.with_suffix(".png")
            fig.savefig(png_path, dpi=300)
            outputs.append(png_path)
        plt.close(fig)

    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="+", default=None,
                        help="ISO3 country codes; omitted means every detected country")
    parser.add_argument("--png", action="store_true",
                        help="Also export each figure as a 300-dpi PNG")
    args = parser.parse_args()

    portfolios = load_portfolios(args.input_dir, countries=args.countries)
    detected = {str(case["country"]) for case in portfolios}
    countries = (
        [country.upper() for country in args.countries if country.upper() in detected]
        if args.countries else sorted(detected)
    )
    for country in dict.fromkeys(countries):
        country_cases = [case for case in portfolios if case["country"] == country]
        for output in plot_country(country, country_cases, args.output_dir, png=args.png):
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
