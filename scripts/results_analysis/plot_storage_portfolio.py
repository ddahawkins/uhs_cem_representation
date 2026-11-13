"""Plot B1/FX1 storage portfolios in a compact four-country journal figure.

Default: a 90 x 185 mm, single-column, four-row PDF (NLD, ESP, DEU, GBR).

    pixi run python -m scripts.results_analysis.plot_storage_portfolio

Options:
    --countries NLD ESP DEU GBR
    --input-dir PATH
    --output-dir PATH
    --show-totals       add numerical bar totals (may be crowded at 90 mm)
    --separate          optional individual-country full-width PDFs
    --png               also save 300-dpi PNGs

Each three-year April--March weather sample has adjacent B (= B1) and FX (= FX1)
stacks. The outer sample labels (e.g. 'Apr 2009') appear only below the last
panel; B/FX appears in EVERY country panel. No pooling or averaging of samples.
Axes are independent (neither sharex nor sharey), but weather-group locations
are aligned across countries. Missing country/weather pairs are marked 'n/a'
rather than plotted as zero.

Fixed / non-expandable hydro storage (combined reservoirs and pumped storage)
is excluded from stacked bars, but its total installed energy capacity is
annotated separately in each country panel. Hydro technology identifiers are
listed in HYDRO_STORAGE_IDS; add any additional identifiers used in the model.
All other recognized storages are assumed to have no pre-existing capacity.

The figures use Plasma for expandable technologies, grey for fixed-hydro text,
and 9-pt lettering where feasible. Output is a vector PDF by default.
Requires scripts/results_analysis/_plots.py from the existing project.
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
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

try:
    from ._plots import capacity_for_tech, model_case, storage_family
except ImportError:  # Also support running directly beside _plots.py.
    from _plots import capacity_for_tech, model_case, storage_family


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "capacity_expansion_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "storage_portfolio"
DEFAULT_COUNTRIES = ("NLD", "ESP", "DEU", "GBR")
FIGURE_SIZE_MM = (90, 220)  # Full-height single-column; caption is external.
MM_TO_INCH = 1 / 25.4
WEATHER_START_MONTH = "Apr"
EXPERIMENTS = ("B1", "FX1")
EXPERIMENT_LABELS = {"B1": "B", "FX1": "FX"}

# Only expandable/model-selected storage belongs in the coloured bars.
TECHNOLOGY_ORDER = (
    "battery_2h", "battery_4h", "battery_8h", "compressed_air_storage",
    "lined_rock_cavern", "salt_cavern", "aquifer", "depleted_gas_field",
)
TECH_LABELS = {
    "battery_2h": "Battery 2h",
    "battery_4h": "Battery 4h",
    "battery_8h": "Battery 8h",
    "compressed_air_storage": "CAES",
    "lined_rock_cavern": "LRC",
    "salt_cavern": "SC",
    "aquifer": "SA",
    "depleted_gas_field": "DGF",
}

# Recognize either the merged hydrostorage tech or the older separate IDs.
# If a model has both pumped and reservoir storage, they are summed here.
HYDRO_STORAGE_IDS = {
    "hydrostorage", "hydro_storage", "hydro-storage",
    "hydro_reservoir", "hydro_reservoir_storage", "reservoir_hydro",
    "pumped_hydro", "pumped_hydro_storage", "pumped_storage",
}
PLASMA = plt.get_cmap("plasma")
TECH_COLORS = {
    tech: PLASMA(x)
    for tech, x in zip(TECHNOLOGY_ORDER, np.linspace(0.06, 0.90, len(TECHNOLOGY_ORDER)))
}


def _is_fixed_hydro(tech_id: str) -> bool:
    """Whether an ID denotes non-expandable hydro storage in this study."""
    return tech_id.lower() in HYDRO_STORAGE_IDS


def _storage_family(tech_id: str) -> str | None:
    """Map expandable technologies; hydro is handled BEFORE this call."""
    tech_id = tech_id.lower()
    if re.fullmatch(r"battery_\d+h", tech_id):
        return tech_id
    return storage_family(tech_id)


def _weather_key(weather: str) -> tuple[int, str]:
    match = re.fullmatch(r"weather_(\d{4})_(.+)", weather.lower())
    return (int(match.group(1)), match.group(2)) if match else (9999, weather)


def _weather_label_short(weather: str) -> str:
    """3-year windows: 'Apr\n2009' represents April 2009--March 2012."""
    match = re.fullmatch(r"weather_(\d{4})_(\d+)([ym])", weather.lower())
    if not match:
        return weather.replace("weather_", "")
    year, duration, unit = int(match.group(1)), int(match.group(2)), match.group(3)
    if duration != 3 or unit != "y":
        warnings.warn(
            f"Weather window {weather!r} is not 3y; labels give start date only. "
            "Check the figure caption before publication.", stacklevel=2,
        )
    return f"{year}–{year+3-2000}"


def read_portfolio(path: Path) -> dict[str, object]:
    """Read installed energy capacity in TWh; separate fixed hydro from bars."""
    import xarray as xr

    case = model_case(path)
    if case["experiment"].upper() not in EXPERIMENTS:
        raise ValueError(f"Not a B1/FX1 result: {path.name}")

    capacities: dict[str, float] = {}
    hydro_twh = 0.0
    with xr.open_dataset(path, group="results") as ds:
        if "techs" not in ds.coords or "storage_cap" not in ds:
            raise ValueError(f"No storage_cap with techs coordinate in {path.name}")
        for raw_id in ds.coords["techs"].values:
            tech_id = str(raw_id)
            capacity_mwh = capacity_for_tech(ds, "storage_cap", tech_id)
            if capacity_mwh < -1e-5:
                raise ValueError(f"Negative capacity for {tech_id} in {path.name}")
            capacity_mwh = max(0.0, capacity_mwh)
            if _is_fixed_hydro(tech_id):
                hydro_twh += capacity_mwh / 1e6
                continue
            family = _storage_family(tech_id)
            if family is None or family not in TECH_COLORS:
                if capacity_mwh > 1e-6:
                    warnings.warn(
                        f"Unrecognised storage technology {tech_id!r} in {path.name}; "
                        "skipping. Check TECHNOLOGY_ORDER or HYDRO_STORAGE_IDS.",
                        stacklevel=2,
                    )
                continue
            capacities[family] = capacities.get(family, 0.0) + capacity_mwh / 1e6

    return {
        "country": case["country"].upper(),
        "experiment": case["experiment"].upper(),
        "weather": case["weather"].lower(),
        "case": case["case"],
        "capacities": capacities,
        "hydro_storage_twh": hydro_twh,
    }


def load_portfolios(
    input_dir: Path, *, countries: list[str] | None = None,
) -> list[dict[str, object]]:
    """Collect B1/FX1 portfolios, rejecting duplicate country/weather results."""
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
            f"{country}: missing a B1 or FX1 result for: "
            f"{', '.join(unpaired)} (will mark n/a, not zero)", stacklevel=2,
        )
    shared = sorted(b1.intersection(fx1), key=_weather_key)
    if not shared:
        warnings.warn(f"{country}: no paired B1/FX1 windows", stacklevel=2)
    return shared, cases


def _format_twh(value: float) -> str:
    """Keep small hydro capacities distinguishable from zero."""
    if abs(value) < 1e-8:
        return "0"
    if abs(value) >= 1:
        return f"{value:.1f}"
    if abs(value) >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _hydro_annotation(country: str, cases: list[dict[str, object]]) -> str:
    """Report fixed hydro once, or show a range if results unexpectedly differ."""
    values = [float(case.get("hydro_storage_twh", 0.0)) for case in cases]
    if not values:
        return "Fixed hydro: n/a"
    lowest, highest = min(values), max(values)
    if not np.isclose(lowest, highest, rtol=1e-3, atol=1e-4):
        warnings.warn(
            f"{country}: hydro storage differs among cases "
            f"({lowest:.5g}--{highest:.5g} TWh); displaying a range.",
            stacklevel=2,
        )
        return f"Hydro: {_format_twh(lowest)}–{_format_twh(highest)} TWh"
    return f"Fixed hydro: {_format_twh(float(np.median(values)))} TWh"


def _country_order(countries: list[str] | tuple[str, ...]) -> list[str]:
    """NLD first, retain requested order for remaining countries."""
    unique = list(dict.fromkeys(c.upper() for c in countries))
    return (["NLD"] if "NLD" in unique else []) + [c for c in unique if c != "NLD"]


def _draw_country_panel(
    ax: plt.Axes,
    country: str,
    country_cases: list[dict[str, object]],
    weather_order: list[str],
    case_lookup: dict[tuple[str, str], dict[str, object]],
    *,
    letter: str,
    show_weather_labels: bool,
    show_totals: bool,
) -> set[str]:
    """Draw independently scaled country panel, always retaining B/FX ticks."""
    ax.set_title(f"({letter})  {country}" if letter else country,
                 loc="left", fontsize=10, weight="bold", pad=5)
    ax.text(1.0, 1.045, _hydro_annotation(country, country_cases),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.40", clip_on=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("0.50")
    ax.tick_params(axis="y", labelsize=9, colors="0.20", length=2.5, pad=2)
    ax.tick_params(axis="x", labelsize=8, length=0, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.set_axisbelow(True)
    # Only light vertical window separators: no horizontal grid or shading.
    for i in range(len(weather_order) - 1):
        ax.axvline(i + 0.5, color="0.84", lw=0.55, zorder=0)

    bar_width, offset = 0.32, 0.205
    used: set[str] = set()
    max_total = 0.0
    totals: dict[tuple[str, str], float] = {}
    available_windows = 0

    for i, weather in enumerate(weather_order):
        if any((weather, experiment) not in case_lookup for experiment in EXPERIMENTS):
            ax.text(i, 0.10, "n/a", ha="center", va="bottom", fontsize=8,
                    color="0.45", transform=ax.get_xaxis_transform())
            continue
        available_windows += 1
        for experiment, side in (("B1", -1), ("FX1", +1)):
            case = case_lookup[(weather, experiment)]
            capacities = case["capacities"]
            x_pos = i + side * offset
            bottom = 0.0
            for tech in TECHNOLOGY_ORDER:
                height = float(capacities.get(tech, 0.0))
                if height <= 1e-12:
                    continue
                used.add(tech)
                ax.bar(x_pos, height, bottom=bottom, width=bar_width,
                       facecolor=TECH_COLORS[tech], edgecolor="white",
                       linewidth=0.25, zorder=3)
                bottom += height
            totals[(weather, experiment)] = bottom
            max_total = max(max_total, bottom)

    # Each axis is independent: B and FX labels appear in all four panels.
    tick_positions = [i + side * offset
                      for i in range(len(weather_order)) for side in (-1, +1)]
    tick_labels = [EXPERIMENT_LABELS[experiment]
                   for _ in weather_order for experiment in EXPERIMENTS]
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlim(-0.52, max(0.5, len(weather_order) - 0.48))

    if show_weather_labels:
        # The two-line start-date labels avoid overlap at 90 mm width.
        for i, weather in enumerate(weather_order):
            ax.text(i, -0.2, _weather_label_short(weather),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=8, color="0.20", clip_on=False)

    if max_total <= 1e-10:
        ax.set_ylim(0, 1)
        if available_windows:
            ax.text(0.5, 0.54, "No model-selected storage", ha="center", va="center",
                    transform=ax.transAxes, color="0.40", fontsize=8)
        ax.set_yticks([0, 1])
    else:
        ax.set_ylim(0, max_total * (1.2 if show_totals else 1.07))
        if show_totals:
            for i, weather in enumerate(weather_order):
                for experiment, side in (("B1", -1), ("FX1", +1)):
                    total = totals.get((weather, experiment), 0.0)
                    if total <= 1e-10:
                        continue
                    text = f"{total:.2f}" if total < 1 else f"{total:.1f}"
                    ax.annotate(text, xy=(i + side * offset, total), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=8, color="0.25")
    # Use readable 0.1 TWh increments for Spain.
    if country == "ESP" and max_total > 1e-10:
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.2))
    return used


def _legend_handles(used: set[str]) -> list[Patch]:
    return [Patch(facecolor=TECH_COLORS[t], edgecolor="none", label=TECH_LABELS[t])
            for t in TECHNOLOGY_ORDER if t in used]


def _legend_row_major(handles: list[Patch], ncols: int) -> list[Patch]:
    """Matplotlib fills legend columns first; reorder for intuitive row reading."""
    nrows = (len(handles) + ncols - 1) // ncols
    rows = [handles[i * ncols:(i + 1) * ncols] for i in range(nrows)]
    return [row[j] for j in range(ncols) for row in rows if j < len(row)]


def _save(fig: plt.Figure, path: Path, *, png: bool) -> tuple[Path, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")  # No bbox_inches='tight': keep mm dimensions.
    outputs = [path]
    if png:
        png_path = path.with_suffix(".png")
        fig.savefig(png_path, dpi=300, facecolor="white")
        outputs.append(png_path)
    plt.close(fig)
    return tuple(outputs)


def plot_combined(
    portfolios: list[dict[str, object]],
    countries: list[str],
    output_dir: Path,
    *,
    png: bool = False,
    show_totals: bool = False,
) -> tuple[Path, ...]:
    """Create vertically stacked, 90 x 185 mm single-column comparison."""
    countries = _country_order(countries)
    if not countries:
        return ()
    if len(countries) > 4:
        raise ValueError("Combined layout supports up to four countries; use --separate")

    # Align weather positions, but DO NOT share Matplotlib x-axes (or y-axes).
    # This also prevents mismatched country samples from being mislabeled.
    by_country: dict[str, list[dict[str, object]]] = {
        country: [case for case in portfolios if case["country"] == country]
        for country in countries
    }
    lookups: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    all_weather: set[str] = set()
    for country, cases in by_country.items():
        available, lookup = _paired_weather(cases, country)
        lookups[country] = lookup
        all_weather.update(available)
    weather_order = sorted(all_weather, key=_weather_key)
    if not weather_order:
        raise ValueError("No paired B1/FX1 weather windows for selected countries")

    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    }):
        fig, axes = plt.subplots(len(countries), 1,
                                 figsize=tuple(d * MM_TO_INCH for d in FIGURE_SIZE_MM),
                                 sharex=False, sharey=False, squeeze=False)
        fig.subplots_adjust(left=0.16, right=1, top=0.91,
                            bottom=0.07, hspace=0.5)
        used: set[str] = set()
        for i, (ax, country) in enumerate(zip(axes.flat, countries)):
            used.update(_draw_country_panel(
                ax, country, by_country[country], weather_order,
                lookups[country], letter="abcd"[i],
                show_weather_labels=(i == len(countries) - 1),
                show_totals=show_totals,
            ))

        fig.supylabel("Model-selected storage capacity (TWh)", x=0.022,
                     y=0.50, fontsize=9)
        handles = _legend_handles(used)
        if handles:
            # At 90 mm, a maximum of four columns; use two rows when needed.
            ncols = min(3, len(handles))
            fig.legend(handles=_legend_row_major(handles, ncols),
                       loc="upper center", bbox_to_anchor=(0.54, 1),
                       ncol=ncols, frameon=False, fontsize=9,
                       handlelength=0.84, handleheight=0.78,
                       handletextpad=0.30, columnspacing=0.82,
                       borderpad=0.0, labelspacing=0.25)
        return _save(fig, output_dir / "storage_portfolio_comparison.pdf", png=png)


def plot_country(
    country: str,
    country_cases: list[dict[str, object]],
    output_dir: Path,
    *,
    png: bool = False,
    show_totals: bool = False,
) -> tuple[Path, ...]:
    """Optional 190 x 80 mm individual-country figure, with sample labels."""
    weather_order, lookup = _paired_weather(country_cases, country)
    if not weather_order:
        return ()
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(190 * MM_TO_INCH, 80 * MM_TO_INCH))
        fig.subplots_adjust(left=0.115, right=0.985, top=0.77, bottom=0.25)
        used = _draw_country_panel(ax, country, country_cases, weather_order,
                                   lookup, letter="", show_weather_labels=True,
                                   show_totals=show_totals)
        ax.set_ylabel("Storage capacity (TWh)", fontsize=9)
        handles = _legend_handles(used)
        if handles:
            fig.legend(handles=_legend_row_major(handles, len(handles)),
                       loc="upper center", bbox_to_anchor=(0.53, 0.985),
                       ncol=len(handles), frameon=False, fontsize=9,
                       handlelength=0.85, handletextpad=0.35, columnspacing=0.9)
        return _save(fig, output_dir / f"storage_portfolio_{country}.pdf", png=png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="+", default=list(DEFAULT_COUNTRIES),
                        help="ISO3 country codes; NLD is always first if included")
    parser.add_argument("--separate", action="store_true",
                        help="Also available: wide, individual-country PDF export")
    parser.add_argument("--show-totals", action="store_true",
                        help="Show numeric TWh totals above each bar (8 pt)")
    parser.add_argument("--png", action="store_true", help="Also export 300-dpi PNGs")
    args = parser.parse_args()

    countries = _country_order(args.countries)
    portfolios = load_portfolios(args.input_dir, countries=countries)
    if args.separate:
        for country in countries:
            cases = [case for case in portfolios if case["country"] == country]
            for output in plot_country(country, cases, args.output_dir,
                                       png=args.png, show_totals=args.show_totals):
                print(f"Saved {output}")
    else:
        for output in plot_combined(portfolios, countries, args.output_dir,
                                    png=args.png, show_totals=args.show_totals):
            print(f"Saved {output}")


if __name__ == "__main__":
    main()
