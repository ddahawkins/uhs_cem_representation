"""Selected UHS design outcomes: EPR and WTIR across countries/formulations.

Creates one 190 x 112 mm, six-panel (3 technologies x 2 design metrics)
vector PDF, with one point per *invested* technology/weather/formulation case
and a black horizontal median for each group. Weather windows are pooled
anonymously in the plot, not averaged before plotting.

    pixi run python -m scripts.results_analysis.plot_storage_design_outcomes

Options:
    --input-dir PATH     NetCDF model results (default: results/capacity_expansion_models)
    --output-dir PATH    Figure + supporting CSVs (default: results/design_outcomes)
    --countries NLD DEU  Optional country list (default: COUNTRIES global)
    --horizon 3y         Only three-year weather samples (default); use 'all' for all
    --png                Also produce a 300 dpi PNG

Selected countries' B1, P1, F1, and FX1 cases are included. B / P / F / FX
identify formulations on the x-axes. With one country, Plasma colours
distinguish formulations; with multiple countries, Plasma colours distinguish
countries and their points/medians are offset within each formulation. Salt cavern,
aquifer, and depleted gas field are retained; CAES, batteries, hydropower,
and lined rock caverns are excluded.

Extracts installed storage energy (MWh), withdrawal (MW), and injection
(MW) using the project's existing `scripts/results_analysis/_plots.py`:
    EPR (days) = storage_cap / flow_out_cap / 24
    WTIR       = flow_out_cap / flow_in_cap
No-investment combinations are missing observations, NEVER zero-valued
EPR / WTIR points. Each weather sample contributes at most one value per
technology/formulation, to avoid accidentally overweighting multi-construct
results. The individual weather windows remain traceable in the CSV export.

The PDF dimensions are exact: do not use bbox_inches='tight' on export.
"""

from __future__ import annotations

import argparse
import hashlib
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

try:
    from ._plots import model_case, read_storage_investments
except ImportError:  # Run directly when the script is alongside _plots.py.
    from _plots import model_case, read_storage_investments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / "results_wtir_2_soc_pressure"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "design_outcomes"
COUNTRIES = ("NLD", "DEU", "GBR")  # E.g. ("NLD", "DEU", "GBR"); CLI --countries overrides.
EXPERIMENTS = ("B1", "P1", "F1", "FX1")
EXPERIMENT_LABELS = {"B1": "B", "P1": "P", "F1": "F", "FX1": "FX"}
TECHNOLOGIES = (
    ("salt_cavern", "Salt cavern"),
    ("aquifer", "Saline aquifer"),
    ("depleted_gas_field", "Depleted gas field"),
    # ("lined_rock_cavern", "Lined Rock Cavern")
)
METRICS = (
    ("EPR_days", "Energy-to-power ratio, EPR (days)"),
    ("WTIR", "Withdrawal-to-injection ratio, WTIR (-)"),
)
FIGURE_MM = (190, 100)  # Full width; deliberately under ~half a typical page height.
MM_TO_INCH = 1 / 25.4
PLASMA = plt.get_cmap("plasma")
EXPERIMENT_COLORS = dict(zip(EXPERIMENTS, PLASMA([0.10, 0.34, 0.58, 0.84])))
COLUMNS = [
    "country", "experiment", "weather", "case", "technology", "tech_id",
    "storage_MWh", "injection_MW", "withdrawal_MW", "capacity_TWh",
    "EPR_days", "WTIR",
]


def normalise_countries(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Accept country codes separated by spaces and/or commas, preserving order."""
    countries = tuple(dict.fromkeys(
        code.strip().upper()
        for item in values
        for code in item.split(",")
        if code.strip()
    ))
    if not countries or any(len(code) != 3 or not code.isalpha() for code in countries):
        raise ValueError("Provide one or more three-letter country codes, e.g. NLD DEU GBR")
    return countries


def _eligible_case(
    path: Path, horizon: str, countries: tuple[str, ...]
) -> dict[str, str] | None:
    """Apply country, formulation, and weather-horizon filters before I/O."""
    if path.name.endswith(".pending.nc"):
        return None
    try:
        case = model_case(path)
    except ValueError:
        return None
    if case["country"].upper() not in countries:
        return None
    if case["experiment"].upper() not in EXPERIMENTS:
        return None
    if horizon.lower() != "all" and not case["weather"].lower().endswith(
        "_" + horizon.lower()
    ):
        return None
    return case


def load_data(
    input_dir: Path, *, horizon: str = "3y", countries: tuple[str, ...] = COUNTRIES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read selected countries and retain one design per invested family per case."""
    countries = normalise_countries(countries)
    input_dir = input_dir.expanduser()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing NetCDF results directory: {input_dir}")

    paths: list[Path] = []
    cases_seen: set[tuple[str, str]] = set()
    for path in sorted(input_dir.glob("*.nc")):
        case = _eligible_case(path, horizon, countries)
        if case is None:
            continue
        key = (case["country"].upper(), case["experiment"].upper(), case["weather"].lower())
        if key in cases_seen:
            raise ValueError(f"Duplicate country / formulation / weather result: {key}")
        cases_seen.add(key)
        paths.append(path)
    if not paths:
        raise FileNotFoundError(
            f"No B1/P1/F1/FX1 *{horizon}* weather results for {countries} in {input_dir}"
        )

    records = []
    case_rows = []
    tech_ids = {x[0] for x in TECHNOLOGIES}
    for path in paths:
        case = _eligible_case(path, horizon, countries)
        assert case is not None
        case_rows.append({
            "country": case["country"].upper(), "experiment": case["experiment"].upper(),
            "weather": case["weather"].lower(), "case": case["case"],
        })
        print(f"Reading {path.name}", flush=True)
        # Do not silently convert unreadable files to missing investments.
        frame = read_storage_investments(path)
        if frame.empty:
            continue
        frame = frame.loc[frame["technology"].isin(tech_ids)].copy()
        if frame.empty:
            continue
        if frame.duplicated(subset=["technology"], keep=False).any():
            duplicates = sorted(frame.loc[
                frame.duplicated(subset=["technology"], keep=False), "technology"
            ].unique())
            raise ValueError(
                f"{path.name}: multiple installed constructs for {duplicates}. "
                "This plot expects one design per technology/weather/formulation "
                "in B1/P1/F1/FX1; do not silently weight a weather case twice."
            )
        frame["country"] = case["country"].upper()
        frame["experiment"] = case["experiment"].upper()
        frame["weather"] = case["weather"].lower()
        if (frame[["storage_MWh", "injection_MW", "withdrawal_MW"]] <= 0).any().any():
            raise ValueError(f"Nonpositive capacities in {path.name}")
        if not np.isfinite(frame[["EPR_days", "WTIR"]].to_numpy()).all():
            raise ValueError(f"Nonfinite EPR or WTIR in {path.name}")
        records.append(frame[COLUMNS])

    cases_df = pd.DataFrame(case_rows)
    points = (pd.concat(records, ignore_index=True) if records
              else pd.DataFrame(columns=COLUMNS))
    if points.empty:
        raise ValueError("No invested salt caverns, aquifers, or depleted gas fields in selected runs")
    for country in countries:
        country_cases = cases_df.loc[cases_df["country"] == country]
        missing = sorted(set(EXPERIMENTS) - set(country_cases["experiment"]))
        if missing:
            warnings.warn(
                f"{country}: missing requested formulations in NetCDF results: {missing}",
                stacklevel=2,
            )
    counts = cases_df.groupby(["country", "experiment"]).size().to_dict()
    print("Eligible weather cases by country/formulation:", counts)
    return points, cases_df


def summarise(
    points: pd.DataFrame, cases: pd.DataFrame, *, countries: tuple[str, ...] = COUNTRIES
) -> pd.DataFrame:
    """One row per country/technology/formulation, even without investment."""
    records = []
    for country in normalise_countries(countries):
        for family, _label in TECHNOLOGIES:
            for experiment in EXPERIMENTS:
                block = points.loc[
                    (points["country"] == country) &
                    (points["technology"] == family) &
                    (points["experiment"] == experiment)
                ]
                record: dict[str, object] = {
                    "country": country,
                    "technology": family,
                    "experiment": experiment,
                    "weather_cases_available": int((
                        (cases["country"] == country) &
                        (cases["experiment"] == experiment)
                    ).sum()),
                    "weather_cases_invested": int(len(block)),
                }
                for metric in ("EPR_days", "WTIR"):
                    values = block[metric].to_numpy(dtype=float)
                    record[f"{metric}_median"] = float(np.median(values)) if len(values) else np.nan
                    record[f"{metric}_min"] = float(np.min(values)) if len(values) else np.nan
                    record[f"{metric}_max"] = float(np.max(values)) if len(values) else np.nan
                records.append(record)
    return pd.DataFrame.from_records(records)


def _stable_jitter(group: pd.DataFrame, *, amplitude: float = 0.12) -> np.ndarray:
    """Deterministic, symmetric jitter; weather labels are never displayed."""
    n = len(group)
    if n <= 1:
        return np.zeros(n)
    # Even spread avoids overplotting identical B/P design values.
    slots = np.linspace(-amplitude, amplitude, n)
    # Keep stable yet non-chronological positions when weather samples change.
    hashes = [hashlib.sha256(str(x).encode("utf-8")).hexdigest()
              for x in group["weather"]]
    order = np.argsort(hashes)
    jitter = np.empty(n)
    jitter[order] = slots
    return jitter


def _set_y_range(ax: plt.Axes, values: np.ndarray, metric: str) -> None:
    """Per-row scales show design variation; WTIR includes physical parity=1."""
    if not len(values):
        ax.set_ylim(0, 1)
        return
    low, high = float(values.min()), float(values.max())
    span = high - low
    margin = max(span * 0.12, abs(high) * 0.025, 0.06)
    if metric == "WTIR":
        lower = min(1.0, low - margin)
        upper = max(1.0, high + margin)
    else:
        lower = low - margin
        upper = high + margin
    ax.set_ylim(max(0.0, lower), upper)


def _set_aquifer_wtir_axis(ax: plt.Axes, values: np.ndarray) -> None:
    """Use integer WTIR ticks without forcing saline aquifers to include 1.

    An integer-valued constant (typically WTIR = 2) is centred in the panel,
    rather than showing arbitrary fractional ticks around that value.
    """
    if not len(values):
        ax.set_ylim(1.5, 2.5)
        ax.set_yticks([2])
        return

    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high) and np.isclose(low, round(low)):
        centre = int(round(low))
        ax.set_ylim(max(0.0, centre - 0.5), centre + 0.5)
        ax.set_yticks([centre])
        return

    # If future scenarios produce varying WTIRs, expand to integer tick bounds.
    lower_tick = max(0, int(np.floor(low)))
    upper_tick = max(lower_tick + 1, int(np.ceil(high)))
    ax.set_ylim(max(0.0, lower_tick - 0.5), upper_tick + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=1))

def _set_aquifer_epr_axis(ax: plt.Axes, values: np.ndarray) -> None:
    """Use integer WTIR ticks without forcing saline aquifers to include 1.

    An integer-valued constant (typically WTIR = 2) is centred in the panel,
    rather than showing arbitrary fractional ticks around that value.
    """
    if not len(values):
        ax.set_ylim(65, 67)
        ax.set_yticks([66])
        return

    low, high = float(values.min()), float(values.max())
    if np.isclose(low, high) and np.isclose(low, round(low)):
        centre = int(round(low))
        ax.set_ylim(max(0.0, centre - 0.5), centre + 0.5)
        ax.set_yticks([centre])
        return

    # If future scenarios produce varying WTIRs, expand to integer tick bounds.
    lower_tick = max(0, int(np.floor(low)))
    upper_tick = max(lower_tick + 1, int(np.ceil(high)))
    ax.set_ylim(max(0.0, lower_tick - 0.5), upper_tick + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True, min_n_ticks=1))


def plot_design_outcomes(
    points: pd.DataFrame, *, output_dir: Path, countries: tuple[str, ...] = COUNTRIES,
    png: bool = False,
) -> tuple[Path, ...]:
    """Create the 3 x 2 journal figure, using black bars for medians.

    For one country, colours represent formulations, as in the original plot.
    For multiple countries, colours represent countries; each country receives
    its own within-formulation position and median, with a country legend.
    """
    countries = normalise_countries(countries)
    multiple_countries = len(countries) > 1
    country_colours = dict(zip(
        countries, PLASMA(np.linspace(0.10, 0.88, len(countries)))
    ))
    # Keep all country groups within their formulation's unit-width column.
    offset_span = min(0.54, 0.22 * (len(countries) - 1))
    country_offsets = dict(zip(
        countries, np.linspace(-offset_span / 2, offset_span / 2, len(countries))
    ))
    median_halfwidth = (min(0.12, 0.30 / len(countries))
                        if multiple_countries else 0.18)
    jitter_amplitude = 0 #(min(0.045, 0.15 / len(countries)) if multiple_countries else 0.12)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }):
        fig, axes = plt.subplots(
            len(TECHNOLOGIES), len(METRICS),
            figsize=(FIGURE_MM[0] * MM_TO_INCH, FIGURE_MM[1] * MM_TO_INCH),
            squeeze=False, sharex=False, sharey=False,gridspec_kw={"height_ratios": [1, 0.6, 1]},
        )
        top = (0.82 if len(countries) > 6 else 0.83) if multiple_countries else 0.88
        fig.subplots_adjust(left=0.05, right=1, bottom=0.08,
                            top=top, wspace=0.1, hspace=0.55)
        for col, (_metric, heading) in enumerate(METRICS):
            pos = axes[0, col].get_position()
            fig.text(pos.x0 + pos.width / 2, .94, heading,
                     ha="center", va="top", fontsize=9.5, weight="bold")

        if multiple_countries:
            handles = [Line2D(
                [0], [0], marker="o", linestyle="None", markersize=5,
                markerfacecolor=country_colours[country], markeredgecolor="white",
                label=country,
            ) for country in countries]
            fig.legend(
                handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1),
                ncol=min(6, len(countries)), frameon=False, fontsize=8,
                columnspacing=1.3, handlelength=0.6, handletextpad=0.45,
                borderaxespad=0,
            )

        for row, (family, title) in enumerate(TECHNOLOGIES):
            subset = points.loc[points["technology"] == family]
            for col, (metric, _heading) in enumerate(METRICS):
                ax = axes[row, col]
                pos = ax.get_position()
                # One compact row label across the two columns; avoids repeating titles.
                if col == 0:
                    fig.text(pos.x0, pos.y1 + 0.012, f"({chr(97 + row)})  {title}",
                             ha="left", va="bottom", fontsize=9.5, weight="bold")

                all_values = subset[metric].to_numpy(dtype=float)
                if family == "aquifer" and metric == "WTIR":
                    _set_aquifer_wtir_axis(ax, all_values)
                elif family == "aquifer" and metric == "EPR_days":
                    _set_aquifer_epr_axis(ax, all_values)
                else:
                    _set_y_range(ax, all_values, metric)
                    ax.yaxis.set_major_locator(
                        MaxNLocator(nbins=4, integer=True, min_n_ticks=3)
                    )
                fmt = ScalarFormatter(useOffset=False)
                fmt.set_scientific(False)
                ax.yaxis.set_major_formatter(fmt)
                ax.yaxis.grid(True, color="0.90", linewidth=0.55)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["left", "bottom"]].set_color("0.56")
                ax.spines[["left", "bottom"]].set_linewidth(0.65)
                ax.tick_params(axis="both", labelsize=9, length=2.4,
                               width=0.65, colors="0.22", pad=2)
                ax.tick_params(axis="x", length=0)
                ax.set_xlim(-0.45, len(EXPERIMENTS) - 0.55)
                ax.set_xticks(range(len(EXPERIMENTS)),
                              [EXPERIMENT_LABELS[e] for e in EXPERIMENTS])

                if metric == "WTIR":
                    y0, y1 = ax.get_ylim()
                    if y0 <= 1.0 <= y1:
                        ax.axhline(1.0, color="0.55", lw=0.65, ls=(0, (3, 3)),
                                   zorder=1)

                for xpos, experiment in enumerate(EXPERIMENTS):
                    for country in countries:
                        group = subset.loc[
                            (subset["experiment"] == experiment) &
                            (subset["country"] == country)
                        ]
                        if group.empty:
                            continue
                        yvalues = group[metric].to_numpy(dtype=float)
                        centre = xpos + country_offsets[country]
                        xvalues = centre + _stable_jitter(
                            group, amplitude=jitter_amplitude
                        )
                        colour = (country_colours[country] if multiple_countries
                                  else EXPERIMENT_COLORS[experiment])
                        ax.scatter(
                            xvalues, yvalues,
                            s=max(12, 27 - 1.2 * len(countries)) if multiple_countries else 27,
                            facecolor=colour,
                            edgecolor="white", linewidth=0.36,
                            alpha=0.95, zorder=3,
                        )
                        # A separate black median, outlined white, for each
                        # country/formulation group (not pooled across countries).
                        med = float(np.median(yvalues))
                        ax.plot(
                            [centre - median_halfwidth, centre + median_halfwidth],
                            [med, med], color="white", lw=2.2,
                            solid_capstyle="butt", zorder=4,
                        )
                        ax.plot(
                            [centre - median_halfwidth, centre + median_halfwidth],
                            [med, med], color="black", lw=1.65 if multiple_countries else 2,
                            solid_capstyle="butt", zorder=5,
                        )
        fig.supxlabel(
            "Cost formulation",
            x=0.535, y=0.00,
            fontsize=9,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        country_suffix = "_".join(countries)
        path = output_dir / f"storage_design_outcomes_{country_suffix}.pdf"
        fig.savefig(path, facecolor="white")  # Physical width and height exact.
        outputs = [path]
        if png:
            png_path = path.with_suffix(".png")
            fig.savefig(png_path, dpi=300, facecolor="white")
            outputs.append(png_path)
        plt.close(fig)
    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--countries", nargs="+", default=None,
        help="Country codes (space- or comma-separated), e.g. NLD DEU GBR; "
             "defaults to COUNTRIES in the script",
    )
    parser.add_argument("--horizon", default="3y",
                        help="Weather horizon suffix (default: 3y); 'all' to disable")
    parser.add_argument("--png", action="store_true", help="Also export 300 dpi PNG")
    args = parser.parse_args()

    countries = normalise_countries(args.countries if args.countries is not None else COUNTRIES)
    points, cases = load_data(args.input_dir, horizon=args.horizon, countries=countries)
    summary = summarise(points, cases, countries=countries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    country_suffix = "_".join(countries)
    points_file = args.output_dir / f"storage_design_outcomes_{country_suffix}_points.csv"
    summary_file = args.output_dir / f"storage_design_outcomes_{country_suffix}_summary.csv"
    points.to_csv(points_file, index=False)
    summary.to_csv(summary_file, index=False)
    for out in plot_design_outcomes(
        points, output_dir=args.output_dir, countries=countries, png=args.png
    ):
        print(f"Saved {out}")
    print(f"Saved {points_file}")
    print(f"Saved {summary_file}")
    print(summary[["country", "technology", "experiment", "weather_cases_available",
                   "weather_cases_invested", "EPR_days_median", "WTIR_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
