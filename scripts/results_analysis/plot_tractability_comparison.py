"""Pair B1 and FX1 Calliope runs by country/weather and compare runtimes.

Run from the project root::

    pixi run python -m scripts.results_analysis.plot_tractability_comparison `
        --countries NLD GBR DEU ESP

Reads timestamps saved by Calliope 0.7 in the NetCDF ``attrs`` group:
``runtime.timings.solve_start``, ``solver_exit``, ``solve_complete``,
``build_start`` and ``build_complete``. The default metric is the elapsed wall-clock time between ``build_start``
and ``solve_complete`` (build + solve). The ``solver`` option measures
``solve_start`` to ``solver_exit``. Neither is a CPU-time measurement.
Other phases can be selected with ``--metric``. Results with explicit non-optimal termination
conditions are excluded from the comparison by default.

Each country/weather pair contributes one FX1/B1 runtime ratio; missing or
unreadable files are kept in the audit CSV, never treated as zero runtime.
Outputs: a single-column (90 x 120 mm) country-comparison PDF;
optional per-pair and per-country CSVs with ``--export-csv``. Each weather case appears at the country's exact
x-position, without jitter, and a black horizontal line marks its median.
Use ``--orientation horizontal`` to retain the original weather-year panels.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "results" / "capacity_expansion_models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tractability"
FONT_SIZE = 9
PLASMA = plt.get_cmap("plasma")
MM_TO_INCH = 1 / 25.4
FIGURE_WIDTH_MM = 90
DEFAULT_HEIGHT_MM = 80
COUNTRY_PRIORITY = ("NLD", "ESP", "DEU", "GBR")


def _country_order(countries) -> list[str]:
    """Always put the primary Dutch case first, then the other study countries."""
    countries = {str(c).upper() for c in countries}
    return [c for c in COUNTRY_PRIORITY if c in countries] + sorted(countries - set(COUNTRY_PRIORITY))

FILENAME_PATTERN = re.compile(
    r"^(?P<country>[A-Za-z]{3})_(?P<experiment>B1|FX1)_"
    r"(?P<weather>weather_\d{4}_\d+y)\.nc$", flags=re.IGNORECASE,
)
PHASES = {
    "solver": ("solve_start", "solver_exit"),
    "solve": ("solve_start", "solve_complete"),
    "build": ("build_start", "build_complete"),
    "build_solve": ("build_start", "solve_complete"),
    "postprocess": ("solver_exit", "solve_complete"),
}


def _unpack(value):
    """Decode JSON/YAML/dict strings written into NetCDF attributes."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or (
        not stripped.startswith(("{", "["))
        and "\n" not in stripped
        and not re.match(r"^(timings|runtime|termination_condition):", stripped)
    ):
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        pass
    try:
        import yaml
    except ImportError:
        pass
    else:
        try:
            candidate = yaml.safe_load(stripped)
            if isinstance(candidate, (dict, list)):
                return candidate
        except yaml.YAMLError:
            pass
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return value


def _flatten(value: Mapping, prefix: str = "") -> dict[str, object]:
    flattened = {}
    for name, raw in value.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        decoded = _unpack(raw)
        if isinstance(decoded, Mapping):
            flattened.update(_flatten(decoded, key))
        else:
            flattened[key.lower().replace("__", ".")] = decoded
    return flattened


def _load_attrs(path: Path) -> dict[str, object]:
    """Read only metadata groups (no multi-year result arrays)."""
    import xarray as xr

    all_attrs = {}
    for group, prefix in (("attrs", ""), (None, "root"), ("results", "results")):
        try:
            with xr.open_dataset(path, group=group) as dataset:
                all_attrs.update(_flatten(dataset.attrs, prefix))
        except (OSError, ValueError, KeyError):
            continue
    if not all_attrs:
        raise ValueError("No readable NetCDF attributes found")
    return all_attrs


def _field(attrs: dict[str, object], key: str):
    """Prefer runtime metadata over similarly named auxiliary attributes."""
    for candidate in (f"runtime.timings.{key}", f"attrs.runtime.timings.{key}",
                      f"timings.{key}", f"runtime.{key}", key):
        matches = [v for k, v in attrs.items() if k == candidate or k.endswith("." + candidate)]
        if matches:
            return matches[0]
    return None


def _event_seconds(value) -> float:
    """Convert UNIX seconds or ISO timestamps to comparable epoch seconds."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    try:
        return float(pd.Timestamp(value).timestamp())
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _duration(attrs: dict[str, object], first: str, last: str) -> float:
    start = _event_seconds(_field(attrs, first))
    finish = _event_seconds(_field(attrs, last))
    if not (math.isfinite(start) and math.isfinite(finish)):
        return float("nan")
    duration = finish - start
    if duration < 0:
        raise ValueError(f"Negative elapsed time between {first} and {last}")
    return duration


def parse_case_name(path: Path) -> dict[str, str] | None:
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return {
        "country": match["country"].upper(),
        "experiment": match["experiment"].upper(),
        "weather": match["weather"].lower(),
    }


def read_runtime(path: Path) -> dict[str, object]:
    attrs = _load_attrs(path)
    row = {f"{phase}_seconds": _duration(attrs, *events)
           for phase, events in PHASES.items()}
    status = _field(attrs, "termination_condition")
    row["termination_condition"] = str(status) if status is not None else "unknown"
    row["available_timing_events"] = ",".join(
        event for event in sorted(set(sum((list(v) for v in PHASES.values()), [])))
        if math.isfinite(_event_seconds(_field(attrs, event)))
    )
    return row


def _weather_order(weather: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"weather_(\d{4})_(\d+)y", weather)
    if match:
        return int(match[1]), int(match[2]), weather
    return 9999, 9999, weather


def _weather_label(weather: str) -> str:
    match = re.fullmatch(r"weather_(\d{4})_(\d+)y", weather)
    return f"{match[1]}–{int(match[1]) + int(match[2])}" if match else weather


def collect_cases(input_dir: Path, countries: list[str] | None) -> pd.DataFrame:
    requested = {c.upper() for c in countries} if countries else None
    paths = sorted(input_dir.expanduser().glob("*.nc"))
    if not paths:
        raise ValueError(f"No NetCDF result files found in {input_dir}")
    rows = []
    for path in paths:
        if path.name.endswith(".pending.nc"):
            continue
        case = parse_case_name(path)
        if case is None or (requested is not None and case["country"] not in requested):
            continue
        print(f"Reading {path.name}")
        row = {**case, "result_file": path.name, "read_error": ""}
        try:
            row.update(read_runtime(path))
        except (OSError, ValueError, KeyError) as exc:
            row["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    if not rows:
        raise ValueError("No B1/FX1 result files match the selected countries")
    cases = pd.DataFrame(rows)
    if cases.duplicated(["country", "experiment", "weather"]).any():
        duplicate = cases.loc[cases.duplicated(["country", "experiment", "weather"], keep=False)]
        raise ValueError("Duplicate country/experiment/weather cases: "
                         + ", ".join(duplicate["result_file"]))
    return cases


def _optimal(status: str) -> bool:
    return status.strip().lower() in {"optimal", "unknown", "none", "nan", ""}


def pair_cases(cases: pd.DataFrame, metric: str, *,
               include_nonoptimal: bool = False) -> pd.DataFrame:
    rows = []
    for (country, weather), group in cases.groupby(["country", "weather"], sort=False):
        by_experiment = group.set_index("experiment")
        row = {"country": country, "weather": weather, "metric": metric,
               "comparison_status": ""}
        for experiment in ("B1", "FX1"):
            prefix = experiment.lower()
            if experiment not in by_experiment.index:
                row.update({f"{prefix}_file": "", f"{prefix}_seconds": float("nan"),
                            f"{prefix}_termination_condition": "missing",
                            f"{prefix}_error": "missing case"})
                continue
            source = by_experiment.loc[experiment]
            row.update({
                f"{prefix}_file": source["result_file"],
                f"{prefix}_seconds": source.get(f"{metric}_seconds", float("nan")),
                f"{prefix}_termination_condition": source.get("termination_condition", "unknown"),
                f"{prefix}_error": source.get("read_error", ""),
            })
            # Keep all independently available phases for later tractability work.
            for phase in PHASES:
                row[f"{prefix}_{phase}_seconds"] = source.get(f"{phase}_seconds", float("nan"))

        b1, fx1 = row["b1_seconds"], row["fx1_seconds"]
        if not row["b1_file"] or not row["fx1_file"]:
            row["comparison_status"] = "unmatched"
        elif row["b1_error"] or row["fx1_error"]:
            row["comparison_status"] = "read_error"
        elif not include_nonoptimal and not all(
            _optimal(str(row[f"{exp}_termination_condition"])) for exp in ("b1", "fx1")
        ):
            row["comparison_status"] = "nonoptimal"
        elif not all(math.isfinite(float(value)) and float(value) > 0 for value in (b1, fx1)):
            row["comparison_status"] = "missing_or_zero_timing"
        else:
            row["comparison_status"] = "paired"
            row["runtime_ratio_fx1_over_b1"] = float(fx1) / float(b1)
            row["runtime_change_pct"] = (float(fx1) / float(b1) - 1) * 100
            row["runtime_difference_seconds"] = float(fx1) - float(b1)
        rows.append(row)
    paired = pd.DataFrame(rows)
    return paired.sort_values(
        ["country", "weather"], key=lambda column: (
            column.map(_weather_order) if column.name == "weather" else column
        ), ignore_index=True,
    )


def summarise_countries(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country, group in pairs.groupby("country", sort=True):
        complete = group.loc[group["comparison_status"] == "paired"]
        ratios = complete["runtime_ratio_fx1_over_b1"].to_numpy(dtype=float)
        rows.append({
            "country": country,
            "paired_weather_scenarios": len(complete),
            "available_weather_scenarios": len(group),
            "fx1_faster_scenarios": int(np.sum(ratios < 1)),
            "median_fx1_over_b1": float(np.median(ratios)) if len(ratios) else float("nan"),
            "geometric_mean_fx1_over_b1": (
                float(np.exp(np.log(ratios).mean())) if len(ratios) else float("nan")
            ),
            "median_runtime_change_pct": (
                float(np.median(complete["runtime_change_pct"])) if len(ratios) else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def plot_comparison_horizontal(pairs: pd.DataFrame, summary: pd.DataFrame,
                               metric: str, destination: Path) -> None:
    """Legacy weather-year-panel comparison, retained as a CLI option."""
    valid = pairs.loc[pairs["comparison_status"] == "paired"]
    if valid.empty:
        raise ValueError("No valid B1/FX1 pairs with runtime metadata for "
                         f"metric '{metric}'. Inspect the audit CSV or use --inspect.")

    countries = sorted(valid["country"].unique())
    values = valid["runtime_change_pct"].to_numpy(dtype=float)
    lo, hi = min(-10.0, float(values.min())), max(10.0, float(values.max()))
    padding = max(3, (hi - lo) * 0.07)
    limits = (lo - padding, hi + padding)
    palette = {c: PLASMA(k) for c, k in zip(
        countries, np.linspace(0.12, 0.88, max(2, len(countries)))
    )}

    with plt.rc_context({"font.size": FONT_SIZE, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(
            len(countries), 1, sharex=True,
            figsize=(9.5, max(3.2, 1.25 * len(countries) + 1.4)),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.16, right=0.97, top=0.94,
                            bottom=max(0.06, 0.8 / fig.get_figheight()),
                            hspace=0.65)
        for ax, country in zip(axes[:, 0], countries):
            sub = valid.loc[valid["country"] == country].copy()
            sub = sub.sort_values("weather", key=lambda col: col.map(_weather_order))
            ys = np.arange(len(sub))[::-1]
            changes = sub["runtime_change_pct"].to_numpy(dtype=float)
            ax.axvline(0, color="0.35", linewidth=0.9, zorder=1)
            for y, change in zip(ys, changes):
                ax.plot([0, change], [y, y], color=palette[country],
                        linewidth=1.4, alpha=0.82, zorder=2)
            ax.scatter(changes, ys, color=palette[country], s=32, zorder=3)
            ax.set_yticks(ys, [_weather_label(w) for w in sub["weather"]])
            ax.tick_params(axis="y", length=0, pad=7)
            ax.set_xlim(limits)
            ax.set_ylim(-0.6, len(sub) - 0.4)
            ax.grid(axis="x", color="0.88", linewidth=0.55)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            median = summary.loc[summary["country"] == country, "median_runtime_change_pct"].iloc[0]
            ax.text(0.99, 1.05, f"{country}  |  median {median:+.1f}%",
                    transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")

        axes[-1, 0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+g}%"))
        axes[-1, 0].set_xlabel(
            f"FX1 {metric.replace('_', ' ')} runtime change vs B1 (%) — negative = faster"
        )
        fig.savefig(destination)
        plt.close(fig)


def plot_comparison(pairs: pd.DataFrame, summary: pd.DataFrame,
                    metric: str, destination: Path,
                    orientation: str = "vertical", height_mm: float = DEFAULT_HEIGHT_MM) -> None:
    """A true single-column figure (90 mm wide) with one point per weather case.

    FX/B runtime change = 100 * (FX_seconds / B_seconds - 1).
    Negative values therefore mean the FX representation is faster.
    Black horizontal bars show the median across paired weather horizons.
    All observations retain their exact country x-position (no jitter).
    """
    if orientation == "horizontal":
        return plot_comparison_horizontal(pairs, summary, metric, destination)
    if orientation != "vertical":
        raise ValueError(f"Unknown plot orientation: {orientation}")
    if height_mm <= 50:
        raise ValueError("height_mm must be greater than 50")

    valid = pairs.loc[pairs["comparison_status"] == "paired"]
    if valid.empty:
        raise ValueError("No valid B1/FX1 runtime pairs for metric "
                         f"{metric!r}; inspect timing metadata or the audit CSV.")

    countries = _country_order(valid["country"].unique())
    changes = valid["runtime_change_pct"].to_numpy(dtype=float)
    # Zero is a meaningful reference. Do not waste y-axis area on a
    # mandatory +/-10% band when all observed effects are much larger.
    lo = min(0.0, float(changes.min()))
    hi = max(0.0, float(changes.max()))
    span = max(hi - lo, 1.0)
    pad = max(1.25, span * 0.055)
    palette = {
        country: PLASMA(shade)
        for country, shade in zip(countries, np.linspace(0.15, 0.80, len(countries)))
    }

    with plt.rc_context({"font.size": FONT_SIZE,
                         "axes.labelsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_MM * MM_TO_INCH,
                                       height_mm * MM_TO_INCH))
        for index, country in enumerate(countries):
            country_changes = valid.loc[
                valid["country"] == country, "runtime_change_pct"
            ].to_numpy(dtype=float)
            ax.scatter(
                np.full(len(country_changes), index), country_changes,
                s=38, color=palette[country], alpha=0.88,
                edgecolor="white", linewidth=0.3, zorder=3,
            )
            ax.hlines(float(np.median(country_changes)), index - 0.26, index + 0.26,
                      color="black", linewidth=2.1, zorder=4)

        ax.axhline(0, color="0.35", linewidth=0.95, zorder=1)
        ax.set_xlim(-0.55, len(countries) - 0.45)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(np.arange(len(countries)), countries)
        if len(countries) > 6:
            ax.tick_params(axis="x", labelrotation=45)
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")
        ax.tick_params(axis="x", length=0, pad=7)
        ax.tick_params(axis="y", length=3, pad=3)
        ax.set_ylabel("Runtime change, FX vs B (%)", labelpad=6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: "0%" if np.isclose(value, 0) else f"{value:+g}%"
        ))
        ax.grid(axis="y", color="0.87", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("0.55")

        fig.legend(
            handles=[
                Line2D([], [], linestyle="none", marker="o", markersize=5.5,
                       markerfacecolor="0.48", markeredgecolor="white",
                       label="Weather horizon"),
                Line2D([], [], color="black", linewidth=2.1, label="Median"),
            ],
            loc="upper center", bbox_to_anchor=(0.53, 0.99),
            ncol=2, frameon=False, columnspacing=1.25,
            handletextpad=0.45, fontsize=9,
        )
        # Leave room for the external manuscript caption, not inside the PDF.
        # Fractions give the data panel ~82% of the printed figure's height.
        fig.subplots_adjust(left=0.2, right=0.99, top=0.9,
                            bottom=0.08 if len(countries) <= 6 else 0.16)
        fig.savefig(destination)
        plt.close(fig)


def inspect_metadata(path: Path) -> None:
    attrs = _load_attrs(path)
    print(f"Attributes in {path.name}:")
    for key, value in sorted(attrs.items()):
        if "runtime" in key or "timing" in key or "termination" in key:
            print(f"  {key} = {str(value)[:140]}")
    print("Parsed metrics:", read_runtime(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--countries", nargs="*", help="ISO3 countries; default: all")
    parser.add_argument("--metric", choices=tuple(PHASES), default="build_solve",
                        help="Timing phase (default: build_solve, from build_start to solve_complete)")
    parser.add_argument("--orientation", choices=("vertical", "horizontal"),
                        default="vertical",
                        help="Compact country dots with black medians (default) "
                             "or original weather-year panels")
    parser.add_argument("--include-nonoptimal", action="store_true",
                        help="Allow pairs with explicitly non-optimal termination")
    parser.add_argument("--height-mm", type=float, default=DEFAULT_HEIGHT_MM,
                        help="Height of 90 mm-wide vertical PDF (default: 120 mm)")
    parser.add_argument("--export-csv", action="store_true",
                        help="Also write paired-case and country-summary audit CSV files")
    parser.add_argument("--inspect", type=Path,
                        help="Print runtime metadata keys from one .nc and exit")
    args = parser.parse_args()
    if args.inspect:
        inspect_metadata(args.inspect)
        return
    cases = collect_cases(args.input_dir, args.countries)
    pairs = pair_cases(cases, args.metric, include_nonoptimal=args.include_nonoptimal)
    summary = summarise_countries(pairs)
    suffix = "_".join(sorted(c.upper() for c in args.countries)) if args.countries else "all_countries"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_csv = args.output_dir / f"tractability_B1_FX1_pairs_{suffix}.csv"
    summary_csv = args.output_dir / f"tractability_B1_FX1_summary_{suffix}.csv"
    orientation_suffix = "" if args.orientation == "vertical" else "_horizontal"
    plot_pdf = args.output_dir / (
        f"tractability_{args.metric}_{suffix}{orientation_suffix}.pdf"
    )
    if args.export_csv:
        pairs.to_csv(pair_csv, index=False)
        summary.to_csv(summary_csv, index=False)
        for output in (pair_csv, summary_csv):
            print(f"Saved {output}")
    missing = pairs.loc[pairs["comparison_status"] != "paired"]
    if not missing.empty:
        warnings.warn(f"{len(missing)} country-weather pairs cannot be plotted; "
                      "run with --export-csv to audit missing cases, timings or solver status", stacklevel=1)
    plot_comparison(pairs, summary, args.metric, plot_pdf,
                    orientation=args.orientation, height_mm=args.height_mm)
    print(f"Saved {plot_pdf}")
    valid = pairs.loc[pairs["comparison_status"] == "paired"]
    print(f"Paired {len(valid)}/{len(pairs)} country-weather scenarios; "
          f"FX1 faster in {(valid['runtime_ratio_fx1_over_b1'] < 1).sum()}/{len(valid)}")


if __name__ == "__main__":
    main()
