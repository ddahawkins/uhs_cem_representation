#!/usr/bin/env python3
"""Check binding of the rolling SoC pressure-envelope constraints in Calliope results.

The model imposes the *forward* storage inequalities

    storage[t + N] >= a_min * storage_cap + b_min * storage[t]
    storage[t + N] <= a_max * storage_cap + b_max * storage[t]

where N is the number of *timesteps* in the saved Calliope math. For an
installed site, dividing by storage_cap and subtracting SoC[t] gives

    L[t] <= (SoC[t + N] - SoC[t]) <= U[t],
    L[t] = a_min + (b_min - 1) * SoC[t],
    U[t] = a_max + (b_max - 1) * SoC[t].

This script reads the coefficients and the optimised SoC from each .nc file;
it does not use site design assumptions from a potentially different run.

IMPORTANT: N=24 at a 2-hour model resolution is a 48-hour window, *not* 24 h.
The reported binding status reproduces the constraint actually enforced by
the model, and the tool explicitly warns when the window is not 24 h.

Dependencies: numpy, pandas, xarray, h5netcdf, matplotlib (for timeseries).

Examples (from the uhs_cem_representation repository):

    pixi run python scripts/results_analysis/plot_soc_pressure_binding.py summary `
        --results-dir hpc/results_wtir_2_manosque --countries NLD DEU ESP GBR --formulations FX1 B1 `
        --weather 2009_3y 2012_3y 2015_3y 2018_3y 2021_ey --output-dir results/soc_pressure

    pixi run python scripts/results_analysis/plot_soc_pressure_binding.py timeseries `
        --results-dir hpc/results_wtir_2_manosque --country DEU --formulation B1 `
        --weather 2009_3y --window-days 21 --output-dir results/soc_pressure

    # Select a specific period rather than the automatically chosen binding event:
    pixi run python scripts/results_analysis/plot_soc_pressure_binding.py timeseries `
        --results-dir hpc/results_wtir_2_manosque --country DEU --formulation B1 `
        --weather 2018_3y --start-date 2018-01-01 --end-date 2021-01-31
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


CASE_PATTERN = re.compile(
    r"^(?P<country>[A-Z]{3})_(?P<formulation>(?:FX|B|P|F)\d+)_(?P<weather>weather_.+)\.nc$",
    flags=re.IGNORECASE,
)
COEFFICIENTS = (
    "soc_24h_a_min",
    "soc_24h_b_min",
    "soc_24h_a_max",
    "soc_24h_b_max",
)
DEFAULT_SHIFT_STEPS = 24  # Fallback ONLY when metadata does not expose the math.


@dataclass(frozen=True)
class Case:
    country: str
    formulation: str
    weather: str
    path: Path

    @property
    def weather_years(self) -> str:
        return self.weather.removeprefix("weather_")


@dataclass
class SiteTrace:
    case: Case
    node: str
    tech: str
    capacity_mwh: float
    timestamps: pd.DatetimeIndex
    soc_fraction: np.ndarray
    change: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    slack_lower: np.ndarray
    slack_upper: np.ndarray
    active_lower: np.ndarray
    active_upper: np.ndarray
    violation_lower: np.ndarray
    violation_upper: np.ndarray
    valid: np.ndarray
    shift_steps: int
    timestep_hours: float
    window_hours: float
    cyclic: bool


def parse_case(path: Path) -> Case | None:
    match = CASE_PATTERN.fullmatch(path.name)
    if not match:
        return None
    return Case(
        country=match.group("country").upper(),
        formulation=match.group("formulation").upper(),
        weather=match.group("weather"),
        path=path,
    )


def weather_matches(actual: str, requested: list[str] | None) -> bool:
    if not requested:
        return True
    actual = actual.removeprefix("weather_").lower()
    return any(
        actual == key.removeprefix("weather_").lower()
        or actual.startswith(key.removeprefix("weather_").lower() + "_")
        for key in requested
    )


def form_matches(actual: str, requested: list[str] | None) -> bool:
    if not requested:
        return True
    actual = actual.upper()
    return any(
        actual == value.upper()
        or (re.fullmatch(r"FX|B|P|F", value.upper()) and actual.startswith(value.upper()))
        for value in requested
    )


def discover_cases(
    results_dir: Path,
    countries: list[str] | None = None,
    weather: list[str] | None = None,
    formulations: list[str] | None = None,
) -> list[Case]:
    if not results_dir.is_dir():
        raise ValueError(f"Not a directory: {results_dir}")
    all_cases = [parse_case(path) for path in results_dir.rglob("*.nc")]
    filtered = [
        case
        for case in all_cases
        if case is not None
        and (not countries or case.country in {v.upper() for v in countries})
        and weather_matches(case.weather, weather)
        and form_matches(case.formulation, formulations)
    ]
    if not filtered:
        raise ValueError(
            f"No matching country/formulation/weather .nc files in {results_dir}. "
            "Expected names such as DEU_FX1_weather_2009_3y.nc."
        )
    return sorted(filtered, key=lambda c: (c.country, c.weather, c.formulation, str(c.path)))


def open_group(path: Path, group: str) -> xr.Dataset:
    """Calliope saves 'inputs' and 'results' as HDF5 groups."""
    try:
        return xr.open_dataset(path, group=group, engine="h5netcdf")
    except (ImportError, ValueError, OSError):
        # Some environments register the NetCDF4 backend but not h5netcdf.
        return xr.open_dataset(path, group=group)


def read_shift_steps(path: Path, override: int | None) -> int:
    if override is not None:
        return override
    try:
        with open_group(path, "attrs") as metadata:
            math = metadata.attrs.get("math", "")
        if isinstance(math, bytes):
            math = math.decode("utf-8")
        # The two constraints must use the same roll and thus the same window.
        matches = re.findall(
            r"^\s*uhs_soc_24h_(min|max):[\s\S]*?roll\(storage,\s*timesteps\s*=\s*-(\d+)\)",
            str(math),
            flags=re.MULTILINE,
        )
        found = {int(count) for _, count in matches}
        bounds = {bound for bound, _ in matches}
        if len(found) == 1 and bounds == {"min", "max"}:
            return found.pop()
        if matches:
            raise ValueError(f"Inconsistent SoC constraint offsets: {matches}")
    except (KeyError, OSError, ValueError, ImportError) as exc:
        warnings.warn(f"Could not read the SoC math from {path.name}: {exc}", stacklevel=2)
    warnings.warn(
        f"{path.name}: no roll offset found in saved model math; assuming "
        f"{DEFAULT_SHIFT_STEPS} timesteps. Use --shift-steps to specify it.",
        stacklevel=2,
    )
    return DEFAULT_SHIFT_STEPS


def time_resolution_hours(timestamps: pd.DatetimeIndex) -> float:
    if len(timestamps) < 2:
        return float("nan")
    gaps = np.diff(timestamps.asi8) / (3600 * 1e9)
    if not np.allclose(gaps, gaps[0], rtol=0, atol=1e-9):
        warnings.warn(
            "Timesteps are not equally spaced. The roll offset is still exact, "
            "but the rolling window does not have a fixed duration in hours.",
            stacklevel=2,
        )
        return float("nan")
    return float(gaps[0])


def evaluate_case(
    case: Case,
    tolerance: float,
    shift_override: int | None,
) -> tuple[list[SiteTrace], int, float, float, int]:
    """Reconstruct both inequalities for every installed, constrained node/tech."""
    shift = read_shift_steps(case.path, shift_override)
    with open_group(case.path, "inputs") as inputs, open_group(case.path, "results") as results:
        missing = [v for v in COEFFICIENTS if v not in inputs]
        missing += [v for v in ("storage_cap", "storage") if v not in results]
        if missing:
            raise ValueError(f"Missing NetCDF variables in {case.path.name}: {missing}")

        timestamps = pd.DatetimeIndex(pd.to_datetime(results.timesteps.values))
        hours_per_step = time_resolution_hours(timestamps)
        window_hours = hours_per_step * shift
        if np.isfinite(window_hours) and not np.isclose(window_hours, 24):
            warnings.warn(
                f"{case.path.name}: SoC '24h' constraints roll {shift} model "
                f"timesteps of {hours_per_step:g} hours = {window_hours:g} HOURS, "
                "not 24 hours. The analysis matches the IMPLEMENTED constraint.",
                stacklevel=2,
            )
        if shift >= len(timestamps):
            raise ValueError(f"Roll offset {shift} >= {len(timestamps)} timesteps")

        records: list[SiteTrace] = []
        n_uninstalled = 0
        for tech_value in results.techs.values:
            tech = str(tech_value)
            try:
                coefficients = [float(inputs[name].sel(techs=tech)) for name in COEFFICIENTS]
            except (KeyError, TypeError, ValueError):
                continue
            if not np.all(np.isfinite(coefficients)):
                continue  # Non-SC technologies normally have no pressure envelope.
            a_min, b_min, a_max, b_max = coefficients
            cyclic = bool(inputs.cyclic_storage.sel(techs=tech)) if "cyclic_storage" in inputs else False

            for node_value in results.nodes.values:
                node = str(node_value)
                capacity = float(results.storage_cap.sel(nodes=node, techs=tech))
                if not np.isfinite(capacity) or capacity <= 1e-9:
                    n_uninstalled += 1
                    continue  # Zero capacity would give spurious 0 = 0 'binding'.
                storage = np.asarray(
                    results.storage.sel(nodes=node, techs=tech).values, dtype=float
                )
                soc = storage / capacity
                forward_soc = np.roll(soc, -shift)
                valid = np.isfinite(soc) & np.isfinite(forward_soc)
                if not cyclic:
                    valid[-shift:] = False  # No cross-horizon constraint for noncyclic sites.

                change = forward_soc - soc
                lower = a_min + (b_min - 1.0) * soc
                upper = a_max + (b_max - 1.0) * soc
                slack_lower = change - lower
                slack_upper = upper - change
                active_lower = valid & (np.abs(slack_lower) <= tolerance)
                active_upper = valid & (np.abs(slack_upper) <= tolerance)
                violations_lower = valid & (slack_lower < -tolerance)
                violations_upper = valid & (slack_upper < -tolerance)

                records.append(
                    SiteTrace(
                        case=case,
                        node=node,
                        tech=tech,
                        capacity_mwh=capacity,
                        timestamps=timestamps,
                        soc_fraction=soc,
                        change=change,
                        lower=lower,
                        upper=upper,
                        slack_lower=slack_lower,
                        slack_upper=slack_upper,
                        active_lower=active_lower,
                        active_upper=active_upper,
                        violation_lower=violations_lower,
                        violation_upper=violations_upper,
                        valid=valid,
                        shift_steps=shift,
                        timestep_hours=hours_per_step,
                        window_hours=window_hours,
                        cyclic=cyclic,
                    )
                )

        return records, shift, hours_per_step, window_hours, n_uninstalled


def count_row(trace: SiteTrace) -> dict[str, object]:
    both = trace.active_lower & trace.active_upper
    either = trace.active_lower | trace.active_upper
    return {
        "country": trace.case.country,
        "weather_years": trace.case.weather_years,
        "formulation": trace.case.formulation,
        "node": trace.node,
        "technology": trace.tech,
        "installed_capacity_mwh": trace.capacity_mwh,
        "evaluated_timesteps": int(trace.valid.sum()),
        "lower_binding_timesteps": int(trace.active_lower.sum()),
        "upper_binding_timesteps": int(trace.active_upper.sum()),
        "either_binding_timesteps": int(either.sum()),
        "both_binding_timesteps": int(both.sum()),
        "lower_violation_timesteps": int(trace.violation_lower.sum()),
        "upper_violation_timesteps": int(trace.violation_upper.sum()),
        "minimum_lower_slack_soc_fraction": float(np.min(trace.slack_lower[trace.valid])),
        "minimum_upper_slack_soc_fraction": float(np.min(trace.slack_upper[trace.valid])),
        "shift_steps": trace.shift_steps,
        "timestep_hours": trace.timestep_hours,
        "actual_window_hours": trace.window_hours,
        "nominal_window_hours": 24,
        "source_file": str(trace.case.path),
    }


def run_summary(args: argparse.Namespace) -> None:
    cases = discover_cases(
        args.results_dir, args.countries, args.weather, args.formulations
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    site_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for case in cases:
        try:
            traces, shift, hours, window_hours, n_uninstalled = evaluate_case(
                case, args.tolerance, args.shift_steps
            )
        except (OSError, KeyError, ValueError, ImportError) as exc:
            failures.append(f"{case.path.name}: {exc}")
            continue

        site_rows.extend(count_row(trace) for trace in traces)
        # Count time indices where *any* constrained installed asset was binding,
        # not the sum across sites (which would double-count hours).
        shape = len(traces[0].valid) if traces else 0
        lower = np.zeros(shape, dtype=bool)
        upper = np.zeros(shape, dtype=bool)
        lower_violations = np.zeros(shape, dtype=bool)
        upper_violations = np.zeros(shape, dtype=bool)
        valid = np.zeros(shape, dtype=bool)
        for trace in traces:
            lower |= trace.active_lower
            upper |= trace.active_upper
            lower_violations |= trace.violation_lower
            upper_violations |= trace.violation_upper
            valid |= trace.valid
        either = lower | upper
        case_rows.append(
            {
                "country": case.country,
                "weather_years": case.weather_years,
                "formulation": case.formulation,
                "installed_constrained_sites": len(traces),
                "uninstalled_constrained_sites": n_uninstalled,
                "evaluated_timesteps": int(valid.sum()),
                "lower_binding_timesteps": int(lower.sum()),
                "upper_binding_timesteps": int(upper.sum()),
                "either_binding_timesteps": int(either.sum()),
                "lower_violation_timesteps": int(lower_violations.sum()),
                "upper_violation_timesteps": int(upper_violations.sum()),
                "binding_share_pct": 100.0 * either.sum() / max(1, valid.sum()),
                "shift_steps": shift,
                "timestep_hours": hours,
                "actual_window_hours": window_hours,
                "nominal_window_hours": 24,
                "source_file": str(case.path),
            }
        )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
    if not case_rows:
        raise SystemExit("No cases could be evaluated; check .nc paths and errors above.")

    cases_df = pd.DataFrame(case_rows)
    sites_df = pd.DataFrame(site_rows)
    all_path = output_dir / "soc_pressure_binding_by_case.csv"
    binding_path = output_dir / "soc_pressure_binding_cases_only.csv"
    sites_path = output_dir / "soc_pressure_binding_by_site.csv"
    cases_df.to_csv(all_path, index=False)
    cases_df.loc[cases_df.either_binding_timesteps > 0].to_csv(binding_path, index=False)
    sites_df.to_csv(sites_path, index=False)

    columns = [
        "country", "weather_years", "formulation", "installed_constrained_sites",
        "lower_binding_timesteps", "upper_binding_timesteps",
        "either_binding_timesteps", "binding_share_pct", "actual_window_hours",
    ]
    shown = cases_df.loc[cases_df.either_binding_timesteps > 0, columns]
    print(f"\nCases with a binding pressure-envelope constraint ({len(shown)}/{len(cases_df)}):")
    print(shown.to_string(index=False, float_format=lambda number: f"{number:.2f}")
          if len(shown) else "(none)")
    if int(cases_df.lower_violation_timesteps.sum() + cases_df.upper_violation_timesteps.sum()) > 0:
        print("WARNING: Some saved SoC trajectories violate the reconstructed bounds; "
              "inspect the violation columns and verify the saved math.", file=sys.stderr)
    print(f"\nSaved: {all_path}\nSaved: {binding_path}\nSaved: {sites_path}")
    if failures:
        raise SystemExit(f"{len(failures)} of {len(cases)} input files failed (listed above).")


def choose_trace(traces: list[SiteTrace], node: str | None, tech: str | None) -> SiteTrace:
    if not traces:
        raise ValueError(
            "This case has no installed storage technology with SoC pressure "
            "constraints; there is no meaningful envelope to plot."
        )
    selected = [
        trace for trace in traces
        if (node is None or trace.node.lower() == node.lower())
        and (tech is None or tech.lower() in trace.tech.lower())
    ]
    if not selected:
        available = ", ".join(f"{trace.node}/{trace.tech}" for trace in traces)
        raise ValueError(f"No installed constrained site matches --node/--tech. Available: {available}")
    if len(selected) > 1:
        # Prefer the largest installed site, but disclose the automatic selection.
        selected.sort(key=lambda trace: trace.capacity_mwh, reverse=True)
        print(f"Multiple installed sites matched; selecting largest: {selected[0].node}/{selected[0].tech}")
    return selected[0]


def select_interval(trace: SiteTrace, args: argparse.Namespace) -> np.ndarray:
    timestamps = trace.timestamps
    start = pd.Timestamp(args.start_date) if args.start_date else None
    end = pd.Timestamp(args.end_date) if args.end_date else None
    if start is None and end is None:
        # Show the first binding event; if none, the smallest feasible slack.
        active = np.flatnonzero(trace.active_lower | trace.active_upper)
        if len(active):
            central = int(active[0])
        else:
            slacks = np.minimum(trace.slack_lower, trace.slack_upper)
            central = int(np.nanargmin(np.where(trace.valid, slacks, np.nan)))
        center_date = timestamps[central]
        start = center_date - pd.Timedelta(days=args.window_days / 2)
        end = center_date + pd.Timedelta(days=args.window_days / 2)
    elif start is None:
        start = end - pd.Timedelta(days=args.window_days)
    elif end is None:
        end = start + pd.Timedelta(days=args.window_days)
    if end < start:
        raise ValueError("--end-date must be after --start-date")
    mask = (timestamps >= start) & (timestamps <= end) & trace.valid
    if not mask.any():
        raise ValueError("No valid windows in the requested date range; adjust dates.")
    print(f"Plotting {timestamps[mask][0]} to {timestamps[mask][-1]}")
    return mask


def run_timeseries(args: argparse.Namespace) -> None:
    cases = discover_cases(
        args.results_dir,
        [args.country],
        [args.weather],
        [args.formulation],
    )
    if len(cases) != 1:
        raise ValueError(
            f"Requested selection matches {len(cases)} cases, expected exactly one: "
            + ", ".join(str(case.path) for case in cases)
        )
    traces, _, _, _, _ = evaluate_case(cases[0], args.tolerance, args.shift_steps)
    trace = choose_trace(traces, args.node, args.tech)
    mask = select_interval(trace, args)

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(7.48, 3.7))
    # Reserve room above the axis so the five-entry legend never covers data.
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.73)
    times = trace.timestamps[mask]
    lower = 100 * trace.lower[mask]
    upper = 100 * trace.upper[mask]
    actual = 100 * trace.change[mask]
    cmap = plt.get_cmap("plasma")
    axis.fill_between(times, lower, upper, color="0.7", alpha=0.24, label="Permitted envelope")
    axis.plot(times, lower, color=cmap(0.20), linewidth=1.3, label="Lower bound (moving)")
    axis.plot(times, upper, color=cmap(0.83), linewidth=1.3, label="Upper bound (moving)")
    axis.plot(times, actual, color="black", linewidth=1.1, label="Observed forward SoC change")

    lower_active = trace.active_lower[mask]
    upper_active = trace.active_upper[mask]
    if np.any(lower_active):
        axis.scatter(times[lower_active], actual[lower_active], s=16, facecolors="none",
                     edgecolors=cmap(0.20), linewidths=0.9, zorder=5,
                     label="Lower binding")
    if np.any(upper_active):
        axis.scatter(times[upper_active], actual[upper_active], s=16, facecolors="none",
                     edgecolors=cmap(0.83), linewidths=0.9, zorder=5,
                     label="Upper binding")
    axis.axhline(0, color="0.65", linewidth=0.6, linestyle=":")
    axis.set_ylabel("Change in SoC (% of storage capacity)")
    axis.set_xlabel("Start of rolling window")
    label = "Salt cavern" if "salt_cavern" in trace.tech.lower() else trace.tech
    fig.suptitle(
        f"{trace.case.country} · {trace.case.formulation} · {trace.case.weather_years} · "
        f"{label} | {trace.shift_steps} steps = {trace.window_hours:g} h",
        y=0.99,
    )
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axis.grid(axis="y", color="0.88", linewidth=0.55)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               frameon=False, ncol=3, columnspacing=1.5, handlelength=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"soc_pressure_timeseries_{trace.case.country}_{trace.case.formulation}_"
        f"{trace.case.weather_years}_{re.sub('[^a-zA-Z0-9]+', '_', trace.tech)}.pdf"
    )
    output = args.output_dir / name
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Selected asset: {trace.node}/{trace.tech} "
        f"(storage_cap={trace.capacity_mwh:,.3f} MWh); "
        f"lower binding={trace.active_lower.sum()}, upper binding={trace.active_upper.sum()} "
        f"of {trace.valid.sum()} valid timesteps in full case."
    )
    print(f"Saved: {output}")


def positive_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be a finite positive number")
    return value


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


def parser() -> argparse.ArgumentParser:
    program = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subcommands = program.add_subparsers(dest="command", required=True)
    for name in ("summary", "timeseries"):
        command = subcommands.add_parser(name, help="Tabulate binding" if name == "summary" else "Plot a single case")
        command.add_argument("--results-dir", type=Path, required=True, help="Root folder searched recursively for .nc results")
        command.add_argument("--output-dir", type=Path, default=Path("results/soc_pressure"))
        command.add_argument("--tolerance", type=positive_float, default=1e-6,
                             help="Slack tolerance in fraction of capacity (default 1e-6)")
        command.add_argument("--shift-steps", type=positive_int, default=None,
                             help="Override ONLY if the saved roll offset cannot be read; "
                                  "a changed offset does not reproduce the original constraint")

    summary = subcommands.choices["summary"]
    summary.add_argument("--countries", nargs="+", help="e.g. NLD DEU GBR")
    summary.add_argument("--weather", nargs="+", help="e.g. 2009_3y or weather_2009_3y; a year alone also matches")
    summary.add_argument("--formulations", nargs="+", help="e.g. B1 P1 F1 FX1 (or FX to match all FX versions)")
    summary.set_defaults(func=run_summary)

    timeseries = subcommands.choices["timeseries"]
    timeseries.add_argument("--country", required=True, help="e.g. DEU")
    timeseries.add_argument("--weather", required=True, help="e.g. 2009_3y")
    timeseries.add_argument("--formulation", required=True, help="e.g. B1 or FX1")
    timeseries.add_argument("--node", help="Specific node name (optional)")
    timeseries.add_argument("--tech", help="Technology substring, e.g. salt_cavern (optional)")
    timeseries.add_argument("--start-date", help="Start ISO date/time; auto-centred on binding if omitted")
    timeseries.add_argument("--end-date", help="End ISO date/time")
    timeseries.add_argument("--window-days", type=positive_float, default=21.0,
                            help="Displayed period when start/end not both given (default 21)")
    timeseries.set_defaults(func=run_timeseries)
    return program


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        parser().error(str(exc))


if __name__ == "__main__":
    main()
