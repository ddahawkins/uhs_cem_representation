"""Compare system and storage costs across Calliope representations.

Run from the repository root::

    pixi run python -m scripts.results_analysis.system_storage_cost_plots

    pixi run python -m scripts.results_analysis.system_storage_cost_plots \
        --input-dir hpc/results_2h --output-dir results/figures

Place alongside ``_plots.py`` in ``scripts/results_analysis/``.

Each experiment is divided by the B1 result for the SAME country and weather
window. The upper panel uses the saved optimisation objective; the lower panel
uses Calliope's per-technology ``cost`` output, which already includes
annualised investment, fixed O&M, and variable O&M. Do not add these components
to ``cost`` again.

Storage includes all battery durations, CAES, pumped hydro, and every UHS
variant (including multiple variants of one family in FX2). It excludes
electrolysers and H2-to-power converters, which are not storage technologies.

Output: one two-panel, single-column PDF and one auditable CSV.
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
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

try:
    from ._plots import model_case, storage_family
except ImportError:  # Allow direct execution from this directory.
    from _plots import model_case, storage_family


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / "results_2h_no_coupling_bounds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
EXPERIMENT_ORDER = ("B1", "P1", "F1", "FX1", "FX2")
COUNTRIES = ("NLD","FIN")

# Additional storage technologies. Geological variants and CAES are recognised
# by the existing `_plots.storage_family` helper.
BATTERY_PATTERN = re.compile(r"battery_\d+(?:\.\d+)?h$")

FIGURE_WIDTH_IN = 3.50  # About 89 mm: one Elsevier two-column page column.
FIGURE_HEIGHT_IN = 5.35
FONT_SIZE = 10
POINT_COLOUR = plt.get_cmap("plasma")(0.33)


def _is_storage(tech_id: str) -> bool:
    """Match concrete technologies, not families after aggregating variants."""
    return (tech_id == "pumped_hydro"
            or BATTERY_PATTERN.fullmatch(tech_id) is not None
            or storage_family(tech_id) is not None)


def _monetary(array: xr.DataArray) -> xr.DataArray:
    """Restrict to monetary costs, avoiding sums of money and emissions."""
    if "costs" not in array.dims:
        return array
    cost_types = {str(value) for value in array.coords["costs"].values}
    if "monetary" not in cost_types:
        raise ValueError(f"Expected 'monetary' cost class, found {sorted(cost_types)}")
    return array.sel(costs="monetary")


def _sum_for_techs(array: xr.DataArray, tech_ids: list[str]) -> float:
    if "techs" not in array.dims:
        raise ValueError(f"{array.name!r} has no 'techs' dimension")
    amount = float(_monetary(array).sel(techs=tech_ids).fillna(0).sum().item())
    if not math.isfinite(amount):
        raise ValueError(f"Non-finite cost in {array.name!r}")
    return amount


def read_case_costs(path: Path) -> dict[str, str | int | float]:
    """Read objective and ALL storage costs from one optimal model's results."""
    case = model_case(path)
    case["experiment"] = case["experiment"].upper()
    case["country"] = case["country"].upper()
    with xr.open_dataset(path, group="results") as ds:
        for field in ("min_cost_optimisation", "cost"):
            if field not in ds:
                raise ValueError(f"Missing {field!r} in results for {path.name}")

        objective = ds["min_cost_optimisation"].squeeze(drop=True)
        if objective.ndim != 0:
            raise ValueError("Expected a scalar min_cost_optimisation objective")
        system_cost = float(objective.item())
        if not (math.isfinite(system_cost) and system_cost > 0):
            raise ValueError(f"Invalid system objective: {system_cost}")

        costs = ds["cost"]
        if "techs" not in costs.dims:
            raise ValueError("Calliope 'cost' must be indexed by 'techs'")
        storage_ids = [str(tech) for tech in costs.coords["techs"].values
                       if _is_storage(str(tech))]
        if not storage_ids:
            raise ValueError("No recognised storage technologies in results")

        storage_cost = _sum_for_techs(costs, storage_ids)
        if storage_cost < 0:
            raise ValueError(f"Negative aggregate storage expenditure: {storage_cost}")

        # Optional breakdown for verification only. These are components of
        # `cost`, not amounts to add to the storage total again.
        component_names = {
            "storage_investment": "cost_investment_annualised",
            "storage_fixed_opex": "cost_operation_fixed",
            "storage_variable_opex": "cost_operation_variable",
        }
        components = {
            output: _sum_for_techs(ds[name], storage_ids) if name in ds else float("nan")
            for output, name in component_names.items()
        }
        if all(math.isfinite(v) for v in components.values()):
            subtotal = sum(components.values())
            if not math.isclose(subtotal, storage_cost, rel_tol=1e-5, abs_tol=1e-5):
                warnings.warn(
                    f"{path.name}: storage cost {storage_cost:.6g} differs from "
                    f"component sum {subtotal:.6g}; check custom math.",
                    stacklevel=2,
                )

    return {
        **case,
        "system_cost": system_cost,
        "storage_cost": storage_cost,
        "storage_technology_count": len(storage_ids),
        **components,
    }


def normalise_against_b1(raw: pd.DataFrame) -> pd.DataFrame:
    """Pair every experiment with its own country/weather B1 benchmark."""
    keys = ["country", "weather"]
    if raw.duplicated([*keys, "experiment"]).any():
        duplicates = raw.loc[raw.duplicated([*keys, "experiment"], keep=False),
                             [*keys, "experiment"]]
        raise ValueError(f"Duplicate cases: {duplicates.to_dict('records')}")

    reference = raw.loc[raw["experiment"] == "B1", [*keys, "system_cost", "storage_cost"]]
    if reference.empty:
        raise ValueError("No B1 results found; cannot normalise the other experiments")
    reference = reference.rename(columns={
        "system_cost": "b1_system_cost", "storage_cost": "b1_storage_cost",
    })
    merged = raw.merge(reference, on=keys, how="left", validate="many_to_one")
    missing = merged["b1_system_cost"].isna()
    if missing.any():
        skipped = merged.loc[missing, "case"].tolist()
        warnings.warn(f"Skipping results without matching B1: {skipped}", stacklevel=2)
        merged = merged.loc[~missing].copy()
    if merged.empty:
        raise ValueError("No experiments could be matched with a B1 benchmark")

    invalid = (merged["b1_system_cost"] <= 0) | (merged["b1_storage_cost"] <= 0)
    if invalid.any():
        cases = merged.loc[invalid, "case"].tolist()
        raise ValueError(f"Cannot normalise against non-positive B1 costs: {cases}")

    merged["system_relative"] = merged["system_cost"] / merged["b1_system_cost"]
    merged["storage_relative"] = merged["storage_cost"] / merged["b1_storage_cost"]
    merged["storage_share_of_system"] = merged["storage_cost"] / merged["system_cost"]
    order = {exp: index for index, exp in enumerate(EXPERIMENT_ORDER)}
    return (merged.assign(_experiment_rank=merged["experiment"].map(order))
            .sort_values(["country", "weather", "_experiment_rank"])
            .drop(columns="_experiment_rank").reset_index(drop=True))


def _draw_panel(ax: plt.Axes, frame: pd.DataFrame, metric: str, ylabel: str) -> None:
    ax.set_axisbelow(True)
    ax.axhline(1.0, color="0.58", linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.grid(axis="y", color="0.86", linewidth=0.6)

    for index, experiment in enumerate(EXPERIMENT_ORDER):
        values = frame.loc[frame["experiment"] == experiment, metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        # Exactly the same x coordinate for every weather window: no jitter.
        ax.scatter(
            np.full(values.size, index), values,
            s=26, marker="o", facecolor=POINT_COLOUR, edgecolor="0.22",
            linewidth=0.4, alpha=0.70, zorder=3,
        )
        ax.hlines(np.median(values), index - 0.23, index + 0.23,
                  color="black", linewidth=2.3, zorder=4)

    valid = frame.loc[frame["experiment"].isin(EXPERIMENT_ORDER), metric]
    valid = pd.to_numeric(valid, errors="coerce").dropna()
    if valid.empty:
        raise ValueError(f"No values to plot for {metric}")
    spread = max(float(valid.max() - valid.min()), 0.02)
    padding = max(0.015, 0.13 * spread)
    ax.set_ylim(max(0.0, min(float(valid.min()), 1.0) - padding),
                max(float(valid.max()), 1.0) + padding)
    ax.set_xlim(-0.5, len(EXPERIMENT_ORDER) - 0.5)
    ax.set_ylabel(ylabel, labelpad=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("0.45")
    ax.tick_params(direction="out", length=3, width=0.75, colors="0.25")


def plot_relative_costs(frame: pd.DataFrame, output_path: Path) -> None:
    """Make a two-panel Elsevier single-column figure (10 pt throughout)."""
    with plt.rc_context({
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "pdf.fonttype": 42,
    }):
        fig, axes = plt.subplots(2, 1, sharex=True,
                                 figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
        fig.subplots_adjust(left=0.265, right=0.965, top=0.915,
                            bottom=0.105, hspace=0.20)

        _draw_panel(axes[0], frame, "system_relative", "Relative system cost")
        _draw_panel(axes[1], frame, "storage_relative", "Relative storage cost")
        axes[1].set_xticks(range(len(EXPERIMENT_ORDER)), EXPERIMENT_ORDER)
        axes[1].set_xlabel("Representation", labelpad=6)
        axes[0].tick_params(axis="x", which="both", bottom=False)

        handles = [
            Line2D([], [], color="none", marker="o", markersize=5,
                   markerfacecolor=POINT_COLOUR, markeredgecolor="0.22",
                   markeredgewidth=0.4, label="Weather window"),
            Line2D([], [], color="black", linewidth=2.3, label="Median"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=2,
                   frameon=False, bbox_to_anchor=(0.50, 0.99),
                   handlelength=1.6, columnspacing=1.0)

        fig.savefig(output_path)
        plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Folder of Calliope .nc results")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination for PDF and CSV")
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.expanduser().glob("*.nc")
                   if not path.name.endswith(".pending.nc"))
    if not files:
        parser.error(f"No .nc results found in {args.input_dir}")

    rows: list[dict[str, str | int | float]] = []
    for path in files:
        print(f"Reading {path.name}", flush=True)
        try:
            record = read_case_costs(path)
            if record["country"] not in COUNTRIES or record["experiment"] not in EXPERIMENT_ORDER:
                warnings.warn(f"Skipping unselected representation: {path.name}",
                              stacklevel=2)
                continue
            rows.append(record)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {type(exc).__name__}: {exc}",
                          stacklevel=2)

    if not rows:
        parser.error("No readable results for selected representations")

    compared = normalise_against_b1(pd.DataFrame(rows))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / "relative_system_and_storage_costs.pdf"
    csv_path = args.output_dir / "relative_system_and_storage_costs.csv"
    plot_relative_costs(compared, figure_path)
    # compared.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}; {len(compared)} paired cases across "
          f"{compared[['country', 'weather']].drop_duplicates().shape[0]} windows")


if __name__ == "__main__":
    main()
