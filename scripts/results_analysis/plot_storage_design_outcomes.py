"""Plot invested long-duration storage designs across Calliope experiments.

From the project root::

    pixi run python -m scripts.results_analysis.storage_design_plots

Defaults:
  input:  hpc/results_2h/*.nc
  output: results/figures/

Optional::

    pixi run python -m scripts.results_analysis.storage_design_plots \
        --input-dir hpc/results_2h --output-dir results/figures

Place this file and _plots.py in scripts/results_analysis/.
The figures deliberately omit batteries, but include CAES and every invested
individual geological construct (including multiple FX2 designs).
"""

from __future__ import annotations

import argparse
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
    from ._plots import read_storage_investments, model_case
except ImportError:  # Also allow `python storage_design_plots.py`.
    from _plots import read_storage_investments, model_case


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / "results_2h_no_coupling_bounds"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"
COUNTRIES = ("NLD",)

TECHNOLOGY_ORDER = (
    "compressed_air_storage", "lined_rock_cavern", "salt_cavern",
    "aquifer", "depleted_gas_field",
)
TECHNOLOGY_LABELS = {
    # "battery_4h": "4h Battery",
    # "battery_8h": "8h Battery",
    "compressed_air_storage": "CAES",
    "lined_rock_cavern": "Lined rock cavern",
    "salt_cavern": "Salt cavern",
    "aquifer": "Saline aquifer",
    "depleted_gas_field": "Depleted gas field",
}
# Fixed technology shapes across runs and figures.
TECHNOLOGY_MARKERS = {
    # "battery_4h": "o",
    # "battery_8h": "o",
    "compressed_air_storage": "^",
    "lined_rock_cavern": "v",
    "salt_cavern": "o",
    "aquifer": "D",
    "depleted_gas_field": "s",
}
EXPERIMENT_ORDER = ("B1", "P1", "F1", "FX1", "FX2", "FX4")


def _experiment_key(name: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Z]+)(\d+)([A-Z]*)", name.upper())
    return (match.group(1), int(match.group(2)), match.group(3)) if match else (name, 0, "")


def _style_keys(frame: pd.DataFrame) -> tuple[dict[str, object], dict[str, str]]:
    """Use technology symbols and stable, ordered plasma colours."""
    active_techs = [tech for tech in TECHNOLOGY_ORDER
                    if tech in set(frame["technology"])]
    markers = {tech: TECHNOLOGY_MARKERS[tech] for tech in active_techs}

    present = set(frame["experiment"].str.upper())
    experiments = [exp for exp in EXPERIMENT_ORDER if exp in present]
    experiments.extend(sorted(present.difference(EXPERIMENT_ORDER), key=_experiment_key))
    # Colour each known representation consistently, even when some are absent
    # from the input files; skip plasma's harshest yellow at the upper end.
    extras = [exp for exp in experiments if exp not in EXPERIMENT_ORDER]
    full_order = [*EXPERIMENT_ORDER, *extras]
    positions = np.linspace(0.10, 0.90, len(full_order))
    full_colors = dict(zip(full_order, plt.get_cmap("plasma")(positions)))
    colors = {exp: full_colors[exp] for exp in experiments}
    return colors, markers


def _plot_scatter(
    frame: pd.DataFrame, output_path: Path, *, x: str, y: str,
    xlabel: str, ylabel: str,
    log_x: bool = False, log_y: bool = False,
) -> None:
    frame = frame.loc[
        np.isfinite(frame[x]) & (frame[x] > 0)
        & np.isfinite(frame[y]) & (frame[y] > 0)
    ]
    if frame.empty:
        warnings.warn(f"No positive invested storage designs for {output_path.name}.")
        return

    colors, markers = _style_keys(frame)
    fig, ax = plt.subplots(figsize=(10.4, 6.1))
    # Keep both legend rows outside the axes, with no additional plot title.
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.13, top=0.85)
    for (tech, experiment), group in frame.groupby(["technology", "experiment"], sort=False):
        ax.scatter(
            group[x], group[y], s=66, color=colors[experiment.upper()],
            marker=markers[tech], edgecolors="0.22",
            linewidths=0.5, alpha=0.86, zorder=3,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.82", linewidth=0.65, alpha=0.55)
    if log_x or log_y:
        ax.grid(which="minor", color="0.88", linewidth=0.4, alpha=0.35)
    if not log_x:
        ax.set_xlim(left=0)
    if not log_y:
        ax.set_ylim(bottom=0)
    ax.margins(x=0.05, y=0.08)
    ax.tick_params(axis="both", labelsize=9.5, colors="0.25")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("0.45")

    tech_handles = [
        Line2D([], [], marker=marker, color="none", markerfacecolor="0.55",
               markeredgecolor="0.25", markeredgewidth=0.5,
               markersize=8, label=TECHNOLOGY_LABELS[tech])
        for tech, marker in markers.items()
    ]
    exp_handles = [
        Line2D([], [], linestyle="none", marker="o",
               markerfacecolor=color, markeredgecolor="0.25",
               markeredgewidth=0.5, markersize=8, label=exp)
        for exp, color in colors.items()
    ]
    # Two horizontal legend rows.  Their headings sit inline at the left,
    # leaving the data panel unobstructed and keeping full technology names.
    legend_style = dict(
        loc="center left", frameon=False, fontsize=10,
        handlelength=1.05, handletextpad=0.35,
        columnspacing=1.35, borderaxespad=0,
    )
    fig.text(0.105, 0.945, "Technology", va="center", fontsize=10, weight="700")
    fig.legend(handles=tech_handles, bbox_to_anchor=(0.255, 0.945),
               ncol=len(tech_handles), **legend_style)
    fig.text(0.105, 0.9, "Representation", va="center", fontsize=10, weight="700")
    fig.legend(handles=exp_handles, bbox_to_anchor=(0.255, 0.9),
               ncol=len(exp_handles), **legend_style)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_epr_vs_capacity(frame: pd.DataFrame, output_dir: Path) -> None:
    """One point per invested CAES or UHS design, in TWh and days."""
    _plot_scatter(
        frame, output_dir / "storage_epr_vs_capacity.pdf",
        x="EPR_days", y="capacity_TWh",
        xlabel="Duration, EPR (days)",
        ylabel="Installed storage energy capacity (TWh)",
    )


def plot_wtir_vs_epr(frame: pd.DataFrame, output_dir: Path) -> None:
    """One point per invested CAES or UHS design; EPR is shown in days."""
    _plot_scatter(
        frame, output_dir / "storage_wtir_vs_epr.pdf",
        x="EPR_days", y="WTIR",
        xlabel="Duration, EPR (days)",
        ylabel="Withdrawal to injection ratio (WTIR)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Folder containing Calliope .nc files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Destination for the two PDF figures")
    args = parser.parse_args()

    files = sorted(args.input_dir.expanduser().glob("*.nc"))
    # Ignore any interrupted NetCDF export that may be left over.
    files = [file for file in files if not file.name.endswith(".pending.nc")]
    if not files:
        parser.error(f"No .nc files found in {args.input_dir}")

    all_frames = []
    for path in files:
        print(f"Reading {path}", flush=True)
        try:
            country =  model_case(path)["country"].upper()
            if country not in COUNTRIES:
                warnings.warn(f"Skipping unselected representation: {path.name}",stacklevel=2)
                continue
            frame = read_storage_investments(path)
        except Exception as exc:
            warnings.warn(f"Skipping {path.name}: {type(exc).__name__}: {exc}")
            continue
        if not frame.empty:
            all_frames.append(frame)
    if not all_frames:
        parser.error("No readable .nc files contained invested UHS or CAES capacity")

    combined = pd.concat(all_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_epr_vs_capacity(combined, args.output_dir)
    plot_wtir_vs_epr(combined, args.output_dir)
    print(f"Plotted {len(combined)} individual investments from "
          f"{combined['case'].nunique()} result files.")


if __name__ == "__main__":
    main()
