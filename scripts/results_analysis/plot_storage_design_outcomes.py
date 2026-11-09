"""Plot invested long-duration storage designs across Calliope experiments.

From the project root::

    pixi run python -m scripts.results_analysis.storage_design_plots

Defaults:
  input:  results/all_uhs_3h/*.nc
  output: results/figures/

Optional::

    pixi run python -m scripts.results_analysis.storage_design_plots \
        --input-dir results/all_uhs_3h --output-dir results/figures

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
    from ._plots import read_storage_investments
except ImportError:  # Also allow `python storage_design_plots.py`.
    from _plots import read_storage_investments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "hpc" / 'results_2h'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"

TECHNOLOGY_ORDER = (
    "compressed_air_storage", "lined_rock_cavern", "salt_cavern",
    "aquifer", "depleted_gas_field",
)
TECHNOLOGY_LABELS = {
    "compressed_air_storage": "CAES",
    "lined_rock_cavern": "Lined rock cavern",
    "salt_cavern": "Salt cavern",
    "aquifer": "Saline aquifer",
    "depleted_gas_field": "Depleted gas field",
}
# Fixed technology shapes across runs and figures.
TECHNOLOGY_MARKERS = {
    "compressed_air_storage": "o",
    "lined_rock_cavern": "v",
    "salt_cavern": "s",
    "aquifer": "D",
    "depleted_gas_field": "^",
}
EXPERIMENT_ORDER = ("B1", "P1", "F1", "FX1", "FX2", "FX4")


def _experiment_key(name: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Z]+)(\d+)([A-Z]*)", name.upper())
    return (match.group(1), int(match.group(2)), match.group(3)) if match else (name, 0, "")


def _style_keys(frame: pd.DataFrame) -> tuple[dict[str, object], dict[str, str]]:
    """Use technology symbols and ordered plasma colours for representations."""
    active_techs = [tech for tech in TECHNOLOGY_ORDER
                    if tech in set(frame["technology"])]
    markers = {tech: TECHNOLOGY_MARKERS[tech] for tech in active_techs}

    present = set(frame["experiment"].str.upper())
    experiments = [exp for exp in EXPERIMENT_ORDER if exp in present]
    experiments.extend(sorted(present.difference(EXPERIMENT_ORDER), key=_experiment_key))
    # Progress through plasma in experiment order; avoid its brightest yellow.
    positions = np.linspace(0.10, 0.90, len(experiments)) if len(experiments) > 1 else [0.5]
    colors = dict(zip(experiments, plt.get_cmap("plasma")(positions)))
    return colors, markers


def _plot_scatter(
    frame: pd.DataFrame, output_path: Path, *, x: str, y: str,
    xlabel: str, ylabel: str, title: str,
    log_x: bool = True, log_y: bool = False,
) -> None:
    frame = frame.loc[
        np.isfinite(frame[x]) & (frame[x] > 0)
        & np.isfinite(frame[y]) & (frame[y] > 0)
    ]
    if frame.empty:
        warnings.warn(f"No positive invested storage designs for {output_path.name}.")
        return

    colors, markers = _style_keys(frame)
    fig, ax = plt.subplots(figsize=(10, 6.3))
    for (tech, experiment), group in frame.groupby(["technology", "experiment"], sort=False):
        ax.scatter(
            group[x], group[y], s=76, color=colors[experiment.upper()],
            marker=markers[tech], edgecolors="white",
            linewidths=0.6, alpha=0.86, zorder=3,
        )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(which="major", alpha=0.25)
    ax.grid(which="minor", alpha=0.10)
    ax.spines[["top", "right"]].set_visible(False)

    tech_handles = [
        Line2D([], [], marker=marker, color="none", markerfacecolor="0.55",
               markeredgecolor="white", markersize=9, label=TECHNOLOGY_LABELS[tech])
        for tech, marker in markers.items()
    ]
    first_legend = ax.legend(
        handles=tech_handles, title="Technology", frameon=False,
        loc="upper left", bbox_to_anchor=(1.02, 1.02),
    )
    ax.add_artist(first_legend)
    exp_handles = [
        Line2D([], [], linestyle="none", marker="o",
               markerfacecolor=color, markeredgecolor="white",
               markersize=8, label=exp)
        for exp, color in colors.items()
    ]
    ax.legend(handles=exp_handles, title="Representation", frameon=False,
              loc="lower left", bbox_to_anchor=(1.02, 0.0))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_epr_vs_capacity(frame: pd.DataFrame, output_dir: Path) -> None:
    """One point per invested CAES or UHS design, in TWh and days."""
    _plot_scatter(
        frame, output_dir / "storage_epr_vs_capacity.pdf",
        x="EPR_days", y="capacity_TWh",
        ylabel="Installed storage energy capacity (TWh)",
        xlabel="Discharge duration, EPR (days)",
        title="Long-duration storage: duration versus installed capacity",
        log_x=False, log_y=False,
    )


def plot_wtir_vs_epr(frame: pd.DataFrame, output_dir: Path) -> None:
    """One point per invested CAES or UHS design; EPR is shown in days."""
    _plot_scatter(
        frame, output_dir / "storage_wtir_vs_epr.pdf",
        x="EPR_days", y="WTIR",
        xlabel="Discharge duration, EPR (days)",
        ylabel="Withdrawal / injection capacity (WTIR)",
        title="Long-duration storage: power asymmetry versus duration",
        log_x=False, log_y=False,
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
