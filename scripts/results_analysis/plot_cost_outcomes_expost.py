"""Compare economic benefits and ex-post UHS costing consequences (Calliope 0.7).

Two 90 mm, one-axis paper figures with paired offset SYSTEM/STORAGE cost points:

  1. expost_economic_benefit.pdf: ORIGINAL model costs relative to each
     country/weather's B1 benchmark. The default is percent change;
     --benefit-unit ratio restores relative costs (B1 = 1).
  2. expost_repricing_impact.pdf: percent changes caused by repricing the
     selected UHS installations ex post. Zero means no repricing adjustment.

Both figures use no jitter: every weather/country case falls at its exact
formulation + channel offset. Black segments indicate channel medians.
The representation check flags cases whose EX-POST CORRECTED system cost exceeds
its paired EX-POST CORRECTED B1 benchmark. This is not re-optimisation.

Three-stage workflow (keep the original cost-model repricer):
  # CEM environment
  pixi run python -m scripts.results_analysis.plot_cost_outcomes_expost export --countries NLD

  # UHS cost-model environment
  pixi run python ../uhs_cem_representation/scripts/results_analysis/reprice_uhs_designs.py `
    --input ../uhs_cem_representation/results/costs_expost/expost_uhs_designs.csv `
    --output ../uhs_cem_representation/results/costs_expost/expost_uhs_repriced.csv `
    --cost-module scripts/experiments/cost_rates.py `
    --cost-samples outputs/cost_rate_samples/uhs_cost_rates.parquet

  # CEM environment
  pixi run python -m scripts.results_analysis.plot_cost_outcomes_expost plot --countries NLD

The original-cost figure can also be made with only the first stage's CSVs;
the correction figure is skipped until ex-post repricing is available.
For comparison, B1 is matched separately for EACH country and weather window.
Only annualised UHS investment is re-priced; no re-optimisation takes place.
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
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

try:
    from ._plots import model_case, storage_family, capacity_for_tech
except ImportError:
    from _plots import model_case, storage_family, capacity_for_tech

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "capacity_expansion_models"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "costs_expost"
EXPERIMENT_ORDER = ("B1", "P1", "F1", "FX1",)
UHS = ("salt_cavern", "aquifer", "depleted_gas_field", "lined_rock_cavern")
LABEL = {"salt_cavern": "SC", "aquifer": "SA", "depleted_gas_field": "DGF", "lined_rock_cavern": "LRC"}
BATTERY_RE = re.compile(r"battery_\d+(?:\.\d+)?h$", re.I)
FONT_SIZE = 9
FIGURE_WIDTH_MM = 90
DEFAULT_HEIGHT_MM = 90
MM_TO_INCH = 1 / 25.4
PLASMA = plt.get_cmap("plasma")
CHANNEL_COLOURS = {"system": PLASMA(0.16), "storage": PLASMA(0.73)}
CHANNEL_MARKERS = {"system": "o", "storage": "s"}
CHANNEL_LABELS = {"system": "Total System cost", "storage": "Total Storage cost"}
EXPERIMENT_LABELS = {"B1": "B", "P1": "P", "F1": "F", "FX1": "FX"}


def _is_storage(t: str) -> bool:
    return t in {"pumped_hydro", "pumped_hydro_storage", "pumped_storage"} or bool(BATTERY_RE.fullmatch(t)) or storage_family(t) is not None


def monetary(da: xr.DataArray) -> xr.DataArray:
    if "costs" not in da.dims:
        return da
    values = {str(v) for v in da.coords["costs"].values}
    if "monetary" not in values:
        raise ValueError(f"No monetary cost class (found {values})")
    return da.sel(costs="monetary")


def tech_cost(ds: xr.Dataset, field: str, techs: list[str]) -> float:
    if field not in ds:
        raise KeyError(f"Missing required Calliope cost output {field}")
    arr = monetary(ds[field])
    if "techs" not in arr.dims:
        raise ValueError(f"{field} is missing techs dimension")
    if not techs:
        return 0.0
    total = float(arr.sel(techs=techs).fillna(0).sum().item())
    if not math.isfinite(total):
        raise ValueError(f"Non-finite {field} cost")
    return total


def scalar_variable(ds: xr.Dataset, field: str) -> float:
    arr = monetary(ds[field]).squeeze(drop=True)
    if arr.size != 1:
        raise ValueError(f"{field} should be scalar, got {arr.shape}")
    value = float(arr.item())
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}")
    return value


def read_case(path: Path) -> tuple[dict, list[dict]]:
    case = model_case(path)
    case.update(country=case["country"].upper(), experiment=case["experiment"].upper(), weather=case["weather"].lower())
    designs = []
    with xr.open_dataset(path, group="results") as ds:
        for field in ("cost", "min_cost_optimisation", "cost_investment_annualised", "storage_cap", "flow_in_cap", "flow_out_cap"):
            if field not in ds:
                raise ValueError(f"{path.name}: missing {field}")
        all_techs = [str(t) for t in ds.coords["techs"].values]
        storage_techs = [t for t in all_techs if _is_storage(t)]
        geological_techs = [t for t in all_techs if storage_family(t) in UHS]
        row = {**case, "file": str(path),
               "system_cost": scalar_variable(ds, "min_cost_optimisation"),
               "storage_cost": tech_cost(ds, "cost", storage_techs),
               "uhs_cost": tech_cost(ds, "cost", geological_techs),
               "uhs_investment_annualised": tech_cost(ds, "cost_investment_annualised", geological_techs),
               "uhs_variants_installed": 0}
        for tech in geological_techs:
            e = capacity_for_tech(ds, "storage_cap", tech)
            pin = capacity_for_tech(ds, "flow_in_cap", tech)
            pout = capacity_for_tech(ds, "flow_out_cap", tech)
            if max(e, pin, pout) < 1e-6:
                continue
            if min(e, pin, pout) < -1e-6:
                raise ValueError(f"Negative installed capacity: {tech}")
            inv = tech_cost(ds, "cost_investment_annualised", [tech])
            rec = {**case, "tech_id": tech, "technology": storage_family(tech),
                   "storage_MWh": e, "injection_MW": pin, "withdrawal_MW": pout,
                   "cem_investment_annualised": inv,
                   "cem_total_cost": tech_cost(ds, "cost", [tech]),
                   "EPR_days": e / pout / 24. if pout > 0 else float("nan"),
                   "WTIR": pout / pin if pin > 0 else float("nan"),
                   "design_status": "ok" if min(e, pin, pout) > 1e-6 else "incomplete_capacities"}
            for field in ("cost_operation_fixed", "cost_operation_variable"):
                rec[field] = tech_cost(ds, field, [tech]) if field in ds else float("nan")
            designs.append(rec)
        row["uhs_variants_installed"] = len(designs)
        if row["system_cost"] <= 0 or row["storage_cost"] < 0:
            raise ValueError(f"Non-positive/negative costs: {path.name}")
    return row, designs


def export_cases(input_dir: Path, output_dir: Path, countries: set[str] | None, experiments: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(p for p in input_dir.glob("*.nc") if not p.name.endswith(".pending.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc results in {input_dir}")
    records, designs = [], []
    for path in files:
        try:
            case = model_case(path)
        except ValueError:
            continue
        if countries is not None and case["country"].upper() not in countries:
            continue
        if case["experiment"].upper() not in experiments:
            continue
        try:
            row, variants = read_case(path)
            print("Read", path.name, "UHS variants:", len(variants), flush=True)
            records.append(row)
            designs.extend(variants)
        except Exception as exc:
            warnings.warn(f"Failed to read {path.name}: {exc}", stacklevel=2)
    if not records:
        raise ValueError("No readable selected model results")
    cases = pd.DataFrame.from_records(records).sort_values(["country", "weather", "experiment"])
    if cases.duplicated(["country", "weather", "experiment"]).any():
        raise ValueError("Duplicate country/weather/experiment cases")
    design_fields = ["country", "weather", "experiment", "case", "tech_id", "technology",
                     "storage_MWh", "injection_MW", "withdrawal_MW", "EPR_days", "WTIR",
                     "cem_investment_annualised", "cem_total_cost", "cost_operation_fixed",
                     "cost_operation_variable", "design_status"]
    variant_frame = pd.DataFrame.from_records(designs, columns=design_fields)
    if not variant_frame.empty and variant_frame.duplicated(["case", "tech_id"]).any():
        raise ValueError("Duplicate UHS variant rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_dir / "expost_case_costs_original.csv", index=False)
    variant_frame.to_csv(output_dir / "expost_uhs_designs.csv", index=False)
    print(f"Exported {len(cases)} cases, {len(variant_frame)} installed UHS designs to {output_dir}")
    return cases, variant_frame


def attach_corrections(cases: pd.DataFrame, designs: pd.DataFrame, repriced: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if designs.empty:
        output = cases.copy()
        output["repricing_expected"] = 0
        output["repricing_valid"] = 0
        output["correction_complete"] = True
        output["uhs_correction_annualised"] = 0.0
        output["system_cost_corrected"] = output["system_cost"]
        output["storage_cost_corrected"] = output["storage_cost"]
        output["system_effect_pct"] = 0.0
        output["storage_effect_pct"] = np.where(output["storage_cost"] > 0, 0.0, np.nan)
        return output, pd.DataFrame(columns=[*designs.columns, "included", "representation_error_pct"])

    required = {"case", "tech_id", "status", "represented_tcr_eur", "detailed_tcr_eur", "adjustment_factor"}
    missing = required - set(repriced.columns)
    if missing:
        raise ValueError(f"Repriced CSV missing columns {sorted(missing)}. Re-run the updated re-pricer.")
    if repriced.duplicated(["case", "tech_id"]).any():
        raise ValueError("Duplicate repricing records")
    merged = designs.merge(repriced, on=["case", "tech_id"], how="left", validate="one_to_one",
                           suffixes=("", "_reprice"), indicator=True)
    for col in ("adjustment_factor", "represented_tcr_eur", "detailed_tcr_eur", "cem_investment_annualised"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    ratio = merged["detailed_tcr_eur"] / merged["represented_tcr_eur"]
    valid = ((merged["_merge"] == "both") & (merged["status"] == "ok") &
             (merged["design_status"] == "ok") &
             (merged["represented_tcr_eur"] > 0) & (merged["detailed_tcr_eur"] > 0) &
             (merged["cem_investment_annualised"] > 0) &
             np.isfinite(ratio) & np.isfinite(merged["adjustment_factor"]) &
             np.isclose(ratio, merged["adjustment_factor"], rtol=1e-7, atol=1e-10))
    if "cem_investment_annualised_reprice" in merged:
        recorded = pd.to_numeric(merged["cem_investment_annualised_reprice"], errors="coerce")
        valid &= np.isclose(recorded, merged["cem_investment_annualised"], rtol=1e-8, atol=1e-5)

    merged["included"] = valid
    merged["corrected_investment_annualised"] = np.where(valid, merged["cem_investment_annualised"] * ratio, np.nan)
    merged["delta_investment_annualised"] = merged["corrected_investment_annualised"] - merged["cem_investment_annualised"]
    merged["representation_error_pct"] = np.where(
        valid, 100.0 * (merged["represented_tcr_eur"] - merged["detailed_tcr_eur"]) /
        merged["detailed_tcr_eur"], np.nan)

    totals = merged.groupby("case", as_index=False).agg(
        repricing_expected=("tech_id", "size"), repricing_valid=("included", "sum"),
        uhs_correction_annualised=("delta_investment_annualised", lambda x: x.sum(min_count=1)),
        represented_uhs_tcr_eur=("represented_tcr_eur", lambda x: x.sum(min_count=1)),
        detailed_uhs_tcr_eur=("detailed_tcr_eur", lambda x: x.sum(min_count=1)),
    )
    output = cases.merge(totals, on="case", how="left", validate="one_to_one")
    output["repricing_expected"] = output["repricing_expected"].fillna(0).astype(int)
    output["repricing_valid"] = output["repricing_valid"].fillna(0).astype(int)
    output["correction_complete"] = output["repricing_expected"] == output["repricing_valid"]
    output.loc[output["repricing_expected"] == 0, ["uhs_correction_annualised", "represented_uhs_tcr_eur", "detailed_uhs_tcr_eur"]] = 0.0
    output.loc[~output["correction_complete"], ["uhs_correction_annualised", "represented_uhs_tcr_eur", "detailed_uhs_tcr_eur"]] = np.nan
    output["uhs_representation_error_pct"] = np.where(
        (output["correction_complete"]) & (output["detailed_uhs_tcr_eur"] > 0),
        100.0 * (output["represented_uhs_tcr_eur"] - output["detailed_uhs_tcr_eur"]) /
        output["detailed_uhs_tcr_eur"].replace(0.0, np.nan), np.nan)
    output["system_cost_corrected"] = output["system_cost"] + output["uhs_correction_annualised"]
    output["storage_cost_corrected"] = output["storage_cost"] + output["uhs_correction_annualised"]
    output["system_effect_pct"] = 100.0 * output["uhs_correction_annualised"] / output["system_cost"]
    output["storage_effect_pct"] = 100.0 * output["uhs_correction_annualised"] / output["storage_cost"].replace(0.0, np.nan)
    if (output.loc[output["correction_complete"], ["system_cost_corrected", "storage_cost_corrected"]] < -1e-9).any().any():
        raise ValueError("Negative corrected costs: check cost/annualisation units")
    return output, merged.drop(columns="_merge")


def compare_b1(cases: pd.DataFrame) -> pd.DataFrame:
    """Pair each scenario with its own B1, independently of correction success.

    Original comparisons must NOT be filtered on available ex-post repricing.
    A corrected comparison requires complete repricing of both its own result
    and the paired B1 result.
    """
    reference = cases.loc[cases["experiment"] == "B1", [
        "country", "weather", "system_cost", "storage_cost",
        "system_cost_corrected", "storage_cost_corrected", "correction_complete",
    ]].copy()
    reference = reference.rename(columns={
        "system_cost": "b1_system", "storage_cost": "b1_storage",
        "system_cost_corrected": "b1_system_corrected",
        "storage_cost_corrected": "b1_storage_corrected",
        "correction_complete": "b1_complete",
    })
    if reference.duplicated(["country", "weather"]).any():
        raise ValueError("Duplicate B1 benchmark for country/weather")
    pairs = cases.merge(reference, on=["country", "weather"], how="left", validate="many_to_one")
    valid_original = (
        (pairs["b1_system"] > 0) & (pairs["b1_storage"] > 0)
        & np.isfinite(pairs["system_cost"]) & np.isfinite(pairs["storage_cost"])
    )
    valid_corrected = (
        valid_original & pairs["correction_complete"].fillna(False)
        & pairs["b1_complete"].fillna(False)
        & (pairs["b1_system_corrected"] > 0)
        & (pairs["b1_storage_corrected"] > 0)
        & np.isfinite(pairs["system_cost_corrected"])
        & np.isfinite(pairs["storage_cost_corrected"])
    )
    pairs["matched_original"] = valid_original
    pairs["matched_corrected"] = valid_corrected
    pairs["matched"] = valid_corrected  # retain legacy downstream CSV name
    for component in ("system", "storage"):
        pairs[f"{component}_original_relative"] = np.where(
            valid_original,
            pairs[f"{component}_cost"] / pairs[f"b1_{component}"], np.nan,
        )
        pairs[f"{component}_corrected_relative"] = np.where(
            valid_corrected,
            pairs[f"{component}_cost_corrected"] / pairs[f"b1_{component}_corrected"],
            np.nan,
        )
        pairs[f"{component}_original_change_pct"] = 100 * (
            pairs[f"{component}_original_relative"] - 1.0
        )
    pairs["corrected_system_exceeds_b1"] = (
        valid_corrected & (pairs["system_corrected_relative"] > 1.0 + 1e-9)
    )
    return pairs


def _values(frame: pd.DataFrame, experiment: str, column: str) -> np.ndarray:
    values = pd.to_numeric(
        frame.loc[frame["experiment"] == experiment, column], errors="coerce"
    ).to_numpy(float)
    return values[np.isfinite(values)]


def _plot_cost_channels(
    data: pd.DataFrame,
    output: Path,
    fields: dict[str, str],
    *,
    ylabel: str,
    reference: float,
    height_mm: float,
    svg: bool,
    mark_crossings: bool = False,
) -> None:
    """One Axes, with system and storage at side-by-side x offsets for each formulation."""
    used = []
    offset = 0.17
    median_width = 0.115
    with plt.rc_context({
        "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, height_mm * MM_TO_INCH))
        fig.subplots_adjust(left=0.14, right=0.99, top=0.9, bottom=0.1)
        ax.axhline(reference, color="0.52", linewidth=0.9,
                   linestyle=(0, (3, 2)), zorder=1)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        for index, experiment in enumerate(EXPERIMENT_ORDER):
            local = data.loc[data["experiment"] == experiment]
            for channel, side in (("system", -1), ("storage", +1)):
                xpos = index + side * offset
                series = pd.to_numeric(local[fields[channel]], errors="coerce")
                valid = series.notna() & np.isfinite(series.to_numpy(float))
                yvals = series.loc[valid].to_numpy(float)
                if not yvals.size:
                    continue
                used.extend(yvals.tolist())
                ax.scatter(np.full(len(yvals), xpos), yvals, s=26,
                           facecolor=CHANNEL_COLOURS[channel],
                           marker=CHANNEL_MARKERS[channel],
                           edgecolor="white", linewidth=0.35,
                           alpha=0.86, zorder=3)
                ax.hlines(float(np.median(yvals)), xpos - median_width,
                          xpos + median_width, color="black", linewidth=1.7, zorder=5)
                if mark_crossings and channel == "system" and "corrected_system_exceeds_b1" in local:
                    crossings = local.loc[valid & local["corrected_system_exceeds_b1"].fillna(False)]
                    if not crossings.empty:
                        ring_y = pd.to_numeric(crossings[fields[channel]], errors="coerce").to_numpy(float)
                        ax.scatter(np.full(len(ring_y), xpos), ring_y, marker="o", s=63,
                                   facecolor="none", edgecolor="black", linewidth=0.95,
                                   zorder=6)
        ax.set_xlim(-0.53, len(EXPERIMENT_ORDER) - 0.47)
        ax.set_xticks(range(len(EXPERIMENT_ORDER)),
                      [EXPERIMENT_LABELS[e] for e in EXPERIMENT_ORDER])
        ax.set_xlabel("Cost formulation", labelpad=4)
        ax.set_ylabel(ylabel, labelpad=5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("0.5")
        ax.tick_params(axis="x", length=0, pad=5)
        ax.tick_params(axis="y", length=3)
        if used:
            lo, hi = min(*used, reference), max(*used, reference)
            spread = max(hi - lo, 0.015 if abs(reference) == 1 else 0.15)
            pad = max(0.095 * spread, 0.005 if abs(reference) == 1 else 0.065)
            ax.set_ylim(lo - pad, round(hi + pad,0))
        else:
            ax.set_ylim(reference - 1, reference + 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        legend = [
            Line2D([], [], linestyle="none", marker=CHANNEL_MARKERS[ch], markersize=6,
                   markerfacecolor=CHANNEL_COLOURS[ch], markeredgecolor="white",
                   label=CHANNEL_LABELS[ch])
            for ch in ("system", "storage")
        ]
        # Black bars are identified in the caption, to keep the legend at one row.
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.015),
                  frameon=False, ncol=2, handlelength=0.8,
                  columnspacing=0.9, handletextpad=0.35)
        output.parent.mkdir(parents=True, exist_ok=True)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        fig.savefig(output)
        if svg:
            fig.savefig(output.with_suffix(".svg"))
        plt.close(fig)
        print("Saved", output)


def plot_economic_benefit(cases: pd.DataFrame, output: Path,
                          *, unit: str, height_mm: float, svg: bool) -> None:
    """Original-cost benefit, paired to B1 by country and weather."""
    data = cases.loc[cases["matched_original"]].copy()
    if data.empty:
        raise ValueError("No country/weather B1 matches available for original-cost comparison")
    if unit == "ratio":
        fields = {"system": "system_original_relative", "storage": "storage_original_relative"}
        label, reference = "Cost relative to B (-)", 1.0
    elif unit == "percent":
        fields = {"system": "system_original_change_pct", "storage": "storage_original_change_pct"}
        label, reference = "Cost change relative to B (%)", 0.0
    else:
        raise ValueError(f"Invalid economic benefit unit: {unit}")
    _plot_cost_channels(data, output, fields, ylabel=label, reference=reference,
                        height_mm=height_mm, svg=svg)


def plot_repricing_impact(cases: pd.DataFrame, output: Path,
                          *, height_mm: float, svg: bool,
                          mark_crossings: bool) -> None:
    """Ex-post cost correction per design; no re-optimisation."""
    data = cases.loc[cases["correction_complete"]].copy()
    if data.empty:
        raise ValueError("No complete UHS repricing outcomes are available")
    _plot_cost_channels(
        data, output,
        {"system": "system_effect_pct", "storage": "storage_effect_pct"},
        ylabel="Ex-post cost adjustment (%)", reference=0.0,
        height_mm=height_mm, svg=svg, mark_crossings=mark_crossings,
    )


def plot_technology_errors(designs: pd.DataFrame, output: Path) -> None:
    valid = designs.loc[designs["included"]].copy()
    if valid.empty:
        return
    experiments = [e for e in EXPERIMENT_ORDER if e in set(valid["experiment"])]
    with plt.rc_context({"font.size": FONT_SIZE, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(len(experiments), 1, figsize=(FIGURE_WIDTH_MM * MM_TO_INCH, max(2.2, 1.65 * len(experiments)+.75)), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.25, right=.97, top=.93, bottom=.13, hspace=.26)
        for ax, exp in zip(axes[:,0], experiments):
            local = valid.loc[valid["experiment"] == exp]
            ax.axhline(0, color="0.5", ls=(0, (3, 2)), lw=.8)
            ax.grid(axis="y", color="0.88", lw=.55)
            for j, family in enumerate(UHS):
                vals = pd.to_numeric(local.loc[local["technology"] == family, "representation_error_pct"], errors="coerce").dropna().to_numpy()
                if not vals.size:
                    continue
                ax.scatter(np.full(len(vals), j), vals, s=24, c=[CHANNEL_COLOURS["storage"]], edgecolor="0.3", lw=.35, alpha=.78)
                ax.hlines(np.median(vals), j-.11, j+.11, lw=2, color="black")
            ax.set_xlim(-.55, len(UHS)-.45)
            ax.set_ylabel(f"{exp}: error (%)", fontsize=FONT_SIZE)
            ax.spines[["right", "top"]].set_visible(False)
        axes[-1,0].set_xticks(range(len(UHS)), [LABEL[t] for t in UHS])
        handles = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=CHANNEL_COLOURS["storage"], markeredgecolor="0.2", markersize=5, label="Cases"),
            Line2D([], [], color="black", linewidth=2, label="Median"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.995), ncol=2, frameon=False, handlelength=1.3, fontsize=FONT_SIZE)
        fig.savefig(output)
        plt.close(fig)
        print("Saved", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "plot"],
                        help="Export CEM designs or plot original and corrected costs")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENT_ORDER))
    parser.add_argument("--benefit-unit", choices=("percent", "ratio"), default="percent",
                        help="Economic figure: percent change from B (default), or cost/B ratio")
    parser.add_argument("--height-mm", type=float, default=DEFAULT_HEIGHT_MM,
                        help="Physical height for each 90-mm figure (default 104 mm)")
    parser.add_argument("--svg", action="store_true", help="Also export editable SVGs")
    parser.add_argument("--crossing-markers", action="store_true",
                        help="Omit outlines on corrections where system cost exceeds corrected B")
    parser.add_argument("--with-diagnostics", action="store_true",
                        help="Also save UHS technology-level approximation errors")
    args = parser.parse_args()
    if args.height_mm < 60:
        parser.error("--height-mm must be at least 60")
    countries = {x.upper() for x in args.countries} if args.countries else None
    experiments = {x.upper() for x in args.experiments}
    if "B1" not in experiments:
        parser.error("B1 must be included for paired economic benchmarking")
    out = args.output_dir
    if args.mode == "export":
        export_cases(args.input_dir, out, countries, experiments)
        return

    cases_file, designs_file, reprice_file = (
        out / n for n in ("expost_case_costs_original.csv", "expost_uhs_designs.csv",
                         "expost_uhs_repriced.csv")
    )
    if not cases_file.is_file():
        parser.error(f"Missing {cases_file}; run export first")
    cases = pd.read_csv(cases_file)
    cases = cases.loc[cases["experiment"].isin(experiments)
                      & (cases["country"].isin(countries) if countries else True)].copy()
    if cases.empty:
        parser.error("No exported cases match selected countries/formulations")

    if not designs_file.is_file():
        parser.error(f"Missing {designs_file}; run export first")
    designs = pd.read_csv(designs_file)
    designs = designs.loc[designs["case"].isin(cases["case"])].copy()

    reprice_available = reprice_file.is_file() or designs.empty
    if reprice_file.is_file():
        repriced = pd.read_csv(reprice_file)
        repriced = repriced.loc[repriced["case"].isin(cases["case"])].copy()
    else:
        repriced = pd.DataFrame(columns=[
            "case", "tech_id", "status", "adjustment_factor",
            "detailed_tcr_eur", "represented_tcr_eur",
        ])
        if not designs.empty:
            warnings.warn("Repriced CSV is missing: economic-benefit figure will still be "
                          "created from original costs; repricing impact will be skipped.",
                          stacklevel=1)

    # Stage 1 remains valid even without stage 2, so plot its economic comparison.
    if reprice_available:
        corrected, detailed = attach_corrections(cases, designs, repriced)
    else:
        corrected = cases.copy()
        corrected["correction_complete"] = False
        corrected["system_cost_corrected"] = np.nan
        corrected["storage_cost_corrected"] = np.nan
        corrected["system_effect_pct"] = np.nan
        corrected["storage_effect_pct"] = np.nan
        detailed = pd.DataFrame()

    comparisons = compare_b1(corrected)
    out.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(out / "expost_cost_comparison.csv", index=False)
    plot_economic_benefit(comparisons, out / "expost_economic_benefit.pdf",
                          unit=args.benefit_unit, height_mm=args.height_mm, svg=args.svg)

    if reprice_available:
        corrected.to_csv(out / "expost_case_costs_corrected.csv", index=False)
        detailed.to_csv(out / "expost_uhs_repricing_audit.csv", index=False)
        plot_repricing_impact(comparisons, out / "expost_repricing_impact.pdf",
                              height_mm=args.height_mm, svg=args.svg,
                              mark_crossings=args.crossing_markers)
        if args.with_diagnostics and not detailed.empty:
            plot_technology_errors(detailed, out / "expost_uhs_representation_error.pdf")
        unresolved = comparisons.loc[~comparisons["correction_complete"], "case"].tolist()
        if unresolved:
            warnings.warn(f"Excluded {len(unresolved)} incompletely repriced cases from "
                          f"correction plot: {unresolved[:20]}", stacklevel=1)
        crossings = comparisons.loc[comparisons["corrected_system_exceeds_b1"],
                                    ["country", "weather", "experiment"]]
        if crossings.empty:
            print("No ex-post corrected system costs exceed the paired corrected B1 benchmark")
        else:
            print("Ex-post corrected system cost exceeds paired corrected B1 in:")
            for row in crossings.itertuples(index=False):
                print(f"  {row.country} / {row.weather} / {row.experiment}")
    original_count = int(comparisons["matched_original"].sum())
    correction_count = int(comparisons["matched_corrected"].sum())
    print(f"Original-cost B1 comparisons: {original_count}; "
          f"complete corrected B1 comparisons: {correction_count}")


if __name__ == "__main__":
    main()
