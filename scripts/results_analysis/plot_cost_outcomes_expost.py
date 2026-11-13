"""Ex-post comparison of CEM-represented and detailed UHS investment totals (Calliope 0.7).

Three-stage, dependency-isolated workflow.

Stage 1, in the CEM environment::
  pixi run python -m scripts.results_analysis.plot_cost_outcomes_expost export `
      --countries NLD DEU GBR

Stage 2, in the UHS cost-model environment::
  pixi run python ../uhs_cem_representation/scripts/results_analysis/reprice_uhs_designs.py `
      --input ../uhs_cem_representation/results/costs_expost/expost_uhs_designs.csv `
      --output ../uhs_cem_representation/results/costs_expost/expost_uhs_repriced.csv `
      --cost-module scripts/experiments/cost_rates.py `
      --cost-samples outputs/cost_rate_samples/uhs_cost_rates.parquet

Stage 3, back in the CEM environment::
  pixi run python -m scripts.results_analysis.plot_cost_outcomes_expost plot `
      --countries NLD DEU GBR

IMPORTANT: This is an EX-POST accounting adjustment, not re-optimisation. Only
UHS annualised investment costs are re-priced. Other investments, dispatch,
fixed and variable operating costs, and system constraints are held unchanged.
The main figure compares original and ex-post corrected relative costs side by
side, anonymised across country-weather cases.
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
import xarray as xr

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
FONT_SIZE = 10
FIG_W = 3.5
ORIGINAL_COLOR = "0.55"
CORRECTED_COLOR = plt.get_cmap("plasma")(0.7)


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
    reference = cases.loc[cases["experiment"] == "B1", [
        "country", "weather", "system_cost", "storage_cost",
        "system_cost_corrected", "storage_cost_corrected", "correction_complete"]].copy()
    reference = reference.rename(columns={
        "system_cost": "b1_system", "storage_cost": "b1_storage",
        "system_cost_corrected": "b1_system_corrected", "storage_cost_corrected": "b1_storage_corrected",
        "correction_complete": "b1_complete"})
    if reference.duplicated(["country", "weather"]).any():
        raise ValueError("Duplicate B1 benchmark")
    pairs = cases.merge(reference, on=["country", "weather"], how="left", validate="many_to_one")
    valid = (pairs["correction_complete"] & pairs["b1_complete"].fillna(False) &
             (pairs["b1_system"] > 0) & (pairs["b1_storage"] > 0) &
             (pairs["b1_system_corrected"] > 0) & (pairs["b1_storage_corrected"] > 0))
    pairs["matched"] = valid
    for component in ("system", "storage"):
        pairs[f"{component}_original_relative"] = np.where(valid, pairs[f"{component}_cost"] / pairs[f"b1_{component}"], np.nan)
        pairs[f"{component}_corrected_relative"] = np.where(valid, pairs[f"{component}_cost_corrected"] / pairs[f"b1_{component}_corrected"], np.nan)
    return pairs


def _series_values(data: pd.DataFrame, experiment: str, field: str) -> np.ndarray:
    vals = pd.to_numeric(data.loc[data["experiment"] == experiment, field], errors="coerce").to_numpy(float)
    return vals[np.isfinite(vals)]


def _combined_panel(ax, data: pd.DataFrame, field_original: str, field_corrected: str, ylabel: str, *, baseline: float = 1.0) -> None:
    ax.axhline(baseline, color="0.5", ls=(0, (3, 2)), lw=.8, zorder=1)
    ax.grid(axis="y", color="0.88", lw=.55, zorder=0)
    all_vals: list[float] = []
    offset = 0.16
    median_half = 0.11
    for i, exp in enumerate(EXPERIMENT_ORDER):
        orig = _series_values(data, exp, field_original)
        corr = _series_values(data, exp, field_corrected)
        if orig.size:
            x = np.full(orig.size, i - offset)
            ax.scatter(x, orig, s=24, marker="o", facecolors="white", edgecolors=ORIGINAL_COLOR,
                       linewidths=.9, alpha=.95, zorder=3)
            ax.hlines(np.median(orig), i - offset - median_half, i - offset + median_half,
                      color=ORIGINAL_COLOR, lw=2.0, zorder=4)
            all_vals.extend(orig.tolist())
        if corr.size:
            x = np.full(corr.size, i + offset)
            ax.scatter(x, corr, s=24, marker="o", facecolors=[CORRECTED_COLOR], edgecolors="0.2",
                       linewidths=.35, alpha=.82, zorder=3)
            ax.hlines(np.median(corr), i + offset - median_half, i + offset + median_half,
                      color="black", lw=2.0, zorder=4)
            all_vals.extend(corr.tolist())
    if not all_vals:
        ax.text(.5, .5, "No complete paired cases", transform=ax.transAxes, ha="center", va="center")
    else:
        low, high = min(min(all_vals), baseline), max(max(all_vals), baseline)
        pad = max((high - low) * .12, .02 if baseline == 1 else .15)
        ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(-.5, len(EXPERIMENT_ORDER) - .5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.tick_params(direction="out", length=3)


def plot_combined_relative_costs(cases: pd.DataFrame, output: Path) -> None:
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
                         "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42}):
        fig, axs = plt.subplots(2, 1, figsize=(FIG_W, 5.15), sharex=True)
        fig.subplots_adjust(left=.28, right=.97, bottom=.11, top=.89, hspace=.2)
        _combined_panel(axs[0], cases, "system_original_relative", "system_corrected_relative", "Relative system cost", baseline=1.0)
        _combined_panel(axs[1], cases, "storage_original_relative", "storage_corrected_relative", "Relative storage cost", baseline=1.0)
        axs[1].set_xticks(range(len(EXPERIMENT_ORDER)), EXPERIMENT_ORDER)
        axs[1].set_xlabel("Representation")
        axs[0].tick_params(axis="x", bottom=False)
        handles = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=ORIGINAL_COLOR,
                   markeredgewidth=.9, markersize=5, label="Original"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=CORRECTED_COLOR, markeredgecolor="0.2",
                   markeredgewidth=.35, markersize=5, label="Ex post corrected"),
            Line2D([], [], color="black", lw=2.0, label="Median"),
        ]
        fig.legend(handles=handles, ncol=3, bbox_to_anchor=(.5, .995), loc="upper center", frameon=False, handlelength=1.4)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output)
        plt.close(fig)
        print("Saved", output)


def plot_two_panel(cases: pd.DataFrame, output: Path, f1: str, f2: str, ylabel1: str, ylabel2: str, *, baseline: float) -> None:
    with plt.rc_context({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
                         "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
                         "legend.fontsize": FONT_SIZE, "pdf.fonttype": 42}):
        fig, axs = plt.subplots(2, 1, figsize=(FIG_W, 5.1), sharex=True)
        fig.subplots_adjust(left=.27, right=.96, bottom=.11, top=.89, hspace=.2)
        _combined_panel(axs[0], cases, f1, f1, ylabel1, baseline=baseline)
        _combined_panel(axs[1], cases, f2, f2, ylabel2, baseline=baseline)
        axs[1].set_xticks(range(len(EXPERIMENT_ORDER)), EXPERIMENT_ORDER)
        axs[1].set_xlabel("Representation")
        axs[0].tick_params(axis="x", bottom=False)
        fig.savefig(output)
        plt.close(fig)
        print("Saved", output)


def plot_technology_errors(designs: pd.DataFrame, output: Path) -> None:
    valid = designs.loc[designs["included"]].copy()
    if valid.empty:
        return
    experiments = [e for e in EXPERIMENT_ORDER if e in set(valid["experiment"])]
    with plt.rc_context({"font.size": FONT_SIZE, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(len(experiments), 1, figsize=(FIG_W, max(2.2, 1.65 * len(experiments)+.75)), sharex=True, squeeze=False)
        fig.subplots_adjust(left=.25, right=.97, top=.93, bottom=.13, hspace=.26)
        for ax, exp in zip(axes[:,0], experiments):
            local = valid.loc[valid["experiment"] == exp]
            ax.axhline(0, color="0.5", ls=(0, (3, 2)), lw=.8)
            ax.grid(axis="y", color="0.88", lw=.55)
            for j, family in enumerate(UHS):
                vals = pd.to_numeric(local.loc[local["technology"] == family, "representation_error_pct"], errors="coerce").dropna().to_numpy()
                if not vals.size:
                    continue
                ax.scatter(np.full(len(vals), j), vals, s=24, c=[CORRECTED_COLOR], edgecolor="0.3", lw=.35, alpha=.78)
                ax.hlines(np.median(vals), j-.11, j+.11, lw=2, color="black")
            ax.set_xlim(-.55, len(UHS)-.45)
            ax.set_ylabel(f"{exp}: error (%)", fontsize=FONT_SIZE)
            ax.spines[["right", "top"]].set_visible(False)
        axes[-1,0].set_xticks(range(len(UHS)), [LABEL[t] for t in UHS])
        handles = [
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=CORRECTED_COLOR, markeredgecolor="0.2", markersize=5, label="Cases"),
            Line2D([], [], color="black", linewidth=2, label="Median"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5,.995), ncol=2, frameon=False, handlelength=1.3, fontsize=FONT_SIZE)
        fig.savefig(output)
        plt.close(fig)
        print("Saved", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "plot"], help="Export CEM designs or plot re-priced comparison")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENT_ORDER))
    parser.add_argument("--with-diagnostics", action="store_true",
                        help="Also export the earlier diagnostic figures (correction impact and technology-level errors).")
    args = parser.parse_args()
    countries = {x.upper() for x in args.countries} if args.countries else None
    experiments = {x.upper() for x in args.experiments}
    out = args.output_dir
    if args.mode == "export":
        export_cases(args.input_dir, out, countries, experiments)
        return
    cases_file, designs_file, reprice_file = (out / n for n in ("expost_case_costs_original.csv", "expost_uhs_designs.csv", "expost_uhs_repriced.csv"))
    for path in (cases_file, designs_file):
        if not path.is_file():
            parser.error(f"Missing {path}; run export first")
    cases, designs = (pd.read_csv(p) for p in (cases_file, designs_file))
    if reprice_file.is_file():
        repriced = pd.read_csv(reprice_file)
    elif designs.empty:
        repriced = pd.DataFrame(columns=["case", "tech_id", "status", "adjustment_factor", "detailed_tcr_eur", "represented_tcr_eur"])
    else:
        parser.error(f"Missing {reprice_file}; run the cost-model repricing step first")
    cases = cases.loc[cases["experiment"].isin(experiments) & ((cases["country"].isin(countries)) if countries else True)].copy()
    designs = designs.loc[designs["case"].isin(cases["case"])].copy()
    repriced = repriced.loc[repriced["case"].isin(cases["case"])].copy()
    corrected, detailed = attach_corrections(cases, designs, repriced)
    combined = compare_b1(corrected)
    out.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out / "expost_uhs_repricing_audit.csv", index=False)
    combined.to_csv(out / "expost_cost_comparison.csv", index=False)
    corrected.to_csv(out / "expost_case_costs_corrected.csv", index=False)
    plot_combined_relative_costs(combined, out / "expost_relative_costs_combined.pdf")
    if args.with_diagnostics:
        plot_two_panel(combined, out / "expost_cost_correction_impact.pdf",
                       "system_effect_pct", "storage_effect_pct", "System cost change (%)", "Storage cost change (%)", baseline=0.)
        plot_technology_errors(detailed, out / "expost_uhs_representation_error.pdf")
    unresolved = combined.loc[~combined["correction_complete"], "case"].tolist()
    print(f"Compared {combined['matched'].sum()} matched, corrected country-weather cases")
    if unresolved:
        warnings.warn(f"Excluded {len(unresolved)} incomplete cases: {unresolved[:20]}", stacklevel=1)


if __name__ == "__main__":
    main()
