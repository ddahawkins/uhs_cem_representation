"""Compare CEM experiment TCR to an EPR/WTIR-only detailed cost benchmark.

The CEM rates are taken from the HISTORICAL cost-rate parquet used for the CEM.
The detailed benchmark is calculated at THAT RATE TABLE'S reference energy
capacity, with the realised EPR and WTIR, then scaled to CEM-installed energy.
This isolates error due to the design ratios, rather than confounding it with
nonlinear effects of changing absolute project size. A bundled B1 solution that
retains its calibration ratios must therefore have negligible error, provided
its cost-rate table and the detailed model use consistent assumptions.

Option --decompose also evaluates the detailed model at actual absolute size,
and at actual size but sample-centre EPR and WTIR, for optional scale diagnostics.
Those scale-specific diagnostics NEVER enter the main corrected-cost figures.

Run in the UHS model Pixi environment, e.g.:
 pixi run python ../uhs_cem_representation/scripts/results_analysis/reprice_uhs_designs.py \
   --cost-module scripts/experiments/cost_rates.py \
   --cost-samples outputs/cost_rate_samples/uhs_cost_rates.parquet \
   --input ../uhs_cem_representation/results/figures/expost_uhs_designs.csv \
   --output ../uhs_cem_representation/results/figures/expost_uhs_repriced.csv
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import re
import sys
from pathlib import Path

import pandas as pd

FAMILIES = {"salt_cavern", "depleted_gas_field", "aquifer", "lined_rock_cavern"}
INPUT_COLUMNS = {"case", "country", "weather", "experiment", "tech_id", "technology",
                 "storage_MWh", "injection_MW", "withdrawal_MW", "design_status",
                 "cem_investment_annualised"}
SAMPLE_COLUMNS = {"experiment_id", "technology", "instance_id",
                  "E_reference_GWh", "EPR_days", "WTIR",
                  "TCR_storage_EUR_per_kWh", "TCR_injection_EUR_per_kW",
                  "TCR_withdrawal_EUR_per_kW"}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cost_model(path: Path):
    """Use the exact detailed evaluator used by generate_costs_parquet.py."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No cost model module at {path}")
    root = path.parents[2]   # repo/scripts/experiments/cost_rates.py -> repo
    for p in (root, root / "src"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("expost_current_cost_rates", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for attr in ("get_technology", "evaluate_design", "CAPITAL_TYPES", "CostDriver"):
        if not hasattr(module, attr):
            raise AttributeError(f"{path.name} lacks {attr}, expected current cost_rates.py")
    return module


def read_cost_samples(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing CEM cost-rate table {path}")
    sample = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    missing = SAMPLE_COLUMNS - set(sample.columns)
    if missing:
        raise ValueError(f"Cost-rate table lacks {sorted(missing)}")
    sample = sample.copy()
    sample["experiment_id"] = sample["experiment_id"].astype(str).str.upper()
    sample["technology"] = sample["technology"].astype(str).str.lower()
    if sample.duplicated(["experiment_id", "technology", "instance_id"]).any():
        raise ValueError("Duplicate experiment, technology and instance_id in cost table")
    for field in ("TCR_storage_EUR_per_kWh", "TCR_injection_EUR_per_kW", "TCR_withdrawal_EUR_per_kW"):
        sample[field] = pd.to_numeric(sample[field], errors="raise")
        if not sample[field].map(lambda x: math.isfinite(x) and x >= -1e-9).all():
            raise ValueError(f"Invalid TCR rate in {field}")
    for field in ("E_reference_GWh", "EPR_days", "WTIR"):
        sample[field] = pd.to_numeric(sample[field], errors="raise")
        if not (sample[field] > 0).all():
            raise ValueError(f"Non-positive reference field {field}")
    return sample


def read_variant_map(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["case", "tech_id", "instance_id"])
    frame = pd.read_csv(path, dtype=str)
    if not {"tech_id", "instance_id"}.issubset(frame.columns):
        raise ValueError("Variant map needs tech_id,instance_id (case optional)")
    if "case" not in frame:
        frame["case"] = ""
    frame = frame.fillna("")
    if frame.duplicated(["case", "tech_id"]).any():
        raise ValueError("Duplicate case/tech_id in variant map")
    return frame


def _variant_number(name: str, token: str) -> int | None:
    match = re.search(rf"{token}[_-]?0*(\d+)(?:\D|$)", name.lower())
    return int(match.group(1)) if match else None


def choose_sample(row: dict, samples: pd.DataFrame, variant_map: pd.DataFrame) -> pd.Series:
    matching = samples.loc[
        (samples["experiment_id"] == str(row["experiment"]).upper()) &
        (samples["technology"] == str(row["technology"]).lower())]
    if matching.empty:
        raise LookupError(f"No sample for {row['experiment']} / {row['technology']}")
    # Explicit identity always takes precedence and supports arbitrary FXn IDs.
    overrides = variant_map.loc[(variant_map["tech_id"] == str(row["tech_id"])) &
                                (variant_map["case"].isin(["", str(row["case"])]))]
    if len(overrides) > 1:
        overrides = overrides.loc[overrides["case"] == str(row["case"])]
    if len(overrides) > 1:
        raise ValueError(f"Ambiguous variant map for {row['case']}/{row['tech_id']}")
    if len(overrides) == 1:
        ident = str(overrides.iloc[0]["instance_id"])
        matched = matching.loc[matching["instance_id"].astype(str) == ident]
        if len(matched) != 1:
            raise ValueError(f"Mapped instance_id {ident} absent or ambiguous for {row['tech_id']}")
        return matched.iloc[0]
    if len(matching) == 1:
        return matching.iloc[0]
    tech = str(row["tech_id"]).lower()
    exact = matching.loc[matching["instance_id"].astype(str).str.lower() == tech]
    if len(exact) == 1:
        return exact.iloc[0]
    epr = _variant_number(tech, "epr")
    wtir = _variant_number(tech, "wtir")
    if epr is not None or wtir is not None:
        chosen = matching.copy()
        for token, val in (("epr", epr), ("wtir", wtir)):
            if val is not None:
                chosen = chosen.loc[chosen["instance_id"].astype(str).map(
                    lambda s: _variant_number(s, token) == val)]
        if len(chosen) == 1:
            return chosen.iloc[0]
    raise ValueError(f"Ambiguous {row['experiment']}/{row['technology']} variant {row['tech_id']}: "
                     f"{len(matching)} possible samples. Supply --variant-map.")


def _positive(value, label: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{label} must be positive, found {value!r}")
    return v


def calculate_detail(module, family: str, e_gwh: float, epr_days: float, wtir: float):
    spec = module.get_technology(family)
    result = module.evaluate_design(spec, e_gwh, epr_days, wtir)
    breakdown = {"injection": 0.0, "storage": 0.0, "withdrawal": 0.0}
    by_driver = {module.CostDriver.INJECTION: "injection",
                 module.CostDriver.STORAGE: "storage",
                 module.CostDriver.WITHDRAWAL: "withdrawal"}
    for component in result.components:
        if component.cost_type not in module.CAPITAL_TYPES:
            continue
        if component.cost_driver not in by_driver:
            raise ValueError(f"Unexpected capital cost driver {component.cost_driver}")
        value = float(component.value_eur)
        if not math.isfinite(value) or value < -1e-7:
            raise ValueError(f"Invalid CAPEX/ABEX cost component {component.name}: {value}")
        breakdown[by_driver[component.cost_driver]] += value
    total = sum(breakdown.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"Non-positive detailed TCR {total}")
    return breakdown, total


def reprice_rows(designs: pd.DataFrame, samples: pd.DataFrame, module, variant_map: pd.DataFrame,
                 *, cost_model_hash: str = "", cost_samples_hash: str = "",
                 decompose: bool = False) -> pd.DataFrame:
    if INPUT_COLUMNS - set(designs.columns):
        raise ValueError(f"Design input missing {sorted(INPUT_COLUMNS - set(designs.columns))}")
    if designs.duplicated(["case", "tech_id"]).any():
        raise ValueError("Duplicate case/tech_id in exported designs")
    results = []
    for _, rec in designs.iterrows():
        r = rec.to_dict()
        output = {k: r.get(k) for k in ("case", "country", "weather", "experiment", "tech_id", "technology")}
        output.update(status="error", error="", cost_model_sha256=cost_model_hash,
                      cost_samples_sha256=cost_samples_hash, representation_source="exported_cost_rate_table")
        try:
            if r["technology"] not in FAMILIES:
                raise ValueError(f"Not geological UHS: {r['technology']}")
            if r["design_status"] != "ok":
                raise ValueError(f"Incomplete installed capacities: {r['design_status']}")
            storage_mwh = _positive(r["storage_MWh"], "storage_MWh")
            injection_mw = _positive(r["injection_MW"], "injection_MW")
            withdrawal_mw = _positive(r["withdrawal_MW"], "withdrawal_MW")
            annualised = _positive(r["cem_investment_annualised"], "CEM annualised investment")
            # GWh and MW are all in H2 LHV. TCR coefficients are EUR/kWh, EUR/kW.
            storage_kwh = storage_mwh * 1000.0
            injection_kw = injection_mw * 1000.0
            withdrawal_kw = withdrawal_mw * 1000.0
            epr_days = storage_mwh / withdrawal_mw / 24.0
            wtir = withdrawal_mw / injection_mw
            sample = choose_sample(r, samples, variant_map)
            rates = {"storage": float(sample["TCR_storage_EUR_per_kWh"]),
                     "injection": float(sample["TCR_injection_EUR_per_kW"]),
                     "withdrawal": float(sample["TCR_withdrawal_EUR_per_kW"])}
            represented_components = {
                "storage": rates["storage"] * storage_kwh,
                "injection": rates["injection"] * injection_kw,
                "withdrawal": rates["withdrawal"] * withdrawal_kw,
            }
            represented_tcr = sum(represented_components.values())
            if not math.isfinite(represented_tcr) or represented_tcr <= 0:
                raise ValueError(f"Invalid represented TCR: {represented_tcr}")
            # Important: keep reference E fixed, to isolate EPR/WTIR error.
            # Direct project re-sizing at actual E would additionally test scale
            # nonlinearity and make B1 nonzero despite matched design ratios.
            e_ref_gwh = _positive(sample["E_reference_GWh"], "E_reference_GWh")
            epr_ref_days = _positive(sample["EPR_days"], "sample EPR_days")
            wtir_ref = _positive(sample["WTIR"], "sample WTIR")
            reference_capacities = {
                "storage": e_ref_gwh * 1e6,
                "withdrawal": e_ref_gwh * 1e6 / (24.0 * epr_ref_days),
                "injection": e_ref_gwh * 1e6 / (24.0 * epr_ref_days * wtir_ref),
            }
            represented_ref_tcr = sum(
                rates[group] * reference_capacities[group]
                for group in ("storage", "injection", "withdrawal")
            )
            _, detailed_ref_tcr = calculate_detail(
                module, r["technology"], e_ref_gwh, epr_ref_days, wtir_ref
            )
            reference_error_pct = 100.0 * (
                represented_ref_tcr - detailed_ref_tcr
            ) / detailed_ref_tcr
            ratio_matches_sample = (math.isclose(epr_days, epr_ref_days, rel_tol=1e-5, abs_tol=1e-6)
                                    and math.isclose(wtir, wtir_ref, rel_tol=1e-5, abs_tol=1e-6))
            if (str(r["experiment"]).upper() == "B1" and abs(reference_error_pct) > 0.05):
                raise ValueError(
                    f"B1 reference cost reconciliation failed: {reference_error_pct:+.4f}% at "
                    f"E_ref={e_ref_gwh:.4f} GWh, EPR_ref={epr_ref_days:.4f} d, "
                    f"WTIR_ref={wtir_ref:.4f}; verify historical cost table and cost-model version"
                )
            detail_ref_shape_components, detail_ref_shape_total = calculate_detail(
                module, r["technology"], e_ref_gwh, epr_days, wtir
            )
            scale = (storage_mwh / 1000.0) / e_ref_gwh
            detail_components = {k: v * scale for k, v in detail_ref_shape_components.items()}
            detailed_tcr = detail_ref_shape_total * scale
            ratio = detailed_tcr / represented_tcr
            implied_factor = annualised / represented_tcr
            if not (math.isfinite(ratio) and ratio > 0 and math.isfinite(implied_factor) and implied_factor > 0):
                raise ValueError("Invalid ratio or implied annualisation factor")
            output.update(
                status="ok", error="", cost_sample_instance_id=str(sample["instance_id"]),
                design_epr_days=epr_days, design_wtir=wtir,
                reference_energy_gwh=e_ref_gwh,
                sample_reference_epr_days=epr_ref_days,
                sample_reference_wtir=wtir_ref,
                actual_to_reference_energy_ratio=scale,
                design_ratios_match_sample=ratio_matches_sample,
                represented_tcr_at_reference_eur=represented_ref_tcr,
                detailed_tcr_at_reference_eur=detailed_ref_tcr,
                reference_centre_error_pct=reference_error_pct,
                detailed_tcr_ratio_only_at_reference_energy_eur=detail_ref_shape_total,
                benchmark_scope="EPR_WTIR_only_reference_energy_scaled",
                represented_tcr_eur=represented_tcr, linear_tcr_eur=represented_tcr,
                detailed_tcr_eur=detailed_tcr, adjustment_factor=ratio,
                detailed_to_cem_ratio=ratio,
                representation_error_pct=100.0 * (represented_tcr - detailed_tcr) / detailed_tcr,
                implied_annualisation_factor=implied_factor,
                cem_investment_annualised=annualised,
                corrected_investment_annualised=annualised * ratio,
                correction_annualised=annualised * (ratio - 1.0),
            )
            if "EPR_days_lower" in sample and "EPR_days_upper" in sample:
                lo, hi = float(sample["EPR_days_lower"]), float(sample["EPR_days_upper"])
                output["inside_sample_epr_range"] = lo - 1e-8 <= epr_days <= hi + 1e-8
            if "WTIR_lower" in sample and "WTIR_upper" in sample:
                lo, hi = float(sample["WTIR_lower"]), float(sample["WTIR_upper"])
                output["inside_sample_wtir_range"] = lo - 1e-8 <= wtir <= hi + 1e-8
            for group in ("storage", "injection", "withdrawal"):
                output[f"cem_{group}_rate"] = rates[group]
                output[f"represented_{group}_tcr_eur"] = represented_components[group]
                output[f"detailed_{group}_tcr_eur"] = detail_components[group]
            if decompose:
                # These optional diagnostic columns do not change the
                # benchmark used for 'adjustment_factor' and plotted errors.
                _, full_actual_total = calculate_detail(
                    module, r["technology"], storage_mwh / 1000.0, epr_days, wtir
                )
                _, size_only_total = calculate_detail(
                    module, r["technology"], storage_mwh / 1000.0,
                    epr_ref_days, wtir_ref
                )
                output["detailed_tcr_full_actual_design_eur"] = full_actual_total
                output["detailed_tcr_size_only_at_sample_ratios_eur"] = size_only_total
                output["full_size_vs_ratio_only_pct"] = (
                    100.0 * (full_actual_total / detailed_tcr - 1.0)
                )
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
        results.append(output)
    return pd.DataFrame(results)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cost-module", type=Path, default=Path("scripts/experiments/cost_rates.py"))
    ap.add_argument("--cost-samples", type=Path, default=Path("outputs/cost_rate_samples/uhs_cost_rates.parquet"),
                    help="Exact experiment-specific TCR coefficients used in CEM (CSV/Parquet).")
    ap.add_argument("--variant-map", type=Path, help="Optional tech_id,instance_id[,case] CSV for FX2/FX4 variants.")
    ap.add_argument("--decompose", action="store_true")
    args = ap.parse_args()
    samples = read_cost_samples(args.cost_samples)
    model = load_cost_model(args.cost_module)
    frames = pd.read_csv(args.input)
    result = reprice_rows(frames, samples, model, read_variant_map(args.variant_map),
                          cost_model_hash=hash_file(args.cost_module),
                          cost_samples_hash=hash_file(args.cost_samples), decompose=args.decompose)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    ok = result.loc[result["status"] == "ok"] if not result.empty else result
    print(f"Saved {args.output}: {len(ok)}/{len(result)} UHS designs repriced")
    if len(ok):
        print(f"  Representation error (CEM - detail) / detail: median {ok['representation_error_pct'].median():+.2f}%")
        print("  Ratio-only correction: detailed cost computed at E_ref and scaled to installed E.")
        print("  Cost-rate table must be the one used by the historical CEM runs.")
        b1 = ok.loc[ok["experiment"].astype(str).str.upper() == "B1"]
        if len(b1):
            print(f"  B1 sanity check: max |representation error| = "
                  f"{b1['representation_error_pct'].abs().max():.6g}%; "
                  f"{int(b1['design_ratios_match_sample'].sum())}/{len(b1)} ratio-matched")
        if {"inside_sample_epr_range", "inside_sample_wtir_range"}.issubset(ok.columns):
            outside = ok.loc[(ok["inside_sample_epr_range"] == False) | (ok["inside_sample_wtir_range"] == False)]
            if len(outside):
                print(f"  NOTE: {len(outside)} design(s) outside their source sample EPR/WTIR intervals.")
        factors = ok["implied_annualisation_factor"].astype(float)
        if len(factors) > 1 and factors.max() / factors.min() > 1.05:
            print("  NOTE: implied annualisation factors vary >5%; verify lifetimes, cost basis and table provenance.")
    if len(result) != len(ok):
        print("Failures (never treated as zero correction):")
        for _, row in result.loc[result["status"] != "ok"].iterrows():
            print(f"  {row['case']} / {row['tech_id']}: {row['error']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
