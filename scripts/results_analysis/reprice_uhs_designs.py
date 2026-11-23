"""Ex-post reprice CEM UHS designs using NetCDF costs, never cost-rate tables.

Input is the CSV produced by plot_cost_outcomes_expost.py export. It contains
the *actual* unannualised investment cost and fixed operating cost saved in
each Calliope .nc file. No current or historical Parquet is consulted.

Detailed TCR and fixed OPEX are evaluated at the experiment's *recorded*
reference energy, the CEM's *realised* EPR/WTIR, and then scaled linearly
to installed energy. Thus this tests capacity-ratio representation rather than
facility economies of scale. Variable OPEX remains unchanged.

For reproducibility the source code must match the historical model fingerprint
in --reference-manifest. To reprice an archived scenario, use the corresponding
archived UHS model revision, not whichever reference site is currently active.

Example (from the UHS detailed-model repository root):

  pixi run python ../uhs_cem_representation/scripts/results_analysis/reprice_uhs_designs.py \
    --input ../uhs_cem_representation/results/costs_expost/expost_uhs_designs.csv \
    --output ../uhs_cem_representation/results/costs_expost/expost_uhs_repriced.csv \
    --cost-module scripts/experiments/cost_rates.py \
    --reference-manifest outputs/cost_rate_samples/WTIR3/uhs_cost_rates.reference.json \
    --cem-cost-unit-eur 1000000
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd


FAMILIES = {"salt_cavern", "depleted_gas_field", "aquifer", "lined_rock_cavern"}
REQUIRED = {
    "case", "country", "weather", "experiment", "tech_id", "technology",
    "storage_MWh", "injection_MW", "withdrawal_MW", "design_status",
    "cem_investment_unannualised", "cem_investment_annualised",
    "cost_operation_fixed", "period_weight_years", "EPR_days", "WTIR",
    "input_reference_epr_days", "input_reference_wtir",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Detailed cost evaluator not found: {path}")
    root = path.parents[2]
    for folder in (root, root / "src"):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("uhs_expost_cost_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import detailed evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for name in ("get_technology", "evaluate_design", "CAPITAL_TYPES", "CostDriver", "CostType"):
        if not hasattr(module, name):
            raise AttributeError(f"Detailed evaluator lacks {name}")
    return module, root


def _positive(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label}: expected finite positive value, got {value!r}")
    return number


def evaluate_detail(module, technology: str, energy_gwh: float, epr_days: float, wtir: float):
    breakdown = module.evaluate_design(
        module.get_technology(technology), energy_gwh, epr_days, wtir
    )
    tcr, fom = 0.0, 0.0
    vom = {"injection": 0.0, "withdrawal": 0.0}
    for component in breakdown.components:
        value = float(component.value_eur)
        if not math.isfinite(value) or value < -1e-7:
            raise ValueError(f"Non-finite/negative detailed component {component.name}")
        if component.cost_type in module.CAPITAL_TYPES:
            tcr += value
        elif component.cost_type == module.CostType.FIXED_OPEX:
            fom += value
        elif component.cost_type == module.CostType.VARIABLE_OPEX:
            if component.cost_driver == module.CostDriver.INJECTION_THROUGHPUT_KWH:
                key = "injection"
            elif component.cost_driver == module.CostDriver.WITHDRAWAL_THROUGHPUT_KWH:
                key = "withdrawal"
            else:
                raise ValueError(f"Unexpected VOM driver for {component.name}")
            vom[key] += value / _positive(component.driver_value, component.name)
        else:
            raise ValueError(f"Unexpected detailed cost category for {component.name}")
    if tcr <= 0 or fom <= 0:
        raise ValueError("Detailed model returned non-positive TCR or fixed OPEX")
    return tcr, fom, vom


def _near(actual: float, target: float, label: str, *, rtol: float = 1e-4) -> None:
    if not math.isclose(actual, target, rel_tol=rtol, abs_tol=1e-7):
        raise ValueError(f"{label}: realised {actual:.8g}, expected {target:.8g}")


def reprice_designs(designs: pd.DataFrame, module, references: dict,
                    *, currency_unit_eur: float, model_hash: str,
                    reference_manifest_hash: str) -> pd.DataFrame:
    missing = REQUIRED.difference(designs.columns)
    if missing:
        raise ValueError(f"NetCDF design export lacks {sorted(missing)}; rerun export")
    if designs.duplicated(["case", "tech_id"]).any():
        raise ValueError("Duplicated CEM case/technology IDs")
    currency_unit_eur = _positive(currency_unit_eur, "cem-cost-unit-eur")
    out = []
    for _, record in designs.iterrows():
        row = record.to_dict()
        result = {key: row[key] for key in ("case", "country", "weather", "experiment", "tech_id", "technology")}
        result.update(status="error", error="", cost_model_sha256=model_hash,
                      reference_manifest_sha256=reference_manifest_hash,
                      representation_source="netcdf_cost_investment",
                      benchmark_scope="EPR_WTIR_only_reference_energy_scaled_TCR_and_FOM")
        try:
            technology = str(row["technology"])
            if technology not in FAMILIES or technology not in references:
                raise ValueError(f"No reference configuration for {technology}")
            if row["design_status"] != "ok":
                raise ValueError(f"Incomplete installed design: {row['design_status']}")
            ref = references[technology]
            energy_gwh = _positive(row["storage_MWh"], "storage_MWh") / 1000.0
            pin = _positive(row["injection_MW"], "injection_MW")
            pout = _positive(row["withdrawal_MW"], "withdrawal_MW")
            epr = energy_gwh * 1000.0 / pout / 24.0
            wtir = pout / pin
            _near(epr, float(row["EPR_days"]), "Exported EPR disagrees with capacities")
            _near(wtir, float(row["WTIR"]), "Exported WTIR disagrees with capacities")

            reference_energy = _positive(ref["reference_energy_gwh"], "reference energy")
            reference_epr = _positive(ref["reference_epr_days"], "reference EPR")
            reference_wtir = _positive(ref["reference_wtir"], "reference WTIR")
            if str(row["experiment"]).upper() == "B1":
                _near(epr, reference_epr, "Bundled EPR")
                _near(wtir, reference_wtir, "Bundled WTIR")
                _near(float(row["input_reference_epr_days"]), reference_epr, "NetCDF bundled EPR input")
                _near(float(row["input_reference_wtir"]), reference_wtir, "NetCDF bundled WTIR input")
            if str(row["experiment"]).upper().startswith(("F", "FX")):
                epr_bounds = ref["epr_domain_days"]
                wtir_bounds = ref["wtir_domain"]
                if not (epr_bounds[0] - 1e-7 <= epr <= epr_bounds[1] + 1e-7
                        and wtir_bounds[0] - 1e-7 <= wtir <= wtir_bounds[1] + 1e-7):
                    raise ValueError(f"Design (EPR={epr:g}, WTIR={wtir:g}) outside fitted domain; do not extrapolate")

            represented_tcr = _positive(row["cem_investment_unannualised"], "NetCDF initial investment")
            investment_period = _positive(row["cem_investment_annualised"], "NetCDF annualised investment")
            fixed_period = _positive(row["cost_operation_fixed"], "NetCDF fixed OPEX")
            period_weight = _positive(row["period_weight_years"], "model period weight")
            raw_detail_tcr, raw_detail_fom, vom = evaluate_detail(
                module, technology, reference_energy, epr, wtir
            )
            energy_scale = energy_gwh / reference_energy
            detailed_tcr = raw_detail_tcr * energy_scale / currency_unit_eur
            detailed_fom_period = raw_detail_fom * energy_scale * period_weight / currency_unit_eur
            ratio = detailed_tcr / represented_tcr
            fom_ratio = detailed_fom_period / fixed_period
            if not (math.isfinite(ratio) and ratio > 0 and math.isfinite(fom_ratio) and fom_ratio > 0):
                raise ValueError("Invalid detailed/model cost ratios")

            if str(row["experiment"]).upper() == "B1":
                _near(ratio, 1.0, "Bundled TCR calibration", rtol=5e-4)
                _near(fom_ratio, 1.0, "Bundled fixed-OPEX calibration", rtol=5e-4)

            result.update(
                status="ok", error="", reference_site=ref["reference_site"],
                design_epr_days=epr, design_wtir=wtir,
                reference_energy_gwh=reference_energy,
                sample_reference_epr_days=reference_epr,
                sample_reference_wtir=reference_wtir,
                design_ratios_match_sample=(math.isclose(epr, reference_epr, rel_tol=1e-4)
                                             and math.isclose(wtir, reference_wtir, rel_tol=1e-4)),
                actual_to_reference_energy_ratio=energy_scale,
                period_weight_years=period_weight,
                cem_cost_unit_eur=currency_unit_eur,
                represented_tcr_eur=represented_tcr,
                detailed_tcr_eur=detailed_tcr,
                adjustment_factor=ratio,
                representation_error_pct=100.0 * (represented_tcr - detailed_tcr) / detailed_tcr,
                cem_investment_annualised=investment_period,
                corrected_investment_annualised=investment_period * ratio,
                correction_investment_annualised=investment_period * (ratio - 1.0),
                # Legacy names retained for compatibility with the companion plotter.
                # Both fields are CEM model-period costs, not 1-year totals.
                cem_fixed_opex_annualised=fixed_period,
                represented_fom_eur_per_year=fixed_period / period_weight,
                detailed_fom_eur_per_year=detailed_fom_period / period_weight,
                fixed_opex_adjustment_factor=fom_ratio,
                fixed_opex_representation_error_pct=(
                    100.0 * (fixed_period - detailed_fom_period) / detailed_fom_period
                ),
                corrected_fixed_opex_annualised=detailed_fom_period,
                correction_fixed_opex_annualised=detailed_fom_period - fixed_period,
                correction_annualised=(investment_period * (ratio - 1.0)
                                       + detailed_fom_period - fixed_period),
                detailed_injection_vom_rate_eur_per_kwh=vom["injection"],
                detailed_withdrawal_vom_rate_eur_per_kwh=vom["withdrawal"],
                implied_annualisation_factor=investment_period / represented_tcr,
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        out.append(result)
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cost-module", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--cem-cost-unit-eur", required=True, type=float,
                        help="Euros per monetary cost unit in saved NetCDF (e.g. 1000000 for MEUR)")
    args = parser.parse_args()
    model, root = load_model(args.cost_module)
    from scripts.experiments.reference_provenance import validate_reference_manifest
    manifest = json.loads(args.reference_manifest.read_text(encoding="utf8"))
    references = validate_reference_manifest(root, manifest, model)
    rows = pd.read_csv(args.input)
    repriced = reprice_designs(
        rows, model, references, currency_unit_eur=args.cem_cost_unit_eur,
        model_hash=sha256_file(args.cost_module),
        reference_manifest_hash=sha256_file(args.reference_manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repriced.to_csv(args.output, index=False)
    failures = repriced.loc[repriced["status"] != "ok"]
    print(f"Saved {args.output}: {len(repriced) - len(failures)}/{len(repriced)} valid UHS installations")
    if not failures.empty:
        for _, row in failures.iterrows():
            print(f"ERROR {row['case']}/{row['tech_id']}: {row['error']}")
        raise SystemExit(2)
    if not repriced.empty:
        print(f"TCR representation error median: {repriced['representation_error_pct'].median():+.3f}%")
        print(f"Fixed-OPEX representation error median: {repriced['fixed_opex_representation_error_pct'].median():+.3f}%")
        print("Variable OPEX unchanged; directional throughput repricing is not yet implemented")


if __name__ == "__main__":
    main()
