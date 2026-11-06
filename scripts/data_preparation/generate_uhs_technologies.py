"""Generate Calliope 0.7 UHS technology definitions from cost-rate samples.

Example (assuming Calliope models hydrogen in MW/MWh):

    pixi run python -m scripts.data_preparation.generate_uhs_technologies `
        --csv 'resources/uhs_techno_economics/uhs_cost_rates.csv' `
        --output 'config/technologies_uhs.yaml' `
        --model-energy-unit MWh `
        --hydrogen-carrier hydrogen `
        --epr-upper-factor 2 `
        --wtir-upper-factor 2

The resulting file is an importable Calliope YAML fragment with ``templates``
and ``techs``; scenario activation is handled separately at ``nodes.*.techs``.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

EXPERIMENTS = {
    # CSV experiment ID: (scenario prefix, expected method, n_epr, n_wtir)
    "B1": ("B1", "bundled", 1, 1),
    "B25": ("B25", "bundled", 5, 5),
    "P1": ("P1", "partially_decoupled", 1, 1),
    "F1": ("F1", "fully_decoupled", 1, 1),
    "FX1": ("FX1", "fully_decoupled_cross_driver", 1, 1),
    "FX2": ("FX2", "fully_decoupled_cross_driver", 2, 1),
    "FX4": ("FX4", "fully_decoupled_cross_driver", 2, 2),
}

TECHNOLOGIES = (
    "salt_cavern",
    "aquifer",
    "depleted_gas_field",
    "lined_rock_cavern",
)

SAMPLE_SUFFIX = re.compile(r"epr(?P<epr>\d{2})_wtir(?P<wtir>\d{2})$")

# CSV field -> Calliope cost parameter, all denominators are per kWh or kW.
COST_FIELDS = {
    "TCR_storage_EUR_per_kWh": "cost_storage_cap",
    "TCR_injection_EUR_per_kW": "cost_flow_in_cap",
    "TCR_withdrawal_EUR_per_kW": "cost_flow_out_cap",
    "FOM_storage_EUR_per_kWh_year": "cost_om_annual_storage_cap",
    "FOM_injection_EUR_per_kW_year": "cost_om_annual_flow_in_cap",
    "FOM_withdrawal_EUR_per_kW_year": "cost_om_annual_flow_out_cap",
    "VOM_injection_EUR_per_kWh": "cost_flow_in",
    "VOM_withdrawal_EUR_per_kWh": "cost_flow_out",
}

SOC_24H_FIELDS = (
    "soc_24h_a_min",
    "soc_24h_b_min",
    "soc_24h_a_max",
    "soc_24h_b_max",
)

VOM_FIELDS = {
    "VOM_injection_EUR_per_kWh",
    "VOM_withdrawal_EUR_per_kWh",
}

# Applied only to variable operating costs (VOM), not CAPEX or fixed O&M.
# In a MW/MWh model this is 1e-7 million EUR per MWh.
DEFAULT_MIN_VOM_EUR_PER_MWH = Decimal("0.10")
# Extend only the outermost optimisation interval for each sampled dimension.
# Intermediate sample-cell boundaries remain unchanged; costs are still taken
# from the original CSV sample, so the extended range is an extrapolation.
DEFAULT_OUTER_UPPER_FACTOR = Decimal("2")
KWH_PER_MWH = Decimal(1000)

KWH_PER_MODEL_ENERGY_UNIT = {
    "kWh": Decimal(1),
    "MWh": Decimal(1000),
    "GWh": Decimal(1_000_000),
}

MEUR_PER_EUR = Decimal("0.000001")


def require_decimal(row: dict[str, str], field: str) -> Decimal:
    raw = row.get(field, "").strip()
    if not raw:
        raise ValueError(f"Missing required {field} in {row.get('instance_id')}")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {field}={raw!r} in {row.get('instance_id')}") from exc
    if not value.is_finite():
        raise ValueError(f"Non-finite {field} in {row.get('instance_id')}")
    return value


def optional_decimal(row: dict[str, str], field: str) -> Decimal | None:
    return require_decimal(row, field) if row.get(field, "").strip() else None


def yaml_number(value: Decimal) -> int | float:
    """Avoid noisy binary-float strings while retaining CSV numeric precision."""
    if value == value.to_integral_value():
        return int(value)
    return float(value)


def bounds(row: dict[str, str], low: str, high: str, multiplier: Decimal) -> dict:
    result = {}
    for field, name in ((low, "min"), (high, "max")):
        value = optional_decimal(row, field)
        if value is not None:
            if value <= 0:
                raise ValueError(f"{field} must be positive in {row['instance_id']}")
            result[name] = yaml_number(value * multiplier)
    if "min" in result and "max" in result and result["min"] > result["max"]:
        raise ValueError(f"Reversed bounds {low}/{high} in {row['instance_id']}")
    return result


def template_definitions(carrier: str, lifetime_years: int) -> dict:
    common = {
        "base_tech": "storage",
        "carrier_in": carrier,
        "carrier_out": carrier,
        "lifetime": lifetime_years,
        "cyclic_storage": True,
        # A default only; each UHS sample below supplies its own equivalent
        # injection efficiency if the source CSV provides one.
        "flow_in_eff": 1.0,
    }
    for parameter in COST_FIELDS.values():
        common[parameter] = {"data": None, "index": "monetary", "dims": "costs"}
    return {
        "uhs_common": common,
        "uhs_bundled": {"template": "uhs_common", "storage_formulation": "bundled"},
        "uhs_partially_decoupled": {
            "template": "uhs_common",
            "storage_formulation": "partially_decoupled",
        },
        "uhs_fully_decoupled": {
            "template": "uhs_common",
            "storage_formulation": "fully_decoupled",
        },
    }


def expected_names() -> dict[str, set[str]]:
    expected = {}
    for csv_experiment, (scenario, _, n_epr, n_wtir) in EXPERIMENTS.items():
        expected[csv_experiment] = {
            f"{scenario.lower()}_{technology}_epr{epr:02d}_wtir{wtir:02d}"
            for technology in TECHNOLOGIES
            for epr in range(1, n_epr + 1)
            for wtir in range(1, n_wtir + 1)
        }
    return expected


def sample_definition(
    row: dict[str, str], scale: Decimal, min_vom_eur_per_mwh: Decimal,
    *,
    epr_upper_factor: Decimal = DEFAULT_OUTER_UPPER_FACTOR,
    wtir_upper_factor: Decimal = DEFAULT_OUTER_UPPER_FACTOR,
) -> tuple[str, dict]:
    csv_experiment = row["experiment_id"]
    if csv_experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment {csv_experiment!r}")
    scenario, expected_method, n_epr, n_wtir = EXPERIMENTS[csv_experiment]
    method = row["method"]
    if method != expected_method:
        raise ValueError(f"Unexpected method {method!r} in {row['instance_id']}")
    technology = row["technology"]
    if technology not in TECHNOLOGIES:
        raise ValueError(f"Unsupported UHS technology {technology!r}")
    match = SAMPLE_SUFFIX.search(row["sample_id"])
    if not match:
        raise ValueError(f"Unrecognised sample_id: {row['sample_id']}")
    epr_sample = int(match.group("epr"))
    wtir_sample = int(match.group("wtir"))
    if not (1 <= epr_sample <= n_epr and 1 <= wtir_sample <= n_wtir):
        raise ValueError(f"Sample outside {csv_experiment} grid: {row['sample_id']}")

    suffix = match.group(0)
    name = f"{scenario.lower()}_{technology}_{suffix}"
    formulation = "fully_decoupled" if method == "fully_decoupled_cross_driver" else method
    template = {
        "bundled": "uhs_bundled",
        "partially_decoupled": "uhs_partially_decoupled",
        "fully_decoupled": "uhs_fully_decoupled",
    }[formulation]
    result = {
        "template": template,
        "name": f"{scenario}: {technology.replace('_', ' ').title()} ({suffix})",
    }

    for source_column, parameter in COST_FIELDS.items():
        # Both MW/MWh and GW/GWh are allowed, but powers and energies must use
        # matching hour-consistent units (i.e. MWh with MW, GWh with GW).
        cost = require_decimal(row, source_column)
        if cost < 0:
            raise ValueError(f"Negative {source_column} in {row['instance_id']}")
        if source_column in VOM_FIELDS and cost * KWH_PER_MWH < min_vom_eur_per_mwh:
            # Suppress economically negligible VOM coefficients before
            # converting EUR/kWh to million EUR/model energy unit.
            cost = Decimal(0)
        result[f"{parameter}.data"] = yaml_number(cost * scale)

    if formulation == "bundled":
        result["energy_to_power_ratio"] = yaml_number(
            require_decimal(row, "EPR_days") * Decimal(24)
        )
        result["withdrawal_to_injection_ratio"] = yaml_number(require_decimal(row, "WTIR"))
    else:
        epr_bounds = bounds(row, "EPR_days_lower", "EPR_days_upper", Decimal(24))
        if epr_sample == n_epr and "max" in epr_bounds:
            # Extend the final EPR bin only. For FX4, epr01 retains its
            # original upper bound and epr02 alone receives this extension,
            # independently of which WTIR bin the sample belongs to.
            epr_bounds["max"] = yaml_number(
                require_decimal(row, "EPR_days_upper")
                * Decimal(24) * epr_upper_factor
            )
        for side, value in epr_bounds.items():
            result[f"energy_to_power_ratio_{side}"] = value

        if formulation == "partially_decoupled":
            result["withdrawal_to_injection_ratio"] = yaml_number(require_decimal(row, "WTIR"))
        else:
            wtir_bounds = bounds(row, "WTIR_lower", "WTIR_upper", Decimal(1))
            if wtir_sample == n_wtir and "max" in wtir_bounds:
                # Likewise, only the highest WTIR bin is extended; the
                # adjacent lower bin retains its original shared boundary.
                wtir_bounds["max"] = yaml_number(
                    require_decimal(row, "WTIR_upper") * wtir_upper_factor
                )
            for side, value in wtir_bounds.items():
                result[f"withdrawal_to_injection_ratio_{side}"] = value

    # The simplified CEM represents compression by equivalent directional flow
    # efficiencies, not by separate compressor-electricity parameters.
    injection_eff = optional_decimal(row, "injection_compression_equivalent_efficiency")
    if injection_eff is not None:
        if not (0 < injection_eff <= 1):
            raise ValueError(f"Invalid injection efficiency in {row['instance_id']}")
        result["flow_in_eff"] = yaml_number(injection_eff)

    # Withdrawal includes purification recovery and, where given, the
    # withdrawal-equivalent efficiency. Missing factors are neutral (1.0).
    recovery = optional_decimal(row, "purification_hydrogen_recovery")
    withdrawal_eff = optional_decimal(row, "withdrawal_compression_equivalent_efficiency")
    for label, value in (("hydrogen recovery", recovery), ("withdrawal efficiency", withdrawal_eff)):
        if value is not None and not (0 < value <= 1):
            raise ValueError(f"Invalid {label} in {row['instance_id']}")
    if recovery is not None or withdrawal_eff is not None:
        result["flow_out_eff"] = yaml_number(
            (recovery if recovery is not None else Decimal(1))
            * (withdrawal_eff if withdrawal_eff is not None else Decimal(1))
        )

    storage_loss = optional_decimal(row, "storage_loss_per_hour")
    if storage_loss is not None:
        if not (0 <= storage_loss < 1):
            raise ValueError(f"Invalid hourly storage loss in {row['instance_id']}")
        result["storage_loss"] = yaml_number(storage_loss)

    soc_values = {field: optional_decimal(row, field) for field in SOC_24H_FIELDS}
    if any(value is not None for value in soc_values.values()) and not all(
        value is not None for value in soc_values.values()
    ):
        raise ValueError(f"Incomplete 24-hour SoC coefficients in {row['instance_id']}")
    for field, value in soc_values.items():
        if value is not None:
            result[field] = yaml_number(value)
    return name, result


def build_yaml(
    rows: list[dict[str, str]], *, carrier: str, scale: Decimal,
    min_vom_eur_per_mwh: Decimal = DEFAULT_MIN_VOM_EUR_PER_MWH,
    lifetime_years: int = 50,
    epr_upper_factor: Decimal = DEFAULT_OUTER_UPPER_FACTOR,
    wtir_upper_factor: Decimal = DEFAULT_OUTER_UPPER_FACTOR,
) -> dict:
    for name, factor in (
        ("epr_upper_factor", epr_upper_factor),
        ("wtir_upper_factor", wtir_upper_factor),
    ):
        if not factor.is_finite() or factor < 1:
            raise ValueError(f"{name} must be finite and >= 1")
    definitions = {"templates": template_definitions(carrier, lifetime_years), "techs": {}}
    supplied = {key: set() for key in EXPERIMENTS}
    for row in rows:
        name, definition = sample_definition(
            row, scale, min_vom_eur_per_mwh,
            epr_upper_factor=epr_upper_factor,
            wtir_upper_factor=wtir_upper_factor,
        )
        if name in definitions["techs"]:
            raise ValueError(f"Duplicate technology definition: {name}")
        definitions["techs"][name] = definition
        supplied[row["experiment_id"]].add(name)

    for experiment, required in expected_names().items():
        missing = sorted(required - supplied[experiment])
        extra = sorted(supplied[experiment] - required)
        if missing or extra:
            raise ValueError(f"Experiment {experiment}: missing={missing}, extra={extra}")

    # Arrange YAML in experiment / WTIR / EPR / geology order to make the
    # generated file easy to compare with scenarios_uhs_representation.yaml.
    exp_order = {v[0].lower(): i for i, v in enumerate(EXPERIMENTS.values())}
    tech_order = {tech: i for i, tech in enumerate(TECHNOLOGIES)}

    def sort_key(name: str) -> tuple:
        scenario, remainder = name.split("_", 1)
        suffix = SAMPLE_SUFFIX.search(remainder)
        assert suffix is not None
        tech = remainder[: -(len(suffix.group()) + 1)]
        return (
            exp_order[scenario],
            int(suffix.group("wtir")),
            int(suffix.group("epr")),
            tech_order[tech],
        )

    definitions["techs"] = dict(sorted(definitions["techs"].items(), key=lambda item: sort_key(item[0])))
    return definitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Combined cost-rate CSV")
    parser.add_argument("--output", type=Path, default=Path("technologies_uhs.yaml"))
    parser.add_argument("--model-energy-unit", choices=KWH_PER_MODEL_ENERGY_UNIT, default="MWh")
    parser.add_argument("--hydrogen-carrier", default="hydrogen")
    parser.add_argument("--lifetime-years", type=int, default=50)
    parser.add_argument(
        "--min-vom-eur-per-mwh", type=Decimal, default=DEFAULT_MIN_VOM_EUR_PER_MWH,
        help="Set positive UHS variable flow costs below this EUR/MWh value to zero (default: 0.10).",
    )
    parser.add_argument(
        "--epr-upper-factor", type=Decimal, default=DEFAULT_OUTER_UPPER_FACTOR,
        help="Multiply only the highest sampled EPR interval's upper bound (default: 2).",
    )
    parser.add_argument(
        "--wtir-upper-factor", type=Decimal, default=DEFAULT_OUTER_UPPER_FACTOR,
        help="Multiply only the highest sampled WTIR interval's upper bound (default: 2).",
    )
    args = parser.parse_args()

    if args.lifetime_years <= 0:
        parser.error("--lifetime-years must be positive")
    if not args.min_vom_eur_per_mwh.is_finite() or args.min_vom_eur_per_mwh < 0:
        parser.error("--min-vom-eur-per-mwh must be finite and nonnegative")
    for option, factor in (
        ("--epr-upper-factor", args.epr_upper_factor),
        ("--wtir-upper-factor", args.wtir_upper_factor),
    ):
        if not factor.is_finite() or factor < 1:
            parser.error(f"{option} must be finite and at least 1")

    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    scale = (
        KWH_PER_MODEL_ENERGY_UNIT[args.model_energy_unit]
        * MEUR_PER_EUR
    )
    result = build_yaml(
        rows,
        carrier=args.hydrogen_carrier,
        scale=scale,
        min_vom_eur_per_mwh=args.min_vom_eur_per_mwh,
        lifetime_years=args.lifetime_years,
        epr_upper_factor=args.epr_upper_factor,
        wtir_upper_factor=args.wtir_upper_factor,
    )

    header = (
        "# AUTO-GENERATED from the combined UHS cost-rates CSV; do not edit sample rates here.\n"
        f"# Units: {args.model_energy_unit} hydrogen energy and matching power capacity "
        f"({'kW' if args.model_energy_unit == 'kWh' else 'MW' if args.model_energy_unit == 'MWh' else 'GW'}).\n"
        f"# Monetary coefficients are million EUR per {args.model_energy_unit} or matching power unit.\n"
        "# EPR parameters are hours (CSV EPR days multiplied by 24); WTIR is dimensionless.\n"
        "# EPR is interpreted by the custom maths as power <=/>= storage_cap / EPR.\n"
        "# B1/B25: both ratios exact; P1: WTIR exact and EPR interval;\n"
        "# F1/FX1/FX2/FX4: both ratio intervals constrain otherwise-independent capacities.\n"
        f"# Only the last sampled EPR bin has its upper limit multiplied by {args.epr_upper_factor};\n"
        f"# only the last sampled WTIR bin has its upper limit multiplied by {args.wtir_upper_factor}.\n"
        "# Original sample boundaries, cost rates, and bundled equality ratios are unchanged.\n"
        "# Extended intervals go beyond the range of the sampled cost-rate fits.\n"
        "# Compression is represented through flow_in_eff / flow_out_eff only;\n"
        "# raw compressor-electricity metadata is intentionally not exported.\n"
        "# 24-hour SoC coefficients are applied by uhs_storage_formulations.yaml.\n"
        f"# All UHS technologies are cyclic with lifetime {args.lifetime_years} years.\n"
        "# Cost annualisation uses the interest rate defined in the main model.\n"
        f"# Positive VOM coefficients below EUR {args.min_vom_eur_per_mwh}/MWh are set to zero.\n"
        "# CAPEX and annual fixed O&M are never clipped.\n"
        "# Adjacent sample cells share LP-compatible non-strict interval boundaries.\n\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
    counts = Counter(row["experiment_id"] for row in rows)
    print(f"Wrote {len(result['techs'])} technologies to {args.output}")
    print("CSV experiments:", dict(counts))
    print(f"Model units: {args.model_energy_unit} + matching power unit; cost scale x{scale}")
    print(
        "Outer sample upper-bound factors: "
        f"EPR x{args.epr_upper_factor}, WTIR x{args.wtir_upper_factor}"
    )
    clipped = Counter(
        source_column
        for row in rows
        for source_column in VOM_FIELDS
        if 0 < require_decimal(row, source_column) * KWH_PER_MWH < args.min_vom_eur_per_mwh
    )
    print(f"VOM entries zeroed below EUR {args.min_vom_eur_per_mwh}/MWh:", dict(clipped))


if __name__ == "__main__":
    main()
