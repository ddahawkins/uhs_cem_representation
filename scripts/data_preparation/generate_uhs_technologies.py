"""Generate Calliope 0.7 UHS technology definitions from cost-rate samples.

Example (assuming Calliope models hydrogen in MW/MWh):

    python scripts/data_preparation/generate_uhs_technologies.py `
        --csv 'resources/uhs_techno_economics/uhs_cost_rates.csv' `
        --output 'config/technologies_uhs.yaml' `
        --model-energy-unit MWh `
        --hydrogen-carrier hydrogen

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
    "B30": ("B30", "bundled", 6, 5),
    "PD1": ("P1", "partially_decoupled", 1, 1),
    "FD1": ("F1", "fully_decoupled", 1, 1),
    "FX1": ("FX1", "fully_decoupled_cross_driver", 1, 1),
    "FX6": ("FX6", "fully_decoupled_cross_driver", 3, 2),
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

OPTIONAL_OPERATING_FIELDS = (
    "injection_compression_electricity_kWh_e_per_kWh_H2",
    "withdrawal_compression_electricity_kWh_e_per_kWh_H2",
    "injection_compression_equivalent_efficiency",
    "withdrawal_compression_equivalent_efficiency",
    "purification_hydrogen_recovery",
    "soc_24h_a_min",
    "soc_24h_b_min",
    "soc_24h_a_max",
    "soc_24h_b_max",
)

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


def template_definitions(carrier: str) -> dict:
    common = {
        "base_tech": "storage",
        "carrier_in": carrier,
        "carrier_out": carrier,
        # The hydrogen stored is not consumed to provide compressor electricity.
        # Compressor electricity must instead be linked in the CEM power balance.
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


def sample_definition(row: dict[str, str], scale: Decimal) -> tuple[str, dict]:
    csv_experiment = row["experiment_id"]
    if csv_experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment {csv_experiment!r}")
    scenario, expected_method, _, _ = EXPERIMENTS[csv_experiment]
    method = row["method"]
    if method != expected_method:
        raise ValueError(f"Unexpected method {method!r} in {row['instance_id']}")
    technology = row["technology"]
    if technology not in TECHNOLOGIES:
        raise ValueError(f"Unsupported UHS technology {technology!r}")
    match = SAMPLE_SUFFIX.search(row["sample_id"])
    if not match:
        raise ValueError(f"Unrecognised sample_id: {row['sample_id']}")

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
        result[f"{parameter}.data"] = yaml_number(require_decimal(row, source_column) * scale)

    if formulation == "bundled":
        result["energy_to_power_ratio"] = yaml_number(
            require_decimal(row, "EPR_days") * Decimal(24)
        )
        result["withdrawal_to_injection_ratio"] = yaml_number(require_decimal(row, "WTIR"))
    else:
        epr_bounds = bounds(row, "EPR_days_lower", "EPR_days_upper", Decimal(24))
        for side, value in epr_bounds.items():
            result[f"energy_to_power_ratio_{side}"] = value

        if formulation == "partially_decoupled":
            result["withdrawal_to_injection_ratio"] = yaml_number(require_decimal(row, "WTIR"))
        else:
            wtir_bounds = bounds(row, "WTIR_lower", "WTIR_upper", Decimal(1))
            for side, value in wtir_bounds.items():
                result[f"withdrawal_to_injection_ratio_{side}"] = value

    # Hydrogen recovery is an actual loss of stored hydrogen. It is distinct
    # from compressor electricity consumption, which is exported as a custom
    # parameter but must not be misrepresented as lost hydrogen.
    recovery = optional_decimal(row, "purification_hydrogen_recovery")
    if recovery is not None:
        if not (0 < recovery <= 1):
            raise ValueError(f"Invalid hydrogen recovery in {row['instance_id']}")
        result["flow_out_eff"] = yaml_number(recovery)

    storage_loss = optional_decimal(row, "storage_loss_per_hour")
    if storage_loss is not None:
        if not (0 <= storage_loss < 1):
            raise ValueError(f"Invalid hourly storage loss in {row['instance_id']}")
        result["storage_loss"] = yaml_number(storage_loss)

    for field in OPTIONAL_OPERATING_FIELDS:
        value = optional_decimal(row, field)
        if value is not None:
            result[field] = yaml_number(value)
    return name, result


def build_yaml(rows: list[dict[str, str]], *, carrier: str, scale: Decimal) -> dict:
    definitions = {"templates": template_definitions(carrier), "techs": {}}
    supplied = {key: set() for key in EXPERIMENTS}
    for row in rows:
        name, definition = sample_definition(row, scale)
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
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    scale = (
        KWH_PER_MODEL_ENERGY_UNIT[args.model_energy_unit]
        * MEUR_PER_EUR
    )
    result = build_yaml(rows, carrier=args.hydrogen_carrier, scale=scale)

    header = (
        "# AUTO-GENERATED from the combined UHS cost-rates CSV; do not edit sample rates here.\n"
        f"# Units: {args.model_energy_unit} hydrogen energy and matching power capacity "
        f"({'kW' if args.model_energy_unit == 'kWh' else 'MW' if args.model_energy_unit == 'MWh' else 'GW'}).\n"
        f"# Monetary coefficients are EUR per {args.model_energy_unit} or its matching power unit.\n"
        "# EPR parameters are hours (CSV EPR days multiplied by 24); WTIR is dimensionless.\n"
        "# B1/B30: both ratios exact; P1: WTIR exact and EPR interval;\n"
        "# F1/FX1/FX6: both ratio intervals constrain otherwise-independent capacities.\n"
        "# CSV experiments PD1 and FD1 become scenario prefixes P1 and F1.\n"
        "# H2 purification recovery is mapped to flow_out_eff; compressor-electricity\n"
        "# demand is preserved as separate technology parameters and is NOT YET\n"
        "# linked to electricity consumption in the custom maths.\n"
        "# SoC 24 h coefficients are exported for salt caverns but are NOT YET\n"
        "# operational constraints: they require a separate custom-math extension.\n"
        "# Set technology lifetimes and financial interest-rate assumptions before\n"
        "# using the investment costs in production CEM runs.\n"
        "# Adjacent sample cells share LP-compatible non-strict interval boundaries.\n\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
    counts = Counter(row["experiment_id"] for row in rows)
    print(f"Wrote {len(result['techs'])} technologies to {args.output}")
    print("CSV experiments:", dict(counts))
    print(f"Model units: {args.model_energy_unit} + matching power unit; cost scale x{scale}")


if __name__ == "__main__":
    main()
