"""Reusable Calliope 0.7 NetCDF extractors for storage result plots.

Each installed UHS construct is retained separately, including multiple
EPR/WTIR variants of one geological technology in FX2/FX4 runs.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
import xarray as xr


UHS_FAMILIES = (
    "salt_cavern", "aquifer", "depleted_gas_field", "lined_rock_cavern"
)
CAES_ID = "compressed_air_storage"
FILENAME = re.compile(
    r"^(?P<country>[A-Za-z]{3})_"
    r"(?P<experiment>(?:FX|B|P|F)\d+[A-Za-z]?)_"
    r"(?P<weather>weather_\d{4}_[A-Za-z0-9]+)$",
    flags=re.IGNORECASE,
)
UHS_ID = re.compile(
    r"^(?:fx|b|p|f)\d+[a-z]?_"
    r"(?P<family>salt_cavern|aquifer|depleted_gas_field|lined_rock_cavern)"
    r"(?:_|$)",
    flags=re.IGNORECASE,
)


def storage_family(tech_id: str) -> str | None:
    """Identify a geological family or CAES (ignore batteries and other techs)."""
    if tech_id == CAES_ID:
        return CAES_ID
    match = UHS_ID.match(tech_id)
    return match.group("family").lower() if match else None


def model_case(path: Path) -> dict[str, str]:
    """Decode a case without conflating different weather years/countries."""
    match = FILENAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Unexpected result filename {path.name!r}. Expected "
            "COUNTRY_EXPERIMENT_weather_YEAR_HORIZON.nc (e.g. NLD_FX2_weather_2015_3y.nc)."
        )
    return {**match.groupdict(), "case": path.stem}


def capacity_for_tech(ds: xr.Dataset, variable: str, tech_id: str) -> float:
    """Sum a result capacity over nodes/carriers for one concrete tech ID."""
    if variable not in ds:
        raise KeyError(f"Result variable {variable!r} missing from NetCDF results")
    array = ds[variable]
    if "techs" not in array.dims:
        raise ValueError(f"{variable!r} does not have a techs dimension")
    result = float(array.sel(techs=tech_id).fillna(0).sum().item())
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {variable} for {tech_id}")
    return result


def storage_soc(ds: xr.Dataset, tech_id: str) -> xr.DataArray:
    """Return a construct's raw storage time series (MWh) for future plots."""
    if "storage" not in ds:
        raise KeyError("Missing storage time series")
    return ds["storage"].sel(techs=tech_id).fillna(0)


def read_storage_investments(path: Path) -> pd.DataFrame:
    """Read invested geological storage constructs and CAES from one NC file.

    Uses results-group capacities, not fixed YAML EPR/WTIR assumptions.
    An FX2 case with two invested DGF constructs therefore produces two rows.
    """
    case = model_case(path)
    records: list[dict[str, str | float]] = []
    with xr.open_dataset(path, group="results") as ds:
        if "techs" not in ds.coords:
            raise ValueError(f"No techs coordinate in results group: {path}")
        for raw_id in ds.coords["techs"].values:
            tech_id = str(raw_id)
            family = storage_family(tech_id)
            if family is None:
                continue
            energy_mwh = capacity_for_tech(ds, "storage_cap", tech_id)
            if family == CAES_ID:
                # CAES has separate charging and discharging infrastructure.
                injection_mw = capacity_for_tech(ds, "flow_in_cap", tech_id)
                withdrawal_mw = capacity_for_tech(ds, "flow_out_cap", tech_id)
            else:
                injection_mw = capacity_for_tech(ds, "flow_in_cap", tech_id)
                withdrawal_mw = capacity_for_tech(ds, "flow_out_cap", tech_id)
            # Require meaningful installed energy and both flow capacities.
            if min(energy_mwh, injection_mw, withdrawal_mw) <= 1e-6:
                continue
            records.append({
                **case, "technology": family, "tech_id": tech_id,
                "storage_MWh": energy_mwh,
                "injection_MW": injection_mw,
                "withdrawal_MW": withdrawal_mw,
                "capacity_TWh": energy_mwh / 1e6,
                "EPR_days": energy_mwh / withdrawal_mw / 24,
                "WTIR": withdrawal_mw / injection_mw,
            })
    return pd.DataFrame.from_records(records)
