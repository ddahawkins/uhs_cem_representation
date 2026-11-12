from pathlib import Path

import numpy as np
import pandas as pd

country = "GBR"

root = Path("C:/dev/modelblocks/module_hydropower")

inflow = pd.read_parquet(
    root / f"results/{country}/disaggregated/inflow_mwh.parquet"
)

plants = pd.read_parquet(
    root / (
        "resources/module/resources/automatic/"
        f"shapes/{country}/adjusted_powerplants.parquet"
    )
)

profiles = {}

for tech, name in [
    ("run of river", "hydro_run_of_river"),
    ("reservoir", "hydro_reservoir"),
]:
    plant_ids = plants.loc[
        plants["technology"] == tech,
        "powerplant_id",
    ].tolist()

    missing = set(plant_ids) - set(inflow.columns)
    if missing:
        raise ValueError(f"Missing inflow columns: {missing}")

    # Sum plant-level MWh into national hourly MWh
    hourly = inflow[plant_ids].sum(axis=1, min_count=1)

    # Convert hourly energy to two-hourly energy
    profiles[name] = hourly

hydro = pd.DataFrame(profiles)

print(hydro.describe())

print("\nTemporal coverage:")
print("First timestep:", hydro.index.min())
print("Last timestep:", hydro.index.max())
print("Number of timesteps:", len(hydro))

print("\nTimestep intervals:")
print(hydro.index.to_series().diff().value_counts().head())

annual_twh = hydro.resample("YE").sum() / 1e6
annual_twh.index = annual_twh.index.year
annual_twh["total"] = annual_twh.sum(axis=1)

print("\nAnnual hydro inflows (TWh):")
print(annual_twh.round(2).to_string())

# ============================================================================
# EXPORT HYDROPOWER INFLOWS
# ============================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

output_dir = (
    PROJECT_ROOT
    / "resources"
    / "timeseries"
    / "hydropower"
    / country
)

output_dir.mkdir(parents=True, exist_ok=True)

hydro_export = hydro[
    ["hydro_run_of_river", "hydro_reservoir"]
].copy()

hydro_export.index = pd.to_datetime(hydro_export.index)
hydro_export.index.name = "timesteps"
hydro_export = hydro_export.sort_index()

# Validate before exporting.
expected_index = pd.date_range(
    "2009-01-01 00:00:00",
    "2024-12-31 23:00:00",
    freq="h",
)

if not hydro_export.index.equals(expected_index):
    raise ValueError(
        f"{country}: Hydropower timeseries has missing, "
        "duplicate, or unexpected timestamps."
    )

if not np.isfinite(hydro_export.to_numpy()).all():
    raise ValueError(f"{country}: Non-finite hydropower inflows.")

if (hydro_export < 0).any().any():
    raise ValueError(f"{country}: Negative hydropower inflows.")

# Preserve original numerical values.
hydro_export.to_parquet(
    output_dir / "inflows_hourly_mwh.parquet"
)

# CSV format for Calliope data_tables.
hydro_export.to_csv(
    output_dir / "inflows_hourly_mwh.csv",
    date_format="%Y-%m-%d %H:%M:%S",
    float_format="%.12g",
)

print(f"\nSaved hydropower inflows: {country}")
print(f"Location: {output_dir}")
print(f"Rows: {len(hydro_export):,}")