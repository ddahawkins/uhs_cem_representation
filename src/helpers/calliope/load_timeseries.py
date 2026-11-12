"""Demand, VRES, and absolute hydropower inflows for Calliope 0.7.

Expected resources (relative to project working directory):
  resources/timeseries/demand.parquet
  resources/timeseries/ESP.parquet (and other countries)
  resources/timeseries/hydropower/ESP/inflows_hourly_mwh.parquet

Hydropower input is already in electrical MWh per *hour*. Do NOT divide by
capacity or resample here; Calliope's configured resampling handles MWh sums.
"""

from pathlib import Path

import numpy as np
import pandas as pd

AVAILABILITY_THRESHOLD = 1e-3
HYDRO_TECHS = ("hydro_run_of_river", "hydro_reservoir")

# These countries had no conventional hydro plants in the ModelBlocks inventory.
# They are intentionally assigned ZERO hydro availability, not an unrestricted
# Calliope default. Revisit this choice if better data become available.
ZERO_HYDRO_COUNTRIES = frozenset({"DNK", "NLD"})


def _utc_indexed(df: pd.DataFrame, label: str, time_column: str | None = None) -> pd.DataFrame:
    """Return a sorted, UTC-indexed copy; reject ambiguous or duplicate times."""
    df = df.copy()
    if time_column is not None and time_column in df.columns:
        df = df.set_index(time_column)
    elif not isinstance(df.index, pd.DatetimeIndex) and df.index.name not in (
        "timestamp", "timesteps", "time"
    ):
        raise ValueError(f"{label}: no recognised datetime index or column")

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timesteps"
    if df.index.has_duplicates:
        raise ValueError(f"{label}: duplicate timestamps")
    return df.sort_index()


def _require_index(df: pd.DataFrame, expected: pd.DatetimeIndex, label: str) -> None:
    if not df.index.equals(expected):
        missing = expected.difference(df.index)
        extra = df.index.difference(expected)
        raise ValueError(
            f"{label}: timestamps do not match the requested hourly horizon; "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"first_missing={missing[:3].tolist()}"
        )


def load_country_timeseries(
    country: str,
    start_date: str,
    end_date: str,
    demand_path: Path = Path("resources/timeseries/demand.parquet"),
    vres_dir: Path = Path("resources/timeseries"),
    hydro_dir: Path = Path("resources/timeseries/hydropower"),
) -> pd.DataFrame:
    """Load one country's hourly demand, VRES, and hydro over [start, end) UTC."""
    start = pd.to_datetime(start_date, utc=True)
    end = pd.to_datetime(end_date, utc=True)
    if end <= start:
        raise ValueError("end_date must be later than start_date")
    if start.minute or start.second or end.minute or end.second:
        raise ValueError("start_date and end_date must fall on full hours")

    expected = pd.date_range(start, end, freq="h", inclusive="left", name="timesteps")

    # Demand: keep the existing parquet structure and selected-country column.
    demand = pd.read_parquet(demand_path, columns=["timestamp", country])
    demand = _utc_indexed(demand, "demand", time_column="timestamp")
    demand = demand.loc[(demand.index >= start) & (demand.index < end)]

    # VRES: per-unit hourly availability (not MWh).
    vres = pd.read_parquet(
        vres_dir / f"{country}.parquet",
        columns=["solar", "onshore_wind", "offshore_wind"],
    )
    vres = _utc_indexed(vres, "VRES")
    vres = vres.loc[(vres.index >= start) & (vres.index < end)]

    small = (vres > 0) & (vres < AVAILABILITY_THRESHOLD)
    if small.any().any():
        print(f"{country}: set negligible VRES availability to zero: {small.sum().to_dict()}")
        vres = vres.mask(small, 0.0)

    _require_index(demand, expected, f"{country} demand")
    _require_index(vres, expected, f"{country} VRES")

    # Hydro: absolute electrical MWh available in each 1-hour timestep.
    hydro_path = hydro_dir / country / "inflows_hourly_mwh.parquet"
    if hydro_path.is_file():
        hydro = pd.read_parquet(hydro_path)
        hydro = _utc_indexed(hydro, f"{country} hydro", time_column="timesteps")
        missing_cols = set(HYDRO_TECHS).difference(hydro.columns)
        if missing_cols:
            raise ValueError(f"{country}: hydro file missing columns {sorted(missing_cols)}")
        hydro = hydro.loc[(hydro.index >= start) & (hydro.index < end), list(HYDRO_TECHS)]
        _require_index(hydro, expected, f"{country} hydro")
    elif country in ZERO_HYDRO_COUNTRIES:
        # Explicit, documented choice; never leave hydro source_use_max undefined.
        print(f"{country}: no hydro inventory; setting both hydro inflows to 0 MWh")
        hydro = pd.DataFrame(0.0, index=expected, columns=list(HYDRO_TECHS))
    else:
        raise FileNotFoundError(
            f"{country}: missing hydropower data: {hydro_path}. "
            "Do not run with unconstrained hydro supply."
        )

    vals = hydro.to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise ValueError(f"{country}: hydro contains NaN, infinite, or negative inflows")

    timeseries = pd.DataFrame(
        {
            "demand_power": demand[country],
            "solar": vres["solar"],
            "onshore_wind": vres["onshore_wind"],
            "offshore_wind": vres["offshore_wind"],
            "hydro_run_of_river": hydro["hydro_run_of_river"],
            "hydro_reservoir": hydro["hydro_reservoir"],
        },
        index=expected,
    )
    if not np.isfinite(timeseries.to_numpy(dtype=float)).all():
        raise ValueError(f"{country}: non-finite values in assembled timeseries")

    return timeseries


def build_calliope_timeseries(
    country: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return one table indexed by timesteps with columns (techs, inputs)."""
    df = load_country_timeseries(country, start_date, end_date)
    mappings = {
        ("demand_power", "sink_use_equals"): "demand_power",
        ("solar", "source_use_max"): "solar",
        ("onshore_wind", "source_use_max"): "onshore_wind",
        ("offshore_wind", "source_use_max"): "offshore_wind",
        ("hydro_run_of_river", "source_use_max"): "hydro_run_of_river",
        ("hydro_reservoir", "source_use_max"): "hydro_reservoir",
    }
    out = pd.DataFrame({key: df[column] for key, column in mappings.items()}, index=df.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["techs", "inputs"])
    out.index.name = "timesteps"
    return out
