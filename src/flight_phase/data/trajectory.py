"""Normalize explicitly named trajectories in ft, ft/min and kt."""

import numpy as np
import pandas as pd

from flight_phase.data.time import _normalize_timestamps
from flight_phase.schema import BASE_FEATURE_COLUMNS

TRAJECTORY_COLUMNS = ("flight_id", "timestamp", *BASE_FEATURE_COLUMNS)
COMMON_COLUMNS = ("latitude", "longitude", "track", "on_ground", "icao24", "callsign")


def normalize_trajectory(
    data: pd.DataFrame,
    *,
    flight_id: str | int | None = None,
    single_flight: bool = False,
    include_common: bool = True,
) -> pd.DataFrame:
    """Validate canonical fields and sort independent flights without filling.

    Required fields are flight_id, timestamp, altitude, vertrate and velocity.
    An explicit flight_id may supply a missing column, but cannot replace an
    existing different ID. Units are already seconds/UTC, ft, ft/min and kt.

    No source detection, field aliases, identity generation or unit conversion
    is performed. Extra columns and incoming indices or labels are dropped.
    Missing/nonfinite observations become NaN. Gaps are retained; duplicate
    timestamps within a flight are rejected. point_index is zero-based per flight.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("trajectory is empty")
    if data.columns.duplicated().any():
        raise ValueError("duplicate column names")
    result = data.copy()
    if flight_id is not None:
        if "flight_id" in result and not result["flight_id"].astype(str).eq(str(flight_id)).all():
            raise ValueError("explicit flight_id differs from trajectory flight_id")
        result["flight_id"] = str(flight_id)
    missing = set(TRAJECTORY_COLUMNS) - set(result)
    if missing:
        raise ValueError(f"missing trajectory columns: {', '.join(sorted(missing))}")
    if result["flight_id"].isna().any() or result["flight_id"].astype(str).str.strip().eq("").any():
        raise ValueError("flight_id must not be empty")
    result["flight_id"] = result["flight_id"].astype(str)
    if single_flight and result["flight_id"].nunique() != 1:
        raise ValueError("one trajectory must contain exactly one flight_id")
    if "icao24" in result and result.groupby("flight_id")["icao24"].nunique().gt(1).any():
        raise ValueError("one flight_id cannot combine multiple icao24 values")
    result["timestamp"] = _normalize_timestamps(result["timestamp"])
    columns = [*TRAJECTORY_COLUMNS]
    if include_common:
        columns.extend(name for name in COMMON_COLUMNS if name in result)
    for name in (*BASE_FEATURE_COLUMNS, "latitude", "longitude", "track"):
        if name in columns:
            result[name] = pd.to_numeric(result[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    result = result.sort_values(["flight_id", "timestamp"], kind="stable").reset_index(drop=True)
    if result.duplicated(["flight_id", "timestamp"]).any():
        raise ValueError("duplicate timestamps within a flight")
    result = result.loc[:, columns]
    result.insert(0, "point_index", result.groupby("flight_id", sort=False).cumcount())
    return result
