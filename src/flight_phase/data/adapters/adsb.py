"""Adapt one ADS-B DataFrame already expressed in imperial units."""

import pandas as pd

from flight_phase.data.trajectory import normalize_trajectory

_REQUIRED_COLUMNS = ("timestamp", "altitude", "vertical_rate", "groundspeed")
_COMMON_COLUMNS = ("flight_id", "latitude", "longitude", "track", "onground", "icao24", "callsign")
_COLUMN_MAP = {"vertical_rate": "vertrate", "groundspeed": "velocity", "onground": "on_ground"}


def adapt_adsb(
    data: pd.DataFrame,
    *,
    flight_id: str | int | None = None,
    include_common: bool = True,
) -> pd.DataFrame:
    """Map timestamp/altitude/vertical_rate/groundspeed explicitly.

    Input units are already ft, ft/min and kt. A flight_id column or explicit
    argument is required. No alternate altitude or identity is inferred. Missing
    required source columns raise KeyError. Other source fields are excluded.
    """
    columns = [*_REQUIRED_COLUMNS, *(name for name in _COMMON_COLUMNS if name in data)]
    mapped = data.loc[:, columns].rename(columns=_COLUMN_MAP)
    return normalize_trajectory(
        mapped, flight_id=flight_id,
        single_flight=True, include_common=include_common,
    )
