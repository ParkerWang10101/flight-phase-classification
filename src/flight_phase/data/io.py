"""Load one explicitly identified canonical flight from CSV or Parquet."""

from pathlib import Path

import pandas as pd

from flight_phase.data.trajectory import normalize_trajectory


def load_trajectory(
    path: str | Path,
    *,
    flight_id: str | int | None = None,
    include_common: bool = True,
) -> pd.DataFrame:
    """Read exactly one flight with canonical column names and known units.

    CSV identifiers remain strings (including leading zeros). Parquet requires
    the optional ``parquet`` dependency. This function opens only the supplied
    path. A flight_id column or explicit argument is required; filenames are not
    interpreted and source formats are not detected or adapted.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        data = pd.read_csv(path, converters={
            name: lambda value: value or None
            for name in ("flight_id", "icao24", "callsign")
        })
    elif path.suffix.lower() == ".parquet":
        data = pd.read_parquet(path, engine="pyarrow")
    else:
        raise ValueError("trajectory files must be CSV or Parquet")
    return normalize_trajectory(
        data, flight_id=flight_id,
        single_flight=True, include_common=include_common,
    )
