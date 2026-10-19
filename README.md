# Flight Phase Classification

A Python package being prepared for ADS-B flight-phase recognition. Python 3.11 or newer is required.

The package currently provides stage labels, time utilities, canonical trajectory normalization, explicit ADS-B field mapping and CSV/Parquet loading. Models and training enter in later reviewed batches.

## Install

```sh
python -m pip install .
```

For Parquet files, install `python -m pip install ".[parquet]"`.

## Trajectory interface

```python
import pandas as pd
from flight_phase import load_trajectory, normalize_trajectory
from flight_phase.data.adapters import adapt_adsb
from flight_phase.data.time import elapsed_seconds

frame = pd.DataFrame({
    "flight_id": ["F001", "F001"],
    "timestamp": [0, 1],
    "altitude": [1000, 1050],
    "vertrate": [600, 620],
    "velocity": [145, 147],
})
trajectory = normalize_trajectory(frame)
seconds = elapsed_seconds(trajectory["timestamp"])

# Choose the ADS-B mapping explicitly when these are your source columns.
adsb_frame = frame.rename(columns={"vertrate": "vertical_rate", "velocity": "groundspeed"})
trajectory = adapt_adsb(adsb_frame)

# Read a canonical file containing one explicitly identified flight.
# trajectory = load_trajectory("flight.csv")
# If the file has no flight_id column, supply it explicitly:
# trajectory = load_trajectory("flight.csv", flight_id="F001")
```

Every entry point returns `point_index, flight_id, timestamp, altitude, vertrate, velocity`, followed by available ADS-B common columns (`latitude`, `longitude`, `track`, `on_ground`, `icao24`, `callsign`). Use `include_common=False` to omit common columns. Extra fields and incoming indices or labels are dropped; annotation alignment is handled separately.

Input units must already be ft, ft/min and kt. Numeric time is seconds and remains numeric; milliseconds and SI units are never inferred or converted. Datetime inputs are converted to UTC (naive datetimes are interpreted as UTC). `timestamp_seconds` preserves numeric origins; `elapsed_seconds` starts at zero and requires strictly increasing times for a single flight.

`normalize_trajectory` requires canonical column names. It sorts independently by string `flight_id` and time, assigns zero-based indices within each flight, and rejects duplicate times within a flight. Multiple flights are supported in memory; `load_trajectory` and `adapt_adsb` require exactly one flight. Missing or nonfinite kinematic observations remain NaN. Gaps are retained without interpolation, filling or segmentation.

`load_trajectory` reads canonical CSV/Parquet files. Dataset conversion and source detection belong in the caller's import scripts. The separate `adapt_adsb` function maps these explicitly named fields without converting units:

| Source | Time | Altitude | Vertical rate | Ground speed |
|---|---|---|---|---|
| Canonical input | `timestamp` | `altitude` | `vertrate` | `velocity` |
| `adapt_adsb` input | `timestamp` | `altitude` | `vertical_rate` | `groundspeed` |

The ADS-B adapter also maps optional `onground` to `on_ground` and preserves available common fields. Its required source columns must be present; missing columns raise `KeyError`. It selects the documented source fields even if other columns are present. Choose and rename any alternative altitude column before calling it. Callers using `traffic` pass `Flight.data`; `traffic` itself is not a dependency.

Provide a flight ID in the DataFrame or with `flight_id=`. Filenames, callsigns and timestamps never generate an ID. An explicit ID cannot override a different existing ID, and one flight_id cannot contain multiple ICAO addresses. CSV identifier strings retain leading zeros and literal values such as `NA`. Only ADS-B-equivalent trajectory fields enter the model interface; additional sensor, engine and fuel fields are excluded.

Four-stage probability order is `GROUND, CLIMB, CRUISE, DESCENT`; five-stage order is `GROUND, CLIMB, LVL, CR, DESCENT`. These are defined in `flight_phase.schema`. `UNKNOWN` is available for rejected predictions and is excluded from annotation classes and probability orders. There is no automatic four/five-stage label conversion.

## Development checks

Build and install the wheel before running tests. The tests exercise the installed package, including an isolated import outside the source checkout.

The public regression suite covers stage order, time semantics, flight isolation, explicit ADS-B mapping and canonical file reading using synthetic data. Dataset conversion checks remain local.

```sh
python -m pip install ".[dev,parquet]"
python -m build --outdir temp/dist
python -m pip install --force-reinstall --no-deps temp/dist/*.whl
python -m pytest -q
```

The three package installation checks can also run with the standard library's test runner (after installing the package and its dependencies):

```sh
python -m unittest discover -s tests -p test_package.py -v
```

CI uses Python 3.11 and 3.12 on Linux. Test scratch files are created under `temp/` and removed by their temporary-directory context. Build artifacts remain under `temp/dist/` for inspection.

## Repository layout

See `structure.md` for the current layout. The earlier migration and dataset conversion scripts are kept outside package discovery in ignored local material. The migration has a SHA-256 preservation manifest.

Raw flight data, annotations, experiment output, papers and trained weights are local artifacts and are excluded from version control.
