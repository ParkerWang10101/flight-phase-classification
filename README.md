# Flight Phase Classification

A Python package being prepared for ADS-B flight-phase recognition. Python 3.11 or newer is required.

The reviewed core implementation is currently on [`data-preprocessing-utils`](https://github.com/ParkerWang10101/flight-phase-classification/tree/data-preprocessing-utils). `main` contains repository documentation and ignore rules while the core awaits integration. Use the feature branch for the installation and examples below.

The package currently provides stage labels, canonical trajectories, explicit ADS-B mapping, file loading, flight-local kinematic cleaning, F1A features and training-input utilities. Model fitting enters in later reviewed batches.

## Install

```sh
git clone --branch data-preprocessing-utils --single-branch https://github.com/ParkerWang10101/flight-phase-classification.git
cd flight-phase-classification
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

Trajectory loading and normalization return `point_index, flight_id, timestamp, altitude, vertrate, velocity`, followed by available ADS-B common columns (`latitude`, `longitude`, `track`, `on_ground`, `icao24`, `callsign`). Use `include_common=False` to omit common columns. Extra fields and incoming indices or labels are dropped; annotation alignment is handled separately.

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

## Cleaning and F1A features

```python
from flight_phase.data.preprocessing import (
    F1A_FEATURE_COLUMNS, build_f1a_features, clean_trajectory,
    differentiate, rolling_mean,
)

# Canonical input; each flight and gap block gets independent history.
featured = build_f1a_features(frame, max_gap_seconds=5)
model_inputs = featured.loc[:, F1A_FEATURE_COLUMNS]

# Explicit offline cleaning when appropriate for the application.
cleaned = clean_trajectory(frame, max_gap_seconds=5)
offline_features = build_f1a_features(cleaned, max_gap_seconds=5)
```

F1A has exactly seven model features in this order: `altitude, vertrate, velocity, altitude_10s_mean, vertrate_10s_mean, velocity_10s_mean, velocity_d1_10s_mean`. Windows are causal sample means over `(t-10s, t]`, with missing samples omitted. Speed acceleration is a backward difference divided by the real elapsed seconds, then averaged over 10 seconds. Its unit is kt/s. The first acceleration of each block is NaN. F1A performs no implicit cleaning, filling or resampling.

Both entry points normalize the input, preserve flight-local `point_index`, and add `block_id`. A new flight or a time gap **greater than** `max_gap_seconds` starts fresh history. The default of 5 seconds is configurable, not a sampling requirement. All filters, interpolation and features stay inside those blocks.

`clean_trajectory` returns a new canonical DataFrame, leaving its input unchanged. It applies range checks (altitude -1500..65000 ft, vertical rate ±12000 ft/min, ground speed 0..800 kt), centered Hampel filtering (30 seconds, sigma 6), linear interpolation only between observed brackets at most 20 seconds apart, then a centered 15-second median for speed and vertical rate. Use `hampel_window_seconds=None` or `median_window_seconds=None` to disable either filter; `interpolation_gap_seconds=0` disables interpolation. Boolean `*_outlier` columns record range/Hampel rejection before filling.

Cleaning is **offline**: centered filters and interpolation use future observations. The original median behavior is retained: it can produce a finite value within its window where interpolation left NaN. Disable the median when that behavior is unwanted. Ground-state imputation, airport matching and collection quality gates are outside this API.

`rolling_mean(values, timestamps, window_seconds=10, center=False)` and `differentiate(values, timestamps, order=1)` are generic numerical operations on **one continuous block**, returning arrays in input order. Call them separately for each flight/gap block; `center=True` is offline. Higher derivatives are repeated backward differences, without bridging missing neighbours. The F1A builder is the only named feature pipeline; it does not expose F0–F3 variants or unused derived columns.

## Training inputs and nested flight folds

```python
from flight_phase.training import load_manifest, nested_folds, training_flights

# A CSV with flight_id, split, trajectory_path and annotation_path columns.
manifest = load_manifest("manifest.csv")
development_ids = manifest.loc[manifest["split"].eq("train"), "flight_id"].tolist()
folds = nested_folds(development_ids, outer=5, inner=4, seed=42)

for flight_id, trajectory, labels, source_hashes in training_flights("manifest.csv", "data"):
    # Canonical rows and labels have the same zero-based point positions.
    # source_hashes contains trajectory_sha256 and annotation_sha256.
    pass
```

Manifest IDs are unique, nonempty strings; values such as `001` and `NA` remain literal. Splits are explicitly `train`, `sealed` or `test`, with at least one training flight. Each training row names a canonical one-flight CSV/Parquet and a CSV annotation file. Paths resolve inside the supplied data root, including when symlinks are used. No filenames or local QAR formats are interpreted. Sealed/test rows may omit paths: their paths and file contents are never resolved, inspected or opened by these utilities.

Annotations require `flight_id, phase, start_index, end_index`. Indices are inclusive, ordered and cover every canonical point exactly once, without gaps or overlaps. Create annotations against the sorted `point_index` returned by `load_trajectory`; raw file row numbers and indices from filtered/resampled versions are not interchangeable. Incoming trajectory indices are discarded by normalization. The reader checks flight identity and interval coverage; callers must keep the corresponding trajectory and annotation versions together. It returns SHA-256 hashes of both source files for recording provenance.

`expand_labels(annotation, flight_id, point_count, stages=5)` also operates directly on an annotation DataFrame. Five-stage labels are the default; choose `stages=4` explicitly in either label expansion or `training_flights` for four-stage labels. Labels are never converted between stage systems, and `UNKNOWN` is rejected. A flight may contain only a subset of valid classes; no samples are copied or filled. If an annotation includes `status`, every interval must be `complete`.

`nested_folds` returns JSON-serializable dictionaries with zero-based `fold`, `train`, `validation` and nested `inner` entries. IDs are sorted before seeded shuffling, and every flight appears in outer validation exactly once. Inner validation covers only the corresponding outer training flights. The outer seed is 42 by default; inner seeds are `seed + outer_fold + 1`. Only feasible fold sizes are required, so 3 flights can form 3 outer / 2 inner folds. Membership matches shuffled scikit-learn KFold; scikit-learn is not required at runtime.

Supply **only development/train IDs** to the split utility; it does not know their manifest assignments. Fit scalers, clustering, calibration and every learned transform separately within each training fold. These utilities do not fit models, select configurations, expand short phases or prepare model-specific arrays. OOF results used for development and model selection remain development evidence; independent sealed/test evaluation is separate.

## Development checks

Build and install the wheel before running tests. The tests exercise the installed package, including an isolated import outside the source checkout.

The public regression suite covers stage order, time semantics, flight isolation, explicit ADS-B mapping, canonical file reading, cleaning, F1A, train-only input reading, label alignment and nested flight partitions using synthetic data. Dataset conversion and legacy-equivalence checks remain local.

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

See [`structure.md`](https://github.com/ParkerWang10101/flight-phase-classification/blob/data-preprocessing-utils/structure.md) for the current core layout. The earlier migration and dataset conversion scripts are kept outside package discovery in ignored local material. The migration has a SHA-256 preservation manifest.

Raw flight data, annotations, experiment output, papers and trained weights are local artifacts and are excluded from version control.
