# Core package structure

The project enters version control through independently reviewed modules. Batches 01–03 provide the package, data interfaces and preprocessing; batch 05 adds training inputs and nested flight folds. The boundary TCN follows on a separate topic branch without depending on the pending evaluation module. README is frozen until the final documentation pass.

```text
flight-phase-classification/
├── .github/workflows/ci.yml       Python 3.11/3.12 build and installation checks
├── pyproject.toml                package metadata and development dependencies
├── MANIFEST.in                   include all Python test support in the sdist
├── README.md                     frozen until the final documentation pass
├── structure.md                  current architecture
├── plan.md                       ignored local publication plan
├── src/flight_phase/
│   ├── __init__.py               version, normalize_trajectory and load_trajectory
│   ├── schema.py                 four/five-stage orders and model field names
│   ├── data/
│       ├── __init__.py            normalization and loading exports
│       ├── time.py                seconds and UTC time utilities
│       ├── time_support.py        block-local right-continuous point duration weights
│       ├── trajectory.py          canonical fields and boundary validation
│       ├── io.py                  one-flight CSV/Parquet loading
│       ├── adapters/
│           ├── __init__.py
│           └── adsb.py            explicit mapping for one ADS-B DataFrame
│       └── preprocessing/
│           ├── __init__.py        public cleaning, numerical operations and F1A
│           ├── _blocks.py         internal flight/gap partitioning
│           ├── cleaning.py        offline kinematic cleaning
│           └── features.py        block-local window means, derivatives and F1A
│   ├── models/
│   │   ├── __init__.py            no eager optional-model imports
│   │   ├── boundary_tcn.py        binary TCN, robust scaler, 1 Hz preparation and bundle
│   │   ├── segment_rf.py          interval features, fold-local RF and fitted state
│   │   └── segment_pipeline.py    point-aligned RF scoring and BGDP composition
│   ├── decoding/
│   │   ├── __init__.py            no eager decoder imports
│   │   └── bgdp.py                candidate peaks, transitions, interval scoring and DP
│   └── training/
│       ├── __init__.py            public input and split utilities
│       ├── dataset.py             manifests, labels, train-only reading and nested folds
│       └── boundary.py            Gaussian boundary targets and masked focal loss
├── tests/
│   ├── conftest.py               test-local temporary directories
│   ├── test_package.py           installed-package and distribution checks
│   ├── test_data.py              labels, time, normalization and explicit ADS-B mapping
│   ├── test_io.py                canonical CSV/Parquet and explicit reader boundaries
│   ├── test_preprocessing.py     numerical and isolation regressions
│   ├── test_training.py          synthetic labels, train-only reading and flight folds
│   ├── test_boundary_tcn.py      optional CPU network, alignment and checkpoint regressions
│   ├── test_segment_rf.py        time support, interval/RF and trusted-artifact regressions
│   └── test_bgdp.py              optimal paths, gaps, failures and TCN/RF integration
├── local/                       ignored local working material
│   ├── conversion/
│   │   ├── convert_trajectory.py dataset mapping and local import policies
│   │   └── README.md            local conversion instructions
│   └── research-staging/
│       ├── manifest.json        preservation paths and SHA-256 checksums
│       └── migration/           unchanged files from the previous migration
└── temp/                        ignored test and build artifacts
```

Only `src/flight_phase/` is searched for distributable packages. The data layer depends on NumPy, pandas and standard-library code; Parquet uses optional PyArrow. It has no dependency on local converters, models, training, evaluation, annotation applications or acquisition tools. The ADS-B adapter selects and renames documented source fields before calling generic normalization. Time tools live in the data layer so later metrics and models can share them without depending on HMM code.

The core requires canonical columns and an existing or explicitly supplied flight ID. Dataset-specific QAR mapping, automatic source detection, altitude fallback and filename-based IDs belong only to ignored local conversion scripts. No local format compatibility is installed or distributed. Dependency direction is local converter -> public data interface.

Public regression tests are organized by data behavior and file reading. They retain representative valid and invalid cases without enumerating every input representation or repeating common validation across all entry points. Package-installation tests remain separate; research and local conversion checks are not distributed.

Cleaning and F1A share private flight/gap partitioning after one canonical normalization at each public DataFrame entry point. Single-block numerical helpers expose sample-based rolling means and backward derivatives. The F1A pipeline calls their shared numerical implementations and emits only its seven model columns alongside canonical metadata and block_id. No multi-feature-set registry, compatibility aliases or plugin mechanism is introduced.

Cleaning returns new canonical values and outlier masks. It is explicitly offline and is not implicitly coupled to the causal F1A builder. Collection-specific rules, airport catalogues, ground semantics and quality reports remain in local research staging.

Training depends only on the public data layer, stage schema, NumPy, pandas and the standard library. It imports no models, metrics, annotation applications or experiment drivers. Manifest reading validates explicit assignments without opening referenced paths. The training iterator loads and validates the manifest at its boundary, opens only train rows, aligns labels to canonical point positions and returns source hashes. The label boundary validates interval coverage and the explicit four/five-stage contract. Nested folds operate on development IDs without reading files or fitting state; callers perform all fitting inside training folds. Seeded NumPy splits preserve the existing shuffled KFold membership without adding a runtime dependency.

Model-specific training packages, historical fold drivers, local QAR split generation, short-LVL relabeling and remote runners remain in ignored research staging. This batch contains no recipe registry, feature fallback, fixed historical sample count or artificial outer-folds × inner-folds minimum.

The optional `tcn` extra supplies PyTorch. Import `flight_phase.models.boundary_tcn` explicitly; core data imports do not load Torch. Residual convolutions, per-time LayerNorm, causal/offline padding, optional local branch/count head and tensor parameter names retain the original model computation. Training primitives preserve Gaussian targets and masked focal loss; experiment window sampling, selection and remote runners stay local.

The boundary input has the seven F1A columns followed by observation_present and observation_age_seconds. Explicit 1 Hz preparation rounds elapsed seconds with NumPy ties-to-even, retains the last observation per cell and forward-fills within each flight/gap. Every omitted grid cell restarts feature/network history, including short holes left after filling is exhausted. This corrects a legacy context-isolation defect; complete grids retain the original calculations. Output point_index is grid seconds; source_point_index maps retained observations to canonical positions. Score output is on this grid, not raw input rows. Subsecond rounding may shift a sample by half a second; causal mode applies to the gridded sequence. Bundles require explicit feature/scaler metadata, load Torch weights only after hash verification and refuse overwriting existing artifacts.

The optional `rf` extra supplies scikit-learn and joblib. Segment RF uses seven per-column F1A means, duration_seconds and log1p(duration_seconds), in a fixed nine-column order. Half-open intervals must remain inside one covered time block. Time support lives in the data layer, independent of evaluation: weights extend to the next point inside the same block, and each block's terminal point has zero duration. Explicit block labels override the automatic gap threshold; a coverage mask excludes unobserved points. Fit receives training rows and labels explicitly, fits medians/scaler/RF only there and retains the original 400-tree, depth-8, balanced-subsample defaults. Missing classes are never supplemented; their probability columns are zero and log scores -inf. Trusted joblib artifacts include feature and phase order and refuse overwriting; estimator/scaler dimensions are checked on loading.

The optional `decoding` extra supplies SciPy for BGDP score normalization. Candidates use plateau midpoint peaks, stable confidence-first time NMS and a per-block cap, with explicit endpoint nodes. Time-support hashes prevent scoring candidates against a different trajectory. Every interval is half-open and normalized over the canonical five phases; None and all--inf scores are infeasible, while NaN/+inf raise. BGDP combines class log probability, internal-boundary log odds, a segment penalty and trained transition scores. Consecutive segments must have different phases. Default candidate settings and minimum duration retain the existing computation; weights are explicit parameters rather than a fixed historical search grid. Forbidden -inf transitions remain forbidden even at zero transition weight. Failed blocks retain UNKNOWN with NO_FEASIBLE_PATH, and uncovered points stay UNKNOWN.

Transition fitting accepts only explicitly supplied labeled training flights and counts changes inside each covered block. It reads no files or validation labels. `models.segment_pipeline.decode_rf_segments` batches RF scores once per candidate block and calls the same BGDP core. Supply one already-featured flight and boundary scores on the same rows; pass TCN sequence_id as explicit block_ids when decoding a prepared grid. This interface does no source guessing, silent realignment, fitted-state updates or hyperparameter selection. Duration/GMM/HSMM research adapters, evaluation and historical tuning loops remain local.

No unified predictor or CLI is registered yet. Stage labels keep the existing four/five-stage distinction. The public data boundary validates fields, times and flight identity; downstream internal code may assume normalized input. CI checks the core and optional CPU model profiles on Python 3.11/3.12.

The original research project remains outside this checkout and is read-only throughout publication. Data, labels, model weights and experiment results are not part of the source package.
