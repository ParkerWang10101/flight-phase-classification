# Core package structure

The project enters version control through independently reviewed batches. Batches 01–02 provide the package and data interfaces; the current candidate is batch 03: kinematic cleaning, general time operations and the isolated F1A pipeline.

```text
flight-phase-classification/
├── .github/workflows/ci.yml       Python 3.11/3.12 build and installation checks
├── pyproject.toml                package metadata and development dependencies
├── MANIFEST.in                   include all Python test support in the sdist
├── README.md                     capabilities available in this batch
├── structure.md                  current architecture
├── plan.md                       ignored local publication plan
├── src/flight_phase/
│   ├── __init__.py               version, normalize_trajectory and load_trajectory
│   ├── schema.py                 four/five-stage orders and model field names
│   └── data/
│       ├── __init__.py            normalization and loading exports
│       ├── time.py                seconds and UTC time utilities
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
├── tests/
│   ├── conftest.py               test-local temporary directories
│   ├── test_package.py           installed-package and distribution checks
│   ├── test_data.py              labels, time, normalization and explicit ADS-B mapping
│   ├── test_io.py                canonical CSV/Parquet and explicit reader boundaries
│   └── test_preprocessing.py     numerical and isolation regressions
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

No model, predictor or CLI is registered yet. Stage labels keep the existing four/five-stage distinction. The public data boundary validates fields, times and flight identity; downstream internal code may assume normalized input.

The original research project remains outside this checkout and is read-only throughout publication. Data, labels, model weights and experiment results are not part of the source package.
