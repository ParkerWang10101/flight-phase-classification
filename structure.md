# Core package structure

The project is entering version control through independently reviewed batches. The current candidate is batch 01: package foundation and checks.

```text
flight-phase-classification/
├── .github/workflows/ci.yml       Python 3.11/3.12 build and installation checks
├── pyproject.toml                package metadata and development dependencies
├── README.md                     capabilities available in this batch
├── structure.md                  current architecture
├── plan.md                       ignored local publication plan
├── src/flight_phase/
│   └── __init__.py               package version only
├── tests/
│   └── test_package.py           installed-package and distribution checks
├── local/                       ignored local working material
│   └── research-staging/
│       ├── manifest.json        preservation paths and SHA-256 checksums
│       └── migration/           unchanged files from the previous migration
└── temp/                        ignored test and build artifacts
```

Only `src/flight_phase/` is searched for distributable packages. Research staging is outside that tree. Future modules are added after their own review; the foundation does not import or advertise future model or command-line APIs.

The original research project remains outside this checkout and is read-only throughout publication. Data, labels, model weights and experiment results are not part of the source package.
