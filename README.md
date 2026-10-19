# Flight Phase Classification

A Python package being prepared for ADS-B flight-phase recognition. Python 3.11 or newer is required.

The current batch contains the installable package foundation, distribution checks and CI. Data interfaces and model implementations will enter in separately reviewed batches. The installed package currently exposes only `flight_phase.__version__`.

## Install

```sh
python -m pip install .
```

## Development checks

Build and install the wheel before running tests. The tests exercise the installed package, including an isolated import outside the source checkout.

```sh
python -m pip install ".[dev]"
python -m build --outdir temp/dist
python -m pip install --force-reinstall --no-deps temp/dist/*.whl
python -m pytest -q
```

The package installation checks can also run with the standard library:

```sh
python -m unittest discover -s tests -v
```

CI uses Python 3.11 and 3.12 on Linux. Test scratch files are created under `temp/` and removed by their temporary-directory context. Build artifacts remain under `temp/dist/` for inspection.

## Repository layout

See `structure.md` for the current layout. The earlier migration is preserved locally under the ignored `local/research-staging/` directory with SHA-256 checksums; it is outside package discovery and is not distributed.

Raw flight data, annotations, experiment output, papers and trained weights are local artifacts and are excluded from version control.
