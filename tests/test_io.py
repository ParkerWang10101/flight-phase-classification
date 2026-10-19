"""Regression checks for canonical files and explicit reader boundaries."""

import pandas as pd
import pytest

from flight_phase import load_trajectory, normalize_trajectory


def trajectory():
    return pd.DataFrame({
        "flight_id": ["001", "001"], "timestamp": [1, 0],
        "altitude": [1050, 1000], "vertrate": [620, 600], "velocity": [147, 145],
    })


@pytest.mark.parametrize("format", ["csv", pytest.param("parquet", marks=pytest.mark.parquet)])
def test_file_roundtrip_preserves_utc_and_literal_identifiers(tmp_path, format):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    source = trajectory().assign(icao24=[None, "000123"], callsign="NA")
    source["timestamp"] = pd.to_datetime(["2026-01-01T00:00:01Z", "2026-01-01T00:00:00Z"])
    path = tmp_path / f"flight.{format}"
    if format == "csv":
        source.to_csv(path, index=False)
    else:
        source.to_parquet(path, index=False, engine="pyarrow")
    pd.testing.assert_frame_equal(load_trajectory(path), normalize_trajectory(source))


def test_reader_requires_an_existing_or_explicit_id_without_using_the_filename(tmp_path):
    path = tmp_path / "0007.csv"
    trajectory().drop(columns="flight_id").to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_trajectory(path)
    result = load_trajectory(path, flight_id="EXPLICIT")
    assert result["flight_id"].tolist() == ["EXPLICIT"] * 2
    assert result["timestamp"].tolist() == [0, 1]
    assert result[["altitude", "vertrate", "velocity"]].to_numpy().tolist() == [[1000, 600, 145], [1050, 620, 147]]


@pytest.mark.parametrize("problem", ["multiple-flights", "source-aliases"])
def test_reader_requires_one_canonical_flight(tmp_path, problem):
    source = trajectory()
    if problem == "multiple-flights":
        source.loc[0, "flight_id"] = "OTHER"
    else:
        source = source.rename(columns={"vertrate": "vertical_rate", "velocity": "groundspeed"})
    path = tmp_path / "flight.csv"
    source.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_trajectory(path)
