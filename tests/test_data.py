"""Key regressions for the public labels, time and trajectory interfaces."""

import numpy as np
import pandas as pd
import pytest

from flight_phase import normalize_trajectory
from flight_phase.data.adapters import adapt_adsb
from flight_phase.data.time import elapsed_seconds, timestamp_seconds
from flight_phase.data.time_support import build_time_support
from flight_phase.schema import (
    ANNOTATABLE_PHASES, FIVE_STAGE_ANNOTATABLE_PHASES,
    FIVE_STAGE_PHASES, FOUR_STAGE_PHASES, FiveStageFlightPhase, FlightPhase,
)


def trajectory(flight_id="001"):
    return pd.DataFrame({
        "flight_id": flight_id, "timestamp": [1, 0, 1000],
        "altitude": [1050, 1000, 2000], "vertrate": [620, 600, 0],
        "velocity": [147, 145, 200],
    })


def adsb_frame():
    return trajectory().rename(columns={"vertrate": "vertical_rate", "velocity": "groundspeed"}).assign(
        icao24="000123", callsign="TEST", onground=False,
    )


def test_stage_order_and_unknown_policy():
    assert FOUR_STAGE_PHASES == ("GROUND", "CLIMB", "CRUISE", "DESCENT")
    assert FIVE_STAGE_PHASES == ("GROUND", "CLIMB", "LVL", "CR", "DESCENT")
    assert FlightPhase.UNKNOWN.value == FiveStageFlightPhase.UNKNOWN.value == "UNKNOWN"
    assert FlightPhase.UNKNOWN not in ANNOTATABLE_PHASES
    assert FiveStageFlightPhase.UNKNOWN not in FIVE_STAGE_ANNOTATABLE_PHASES


@pytest.mark.parametrize(("values", "seconds"), [
    ([100, 102.5, 110], [100, 102.5, 110]),
    (["2026-01-01T09:00:00+09:00", "2026-01-01T00:00:00.5Z", "2026-01-01T00:00:02Z"], [0, 0.5, 2]),
], ids=["numeric-seconds", "utc-offsets"])
def test_time_origins_and_elapsed_seconds(values, seconds):
    np.testing.assert_array_equal(timestamp_seconds(values), seconds)
    np.testing.assert_array_equal(elapsed_seconds(values), np.asarray(seconds) - seconds[0])


@pytest.mark.parametrize("values", [[1, 0], [0, np.inf]], ids=["decreasing", "nonfinite"])
def test_elapsed_time_rejects_invalid_sequences(values):
    with pytest.raises(ValueError):
        elapsed_seconds(values)


def test_normalization_sorts_flights_independently_and_keeps_gaps():
    source = pd.concat([trajectory("B"), trajectory("A")], ignore_index=True).sample(frac=1, random_state=42)
    original = source.copy(deep=True)
    result = normalize_trajectory(source)
    assert result["flight_id"].tolist() == ["A"] * 3 + ["B"] * 3
    assert result["timestamp"].tolist() == [0, 1, 1000] * 2
    assert result["point_index"].tolist() == [0, 1, 2] * 2
    assert result["altitude"].tolist() == [1000, 1050, 2000] * 2
    pd.testing.assert_frame_equal(source, original)


def test_normalization_keeps_missing_observations_and_only_public_fields():
    source = trajectory().astype({"altitude": float, "velocity": object})
    source.loc[0, "altitude"] = np.inf
    source.loc[0, "velocity"] = "bad"
    source = source.assign(engine_power=90, phase="CLIMB", point_index=999, groundspeed=999)
    result = normalize_trajectory(source)
    assert result.columns.tolist() == ["point_index", "flight_id", "timestamp", "altitude", "vertrate", "velocity"]
    assert result.loc[0, ["altitude", "vertrate", "velocity"]].tolist() == [1000, 600, 145]
    assert result.loc[1, ["altitude", "velocity"]].isna().all()


@pytest.mark.parametrize("problem", ["duplicate-time", "missing-time", "missing-id"])
def test_normalization_rejects_broken_alignment(problem):
    source = trajectory()
    if problem == "duplicate-time":
        source.loc[0, "timestamp"] = 0
    elif problem == "missing-time":
        source.loc[0, "timestamp"] = np.nan
    else:
        source.loc[0, "flight_id"] = None
    with pytest.raises(ValueError):
        normalize_trajectory(source)


def test_explicit_id_supplies_a_missing_column_without_overwriting_an_existing_id():
    with pytest.raises(ValueError):
        normalize_trajectory(trajectory(), flight_id="OTHER")
    result = normalize_trajectory(trajectory().drop(columns="flight_id"), flight_id="EXPLICIT")
    assert result["flight_id"].unique().tolist() == ["EXPLICIT"]


def test_adsb_mapping_uses_documented_fields_without_altitude_fallback():
    source = adsb_frame().astype({"altitude": float}).assign(geoaltitude=9000, velocity=999)
    source["timestamp"] = ["2026-01-01T09:00:01+09:00", "2026-01-01T00:00:00Z", "2026-01-01T00:16:40Z"]
    result = adapt_adsb(source)
    assert str(result["timestamp"].dt.tz) == "UTC"
    assert result["altitude"].tolist() == [1000, 1050, 2000]
    assert result["vertrate"].tolist() == [600, 620, 0]
    assert result["velocity"].tolist() == [145, 147, 200]
    assert result["on_ground"].tolist() == [False] * 3
    source["altitude"] = np.nan
    assert adapt_adsb(source)["altitude"].isna().all()


@pytest.mark.parametrize(("missing", "error"), [("flight_id", ValueError), ("altitude", KeyError)])
def test_adsb_requires_explicit_identity_and_source_columns(missing, error):
    source = adsb_frame().assign(geoaltitude=9000).drop(columns=missing)
    with pytest.raises(error):
        adapt_adsb(source)


def test_multiple_aircraft_are_rejected_even_when_common_columns_are_omitted():
    source = adsb_frame()
    source.loc[0, "icao24"] = "OTHER"
    with pytest.raises(ValueError):
        adapt_adsb(source, include_common=False)


def test_time_support_counts_only_covered_continuous_intervals():
    support = build_time_support([0, 1, 3, 20, 21])
    assert support.block_ids.tolist() == [0, 0, 0, 1, 1]
    assert support.weights_seconds.tolist() == [1, 2, 0, 1, 0]
    assert support.effective_seconds == 4
    explicit = build_time_support([0, 1, 3, 20, 21], block_ids=["A"] * 5,
                                  coverage_mask=[True, True, False, True, True])
    assert explicit.block_ids.tolist() == [0, 0, -1, 1, 1]
    assert explicit.weights_seconds.tolist() == [1, 0, 0, 1, 0]
    bridged = build_time_support([0, 20], block_ids=["A", "A"])
    assert bridged.weights_seconds.tolist() == [20, 0]
