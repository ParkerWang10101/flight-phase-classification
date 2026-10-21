"""Key numerical and isolation regressions for the public preprocessing API."""

import numpy as np
import pandas as pd
import pytest

from flight_phase.data.preprocessing import (
    F1A_FEATURE_COLUMNS, build_f1a_features, clean_trajectory, differentiate, rolling_mean,
)


def trajectory(times, *, flight_id="A", velocity=None):
    return pd.DataFrame({
        "flight_id": flight_id, "timestamp": times,
        "altitude": 1000.0, "vertrate": 0.0,
        "velocity": np.full(len(times), 100.0) if velocity is None else velocity,
    })


@pytest.mark.parametrize("utc", [False, True])
def test_real_time_windows_and_backward_differences(utc):
    times = np.array([0.0, 1.0, 3.0, 6.0])
    timestamps = pd.Timestamp("2026-01-01T00:00:00Z") + pd.to_timedelta(times, unit="s") if utc else times
    values = [0, 2, 8, 20]
    np.testing.assert_allclose(rolling_mean(values, timestamps, window_seconds=3), [0, 1, 5, 20])
    np.testing.assert_allclose(rolling_mean(values, timestamps, window_seconds=4, center=True), [1, 10 / 3, 8, 20])
    np.testing.assert_allclose(differentiate(values, timestamps), [np.nan, 2, 3, 4], equal_nan=True)
    np.testing.assert_allclose(differentiate(values, timestamps, order=2), [np.nan, np.nan, 0.5, 1 / 3], equal_nan=True)


def test_missing_neighbours_do_not_create_derivatives():
    values, times = [100, np.nan, 120], [0, 1, 2]
    assert np.isnan(differentiate(values, times)).all()
    np.testing.assert_allclose(rolling_mean(values, times), [100, 100, 110])


def test_f1a_has_seven_fixed_features_and_causal_history_without_implicit_cleaning():
    frame = trajectory([0, 1, 3, 6], velocity=[100, 102, 108, 120])
    frame["altitude"] = [1000, 1100, 1300, 1600]
    frame["vertrate"] = [100, 200, 300, 400]
    result = build_f1a_features(frame)
    assert F1A_FEATURE_COLUMNS == (
        "altitude", "vertrate", "velocity", "altitude_10s_mean",
        "vertrate_10s_mean", "velocity_10s_mean", "velocity_d1_10s_mean",
    )
    np.testing.assert_allclose(result.loc[:, F1A_FEATURE_COLUMNS].to_numpy(), [
        [1000, 100, 100, 1000, 100, 100, np.nan],
        [1100, 200, 102, 1050, 150, 101, 2],
        [1300, 300, 108, 3400 / 3, 200, 310 / 3, 2.5],
        [1600, 400, 120, 1250, 250, 107.5, 3],
    ], equal_nan=True)
    assert set(result) == {"point_index", "flight_id", "timestamp", "block_id", *F1A_FEATURE_COLUMNS}
    changed = frame.copy()
    changed.loc[3, "velocity"] = 5000
    updated = build_f1a_features(changed)
    pd.testing.assert_frame_equal(result.iloc[:3], updated.iloc[:3])
    assert updated.loc[3, "velocity"] == 5000


@pytest.mark.parametrize("operation", [clean_trajectory, build_f1a_features], ids=["cleaning", "f1a"])
def test_flights_and_gaps_match_independently_processed_blocks(operation):
    first = trajectory([0, 1, 2, 8, 9, 10], velocity=[100, 110, np.nan, np.nan, 710, 720])
    first["altitude"] = [1000, 1010, 1020, 50000, 50010, 50020]
    second = trajectory([0], flight_id="B", velocity=[250])
    parts = [first.iloc[:3], first.iloc[3:], second]
    combined = pd.concat([first, second], ignore_index=True).sample(frac=1, random_state=42)
    actual = operation(combined, max_gap_seconds=5)
    expected = pd.concat([operation(part, max_gap_seconds=5) for part in parts], ignore_index=True)
    pd.testing.assert_frame_equal(
        actual.drop(columns=["point_index", "block_id"]),
        expected.drop(columns=["point_index", "block_id"]),
    )
    assert actual["block_id"].tolist() == [0] * 3 + [1] * 3 + [2]
    assert actual["point_index"].tolist() == [0, 1, 2, 3, 4, 5, 0]


@pytest.mark.parametrize("interpolation_limit", [1.0, 2.0])
def test_cleaning_range_and_bracketed_interpolation_preserve_raw_input(interpolation_limit):
    frame = trajectory(np.arange(6), velocity=[np.nan, 100, np.nan, 120, np.nan, np.nan])
    frame["altitude"] = [np.nan, 1000, 100000, 1200, np.nan, np.nan]
    original = frame.copy(deep=True)
    result = clean_trajectory(frame, hampel_window_seconds=None, median_window_seconds=None,
                              interpolation_gap_seconds=interpolation_limit)
    assert result.loc[2, "altitude_outlier"]
    assert result.loc[[0, 4, 5], ["altitude", "velocity"]].isna().all(axis=None)
    if interpolation_limit == 2:
        assert result.loc[2, ["altitude", "velocity"]].tolist() == [1100, 110]
    else:
        assert result.loc[2, ["altitude", "velocity"]].isna().all()
    pd.testing.assert_frame_equal(frame, original)


def test_offline_hampel_and_median_remove_kinematic_spikes():
    frame = trajectory(np.arange(7))
    frame.loc[3, ["altitude", "vertrate", "velocity"]] = [20000, 8000, 400]
    result = clean_trajectory(frame)
    np.testing.assert_allclose(result[["altitude", "vertrate", "velocity"]], [[1000, 0, 100]] * 7)
    assert result.loc[3, ["altitude_outlier", "vertrate_outlier", "velocity_outlier"]].all()
    assert frame.loc[3, "velocity"] == 400
