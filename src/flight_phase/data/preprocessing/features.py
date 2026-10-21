"""Generic single-block time operations and the causal F1A feature pipeline."""

import numpy as np
import pandas as pd

from flight_phase.data.preprocessing._blocks import blocks
from flight_phase.data.time import timestamp_seconds
from flight_phase.data.trajectory import normalize_trajectory
from flight_phase.schema import BASE_FEATURE_COLUMNS

F1A_FEATURE_COLUMNS = (
    *BASE_FEATURE_COLUMNS,
    "altitude_10s_mean", "vertrate_10s_mean", "velocity_10s_mean",
    "velocity_d1_10s_mean",
)


def _inputs(values, timestamps):
    times = timestamp_seconds(timestamps)
    signal = np.asarray(values, dtype=float)
    if signal.ndim != 1 or len(signal) != len(times):
        raise ValueError("values must be one-dimensional and match timestamps")
    if (np.diff(times) <= 0).any():
        raise ValueError("timestamps must be strictly increasing")
    return signal, times


def _rolling_mean(values, times, window_seconds, center):
    series = pd.Series(values, index=pd.to_timedelta(times, unit="s"), dtype=float)
    return series.rolling(
        pd.to_timedelta(window_seconds, unit="s"),
        center=center, closed="right", min_periods=1,
    ).mean().to_numpy()


def rolling_mean(values, timestamps, *, window_seconds=10.0, center=False):
    """Sample mean in a real-time window on one continuous block.

    Input times must be finite and strictly increasing, in seconds or UTC.
    Call separately for each flight/gap block. Default support is (t-window, t];
    center=True is an offline window. NaN values are omitted, with at least one
    finite observation required. No samples are inserted or interpolated.
    """
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    signal, times = _inputs(values, timestamps)
    return _rolling_mean(signal, times, window_seconds, center)


def _differentiate(values, times, order):
    # Keep the original nanosecond time grid used by the feature implementation.
    seconds = pd.to_timedelta(times, unit="s").to_numpy().astype("timedelta64[ns]").astype(np.int64) / 1e9
    delta = np.diff(seconds)
    derivative = values
    for _ in range(order):
        result = np.full(len(values), np.nan)
        valid = (delta > 0) & np.isfinite(delta) & np.isfinite(np.diff(derivative))
        valid &= np.isfinite(derivative[:-1]) & np.isfinite(derivative[1:])
        positions = np.flatnonzero(valid) + 1
        result[positions] = np.diff(derivative)[valid] / delta[valid]
        derivative = result
    return derivative


def differentiate(values, timestamps, *, order=1):
    """Backward finite differences per second on one continuous block.

    Uses actual sample spacing. Missing neighbouring values yield NaN; the first
    point of each derivative order has no left support. Call separately for each
    flight/gap block. Velocity produces kt/s, vertical rate (ft/min)/s.
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    signal, times = _inputs(values, timestamps)
    return _differentiate(signal, times, order)


def build_f1a_features(data: pd.DataFrame, *, max_gap_seconds: float = 5.0) -> pd.DataFrame:
    """Normalize and compute the fixed seven-feature F1A view independently.

    Each flight and each timestamp gap > max_gap_seconds starts fresh history.
    Original fields remain canonical; four derived columns and block_id are
    added. Means use causal 10-second windows. No cleaning, resampling, forward
    filling or feature-set guessing occurs here.
    """
    result = normalize_trajectory(data)
    generated = np.full((len(result), 4), np.nan)
    block_ids = np.empty(len(result), dtype=int)
    for block_id, positions, times in blocks(result, max_gap_seconds):
        values = result.iloc[positions].loc[:, BASE_FEATURE_COLUMNS].to_numpy(dtype=float)
        block_ids[positions] = block_id
        for index in range(3):
            generated[positions, index] = _rolling_mean(values[:, index], times, 10.0, False)
        acceleration = _differentiate(values[:, 2], times, 1)
        generated[positions, 3] = _rolling_mean(acceleration, times, 10.0, False)
    result["block_id"] = block_ids
    result[list(F1A_FEATURE_COLUMNS[3:])] = generated
    return result
