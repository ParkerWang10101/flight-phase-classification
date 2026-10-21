"""Offline kinematic cleaning, isolated by flight and observation gaps."""

import numpy as np
import pandas as pd

from flight_phase.data.preprocessing._blocks import blocks
from flight_phase.data.trajectory import normalize_trajectory
from flight_phase.schema import BASE_FEATURE_COLUMNS

_RANGES = {"altitude": (-1500.0, 65000.0), "vertrate": (-12000.0, 12000.0), "velocity": (0.0, 800.0)}


def _hampel_mask(values, times, seconds, sigma):
    series = pd.Series(values, index=pd.to_timedelta(times, unit="s"), dtype=float)
    window = pd.to_timedelta(seconds, unit="s")
    median = series.rolling(window, center=True, min_periods=3).median()
    deviation = (series - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=3).median()
    return (deviation > sigma * 1.4826 * mad).fillna(False).to_numpy()


def _interpolate(values, times, maximum_seconds):
    result = values.copy()
    missing = np.isnan(result)
    position = 0
    while position < len(result):
        if not missing[position]:
            position += 1
            continue
        start = position
        while position < len(result) and missing[position]:
            position += 1
        left, right = start - 1, position
        if left < 0 or right == len(result):
            continue
        gap = times[right] - times[left]
        if gap <= maximum_seconds:
            fractions = (times[start:right] - times[left]) / gap
            result[start:right] = result[left] + fractions * (result[right] - result[left])
    return result


def clean_trajectory(
    data: pd.DataFrame,
    *,
    max_gap_seconds: float = 5.0,
    hampel_window_seconds: float | None = 30.0,
    hampel_sigma: float = 6.0,
    interpolation_gap_seconds: float = 20.0,
    median_window_seconds: float | None = 15.0,
) -> pd.DataFrame:
    """Return cleaned canonical values without modifying the input.

    Range checks -> centered Hampel -> bracketed short-gap interpolation ->
    centered median for velocity/vertrate. None disables a filter. This is an
    offline operation and is never implicitly applied by the causal F1A builder.
    No airport, ground-state or source-format assumptions are made.

    All operations restart per flight and gap > max_gap_seconds. Outlier flags
    describe range/Hampel rejection before filling. The legacy median can fill
    NaNs within its window even when interpolation did not fill them.
    """
    for name, value in (("hampel_window_seconds", hampel_window_seconds),
                        ("hampel_sigma", hampel_sigma), ("median_window_seconds", median_window_seconds)):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(interpolation_gap_seconds) or interpolation_gap_seconds < 0:
        raise ValueError("interpolation_gap_seconds must be finite and nonnegative")
    result = normalize_trajectory(data)
    cleaned = result.loc[:, BASE_FEATURE_COLUMNS].to_numpy(dtype=float, copy=True)
    outliers = np.zeros(cleaned.shape, dtype=bool)
    block_ids = np.empty(len(result), dtype=int)
    for block_id, positions, times in blocks(result, max_gap_seconds):
        block_ids[positions] = block_id
        for index, column in enumerate(BASE_FEATURE_COLUMNS):
            values = cleaned[positions, index].copy()
            lower, upper = _RANGES[column]
            invalid = np.isfinite(values) & ((values < lower) | (values > upper))
            values[invalid] = np.nan
            if hampel_window_seconds is not None:
                rejected = _hampel_mask(values, times, hampel_window_seconds, hampel_sigma)
                invalid |= rejected
                values[rejected] = np.nan
            values = _interpolate(values, times, interpolation_gap_seconds)
            if column != "altitude" and median_window_seconds is not None:
                series = pd.Series(values, index=pd.to_timedelta(times, unit="s"), dtype=float)
                values = series.rolling(
                    pd.to_timedelta(median_window_seconds, unit="s"), center=True, min_periods=1,
                ).median().to_numpy()
            cleaned[positions, index] = values
            outliers[positions, index] = invalid
    for index, column in enumerate(BASE_FEATURE_COLUMNS):
        result[column] = cleaned[:, index]
        result[f"{column}_outlier"] = outliers[:, index]
    result["block_id"] = block_ids
    return result
