"""Numeric seconds and UTC timestamps, independent of models and metrics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _normalize_timestamps(values: pd.Series) -> pd.Series:
    """Validate timestamps without sorting or changing numeric seconds."""
    if values.empty or values.isna().any():
        raise ValueError("timestamp must be nonempty and contain no missing values")
    if pd.api.types.is_bool_dtype(values) or pd.api.types.is_complex_dtype(values):
        raise ValueError("timestamp must contain real seconds or UTC timestamps")
    if pd.api.types.is_timedelta64_dtype(values):
        raise ValueError("timedeltas must be converted explicitly to seconds")
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, utc=True, errors="raise")
    if pd.api.types.is_numeric_dtype(values):
        numeric = values
    else:
        numeric = pd.to_numeric(values, errors="coerce")
        if not numeric.notna().all():
            if numeric.notna().any():
                raise ValueError("mixed numeric and datetime timestamps")
            parsed = pd.to_datetime(values, utc=True, errors="raise", format="mixed")
            if parsed.isna().any():
                raise ValueError("timestamp must contain no missing values")
            return parsed
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("timestamp must contain finite seconds")
    return numeric


def timestamp_seconds(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    """Keep numeric seconds; express UTC datetimes relative to the first sample.

    Order is preserved. Use ``elapsed_seconds`` when strict increasing order is
    required. Neither helper guesses a numeric epoch or converts milliseconds.
    """
    if np.ndim(values) != 1:
        raise ValueError("timestamps must be a one-dimensional sequence")
    timestamps = _normalize_timestamps(pd.Series(values))
    if pd.api.types.is_numeric_dtype(timestamps):
        return timestamps.to_numpy(dtype=float)
    return (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()


def elapsed_seconds(values: Sequence[object] | np.ndarray | pd.Series) -> np.ndarray:
    """Return a single flight's finite, strictly increasing seconds from zero."""
    seconds = timestamp_seconds(values)
    if (np.diff(seconds) <= 0).any():
        raise ValueError("timestamps must be strictly increasing within a flight")
    return seconds - seconds[0]
