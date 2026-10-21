"""Internal block iteration; callers supply a normalized trajectory."""

import numpy as np

from flight_phase.data.time import timestamp_seconds


def blocks(frame, max_gap_seconds):
    if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be finite and positive")
    block_id = 0
    for positions in frame.groupby("flight_id", sort=False).indices.values():
        times = timestamp_seconds(frame.iloc[positions]["timestamp"])
        boundaries = np.r_[0, np.flatnonzero(np.diff(times) > max_gap_seconds) + 1, len(times)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            yield block_id, positions[start:stop], times[start:stop]
            block_id += 1
