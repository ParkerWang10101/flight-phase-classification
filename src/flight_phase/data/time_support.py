"""Right-continuous point support for one canonical flight, without metrics."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from flight_phase.data.time import elapsed_seconds

DEFAULT_GAP_SECONDS = 5.0


@dataclass(frozen=True)
class TimeSupport:
    """每个采样点对应的右连续时间支持及有效连续块。"""

    times: np.ndarray
    block_ids: np.ndarray
    weights_seconds: np.ndarray
    coverage_mask: np.ndarray
    gap_threshold_seconds: float
    explicit_blocks: bool

    @property
    def effective_seconds(self) -> float:
        return float(self.weights_seconds.sum())


def build_time_support(
    timestamps: Sequence[object] | np.ndarray,
    *,
    block_ids: Sequence[object] | np.ndarray | None = None,
    coverage_mask: Sequence[bool] | np.ndarray | None = None,
    gap_threshold_seconds: float = DEFAULT_GAP_SECONDS,
) -> TimeSupport:
    """建立右连续时间支持；显式块优先于缺口阈值。"""

    times = elapsed_seconds(timestamps)
    n = len(times)
    if not np.isfinite(gap_threshold_seconds) or gap_threshold_seconds <= 0:
        raise ValueError("gap_threshold_seconds 必须大于 0")
    coverage = (
        np.ones(n, dtype=bool)
        if coverage_mask is None
        else np.asarray(coverage_mask, dtype=bool)
    )
    if coverage.shape != (n,):
        raise ValueError("coverage_mask 长度必须与 timestamps 一致")

    explicit = block_ids is not None
    raw_blocks = None if block_ids is None else np.asarray(block_ids, dtype=object)
    if raw_blocks is not None and raw_blocks.shape != (n,):
        raise ValueError("block_ids 长度必须与 timestamps 一致")

    normalized = np.full(n, -1, dtype=int)
    current = -1
    previous_valid = False
    previous_raw: object | None = None
    for index in range(n):
        if not coverage[index]:
            previous_valid = False
            previous_raw = None
            continue
        starts_new = not previous_valid
        if previous_valid:
            if raw_blocks is not None:
                starts_new = raw_blocks[index] != previous_raw
            else:
                starts_new = (times[index] - times[index - 1]) > gap_threshold_seconds
        if starts_new:
            current += 1
        normalized[index] = current
        previous_valid = True
        previous_raw = raw_blocks[index] if raw_blocks is not None else current

    weights = np.zeros(n, dtype=float)
    for index in range(n - 1):
        if normalized[index] >= 0 and normalized[index] == normalized[index + 1]:
            weights[index] = times[index + 1] - times[index]
    return TimeSupport(
        times=times,
        block_ids=normalized,
        weights_seconds=weights,
        coverage_mask=coverage,
        gap_threshold_seconds=float(gap_threshold_seconds),
        explicit_blocks=explicit,
    )
