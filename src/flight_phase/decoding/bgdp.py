"""Boundary-guided dynamic programming on explicit five-stage interval scores.

Point indices are half-open. Candidate extraction and decoding restart at every
covered time block. No training truth, metrics or experiment grid is imported.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from operator import index

import numpy as np
from scipy.special import logsumexp

from flight_phase.data.time_support import TimeSupport
from flight_phase.schema import FIVE_STAGE_PHASES as PHASES

CANDIDATE_THRESHOLD = 0.05
MIN_PEAK_DISTANCE_SECONDS = 20.0
MAX_CANDIDATES_PER_BLOCK = 160
MIN_SEGMENT_SECONDS = 5.0


@dataclass(frozen=True)
class DPConfig:
    boundary_weight: float = 0.5
    transition_weight: float = 0.1
    segment_penalty: float = 0.05
    class_weight: float = 1.0

    def validate(self) -> None:
        values = [self.boundary_weight, self.transition_weight, self.segment_penalty, self.class_weight]
        if not np.isfinite(values).all() or min(values[:3]) < 0 or self.class_weight <= 0:
            raise ValueError("DP weights must be finite, nonnegative and class_weight positive")


@dataclass(frozen=True)
class _Peak:
    point_index: int
    timestamp: float
    confidence: float


def _boundary_peaks(scores, times, *, threshold, min_distance_seconds):
    """Plateau midpoint peaks and stable confidence-first time NMS; inputs validated."""
    times = times - times[0]
    changes = np.flatnonzero(np.diff(scores) != 0.0) + 1
    starts = np.concatenate(([0], changes))
    stops = np.concatenate((changes, [len(scores)]))
    run_scores = scores[starts]
    left = np.concatenate(([-np.inf], run_scores[:-1]))
    right = np.concatenate((run_scores[1:], [-np.inf]))
    peaks = (run_scores >= threshold) & (run_scores > left) & (run_scores > right)
    points = ((starts[peaks] + stops[peaks] - 1) // 2).astype(int)
    candidates = [
        _Peak(int(point), float(times[point]), float(scores[point]))
        for point in points
    ]

    kept: list[_Peak] = []
    kept_times: list[float] = []
    for candidate in sorted(
        candidates, key=lambda item: (-item.confidence, item.timestamp, item.point_index)
    ):
        position = bisect_left(kept_times, candidate.timestamp)
        neighbors = kept_times[max(0, position - 1) : position + 1]
        if all(
            abs(candidate.timestamp - existing) >= min_distance_seconds
            for existing in neighbors
        ):
            kept_times.insert(position, candidate.timestamp)
            kept.insert(position, candidate)
    return kept


@dataclass(frozen=True)
class CandidateBlock:
    block_id: int
    start: int
    stop: int
    nodes: np.ndarray  # absolute point indices, including start and stop
    confidence: np.ndarray


@dataclass(frozen=True)
class CandidateSet:
    flight_id: str
    point_count: int
    blocks: tuple[CandidateBlock, ...]
    sha256: str  # binds time, blocks, probabilities and candidate nodes
    support_sha256: str

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(int(index) for block in self.blocks for index in block.nodes[1:-1])


@dataclass(frozen=True)
class ScoredFlight:
    flight_id: str
    candidates: CandidateSet
    support: TimeSupport
    interval_log_probability: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class DPBlockResult:
    block_id: int
    start: int
    stop: int
    selected_indices: tuple[int, ...]
    path_score: float | None
    error: str | None


@dataclass(frozen=True)
class DPResult:
    labels: np.ndarray
    selected_indices: tuple[int, ...]
    blocks: tuple[DPBlockResult, ...]
    candidate_sha256: str

    @property
    def errors(self) -> tuple[DPBlockResult, ...]:
        return tuple(block for block in self.blocks if block.error is not None)


@dataclass(frozen=True)
class LabeledFlight:
    flight_id: str
    support: TimeSupport
    truth: np.ndarray


def fit_transition_log_probability(flights: Sequence[LabeledFlight]) -> np.ndarray:
    """Laplace-smoothed counts from training truth, within complete flights only.

    Pass the current training fold only. No transitions are counted across a
    gap, uncovered point or flight boundary; all five classes retain a prior.
    """
    if not flights or any(not flight.flight_id for flight in flights):
        raise ValueError("transition fitting requires identified training flights")
    if len({flight.flight_id for flight in flights}) != len(flights):
        raise ValueError("transition training flight IDs must be unique")
    lookup = {phase: i for i, phase in enumerate(PHASES)}
    counts = np.ones((len(PHASES), len(PHASES)), dtype=float)
    for flight in flights:
        labels = np.asarray(flight.truth)
        blocks = flight.support.block_ids
        if labels.shape != blocks.shape or not np.isin(labels, PHASES).all():
            raise ValueError("transition labels must be aligned canonical five-stage classes")
        for i in range(1, len(labels)):
            if blocks[i] >= 0 and blocks[i] == blocks[i - 1] and labels[i] != labels[i - 1]:
                counts[lookup[labels[i - 1]], lookup[labels[i]]] += 1
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def _hash_array(digest: object, values: np.ndarray, dtype: str) -> None:
    array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def _support_sha256(support: TimeSupport) -> str:
    digest = hashlib.sha256()
    for values, dtype in [(support.times, "<f8"), (support.block_ids, "<i8"), (support.weights_seconds, "<f8")]:
        _hash_array(digest, values, dtype)
    return digest.hexdigest()


def _validate_candidates(candidates: CandidateSet, support: TimeSupport) -> None:
    if candidates.point_count != len(support.times) or candidates.support_sha256 != _support_sha256(support):
        raise ValueError("candidate time support does not match scores or trajectory")
    changes = np.r_[0, np.flatnonzero(np.diff(support.block_ids) != 0) + 1, len(support.times)]
    runs = [(int(support.block_ids[a]), int(a), int(b)) for a, b in zip(changes[:-1], changes[1:])
            if support.block_ids[a] >= 0]
    if [(block.block_id, block.start, block.stop) for block in candidates.blocks] != runs:
        raise ValueError("candidates must cover each continuous block exactly once")
    for block in candidates.blocks:
        nodes, confidence = block.nodes, block.confidence
        if nodes.ndim != 1 or len(nodes) < 2 or not np.issubdtype(nodes.dtype, np.integer):
            raise ValueError("candidate nodes must be an integer vector including both endpoints")
        if nodes[0] != block.start or nodes[-1] != block.stop or np.any(np.diff(nodes) <= 0):
            raise ValueError("candidate nodes must increase inside their time block")
        if confidence.shape != nodes.shape or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("candidate confidences must align with nodes and lie in [0, 1]")


def build_candidates(
    flight_id: str,
    boundary_probability: Sequence[float],
    support: TimeSupport,
    *,
    threshold: float = CANDIDATE_THRESHOLD,
    min_distance_seconds: float = MIN_PEAK_DISTANCE_SECONDS,
    max_candidates: int = MAX_CANDIDATES_PER_BLOCK,
) -> CandidateSet:
    """Extract low-threshold TCN peaks independently in every time block."""

    max_candidates = index(max_candidates)
    if not 0 < threshold < 1 or not np.isfinite(min_distance_seconds) or min_distance_seconds <= 0 or max_candidates < 1:
        raise ValueError("invalid candidate extraction settings")
    probabilities = np.asarray(boundary_probability, dtype=float)
    n = len(support.times)
    if not flight_id or probabilities.shape != (n,) or support.block_ids.shape != (n,):
        raise ValueError("flight ID, TCN probability and time support disagree")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("TCN probabilities must be finite values in [0, 1]")
    blocks: list[CandidateBlock] = []
    start = 0
    while start < n:
        stop = start + 1
        block_id = int(support.block_ids[start])
        while stop < n and support.block_ids[stop] == block_id:
            stop += 1
        if block_id >= 0:
            events = _boundary_peaks(
                probabilities[start:stop], support.times[start:stop],
                threshold=threshold,
                min_distance_seconds=min_distance_seconds,
            )
            events = [event for event in events if 0 < event.point_index < stop - start]
            if len(events) > max_candidates:
                events = sorted(
                    sorted(events, key=lambda event: (-event.confidence, event.timestamp,
                                                      event.point_index))[:max_candidates],
                    key=lambda event: event.point_index,
                )
            nodes = np.asarray([start, *(start + event.point_index for event in events), stop],
                               dtype=int)
            confidence = np.asarray([1.0, *(event.confidence for event in events), 1.0],
                                    dtype=float)
            blocks.append(CandidateBlock(block_id, start, stop, nodes, confidence))
        start = stop
    digest = hashlib.sha256()
    digest.update(b"flight-phase-candidates-v1\0")
    digest.update(flight_id.encode("utf-8"))
    _hash_array(digest, support.times, "<f8")
    _hash_array(digest, support.block_ids, "<i8")
    _hash_array(digest, support.weights_seconds, "<f8")
    _hash_array(digest, probabilities, "<f8")
    for block in blocks:
        _hash_array(digest, block.nodes, "<i8")
        _hash_array(digest, block.confidence, "<f8")
    return CandidateSet(flight_id, n, tuple(blocks), digest.hexdigest(), _support_sha256(support))


def score_candidate_intervals(
    candidates: CandidateSet,
    support: TimeSupport,
    scorer: Callable[[int, int], Sequence[float] | None],
    *, min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> tuple[np.ndarray, ...]:
    """Build per-block log_softmax tensors; impossible intervals stay -inf."""

    _validate_candidates(candidates, support)
    if not np.isfinite(min_segment_seconds) or min_segment_seconds <= 0:
        raise ValueError("minimum segment duration must be finite and positive")
    matrices: list[np.ndarray] = []
    for block in candidates.blocks:
        nodes = block.nodes
        matrix = np.full((len(nodes), len(nodes), len(PHASES)), -np.inf, dtype=float)
        for left in range(len(nodes) - 1):
            for right in range(left + 1, len(nodes)):
                start, stop = int(nodes[left]), int(nodes[right])
                duration = float(support.weights_seconds[start:stop].sum())
                if duration < min_segment_seconds:
                    continue
                raw = scorer(start, stop)
                if raw is None:
                    continue
                values = np.asarray(raw, dtype=float)
                if values.shape != (len(PHASES),):
                    raise ValueError("interval scorer must return five class scores")
                if np.isnan(values).any() or np.isposinf(values).any():
                    raise ValueError("interval scores must be finite or -inf")
                if np.isneginf(values).all():
                    continue
                matrix[left, right] = values - logsumexp(values)
        matrices.append(matrix)
    return tuple(matrices)


def scored_flight(
    candidates: CandidateSet,
    support: TimeSupport,
    scorer: Callable[[int, int], Sequence[float] | None],
    *,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> ScoredFlight:
    return ScoredFlight(candidates.flight_id, candidates, support,
                        score_candidate_intervals(candidates, support, scorer, min_segment_seconds=min_segment_seconds))


def decode_scored_flight(
    flight: ScoredFlight,
    transition_log_probability: np.ndarray,
    config: DPConfig = DPConfig(),
) -> DPResult:
    """Solve each continuous block; impossible blocks remain UNKNOWN with error."""

    config.validate()
    _validate_candidates(flight.candidates, flight.support)
    if flight.flight_id != flight.candidates.flight_id:
        raise ValueError("scored flight ID and candidates disagree")
    n = flight.candidates.point_count
    if flight.support.block_ids.shape != (n,) or len(flight.interval_log_probability) != len(flight.candidates.blocks):
        raise ValueError("scored flight block dimensions disagree")
    transitions = np.asarray(transition_log_probability, dtype=float)
    if transitions.shape != (len(PHASES), len(PHASES)) or np.isnan(transitions).any() or np.isposinf(transitions).any():
        raise ValueError("transition log probability must be five by five, finite or -inf")
    weighted_transitions = np.full_like(transitions, -np.inf)
    finite = np.isfinite(transitions)
    weighted_transitions[finite] = config.transition_weight * transitions[finite]
    labels = np.full(n, "UNKNOWN", dtype=object)
    results: list[DPBlockResult] = []
    selected: list[int] = []
    for block, matrix in zip(flight.candidates.blocks, flight.interval_log_probability):
        nodes = block.nodes
        m = len(nodes)
        if matrix.shape != (m, m, len(PHASES)):
            raise ValueError("interval score tensor dimensions disagree")
        if np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise ValueError("interval tensor must contain finite scores or -inf")
        dp = np.full((m, len(PHASES)), -np.inf, dtype=float)
        prior_node = np.full((m, len(PHASES)), -1, dtype=int)
        prior_phase = np.full((m, len(PHASES)), -1, dtype=int)
        for right in range(1, m):
            bonus = 0.0
            if right < m - 1:
                probability = float(np.clip(block.confidence[right], 1e-6, 1 - 1e-6))
                bonus = config.boundary_weight * np.log(probability / (1 - probability)) - config.segment_penalty
            for phase in range(len(PHASES)):
                for left in range(right):
                    segment_score = matrix[left, right, phase]
                    if not np.isfinite(segment_score):
                        continue
                    base = config.class_weight * segment_score + bonus
                    if left == 0:
                        value, previous = base, -1
                    else:
                        choices = dp[left] + weighted_transitions[:, phase]
                        choices = choices.copy()
                        choices[phase] = -np.inf
                        previous = int(np.argmax(choices))
                        value = float(choices[previous]) + base
                    if value > dp[right, phase]:
                        dp[right, phase] = value
                        prior_node[right, phase] = left
                        prior_phase[right, phase] = previous
        final_phase = int(np.argmax(dp[-1]))
        if not np.isfinite(dp[-1, final_phase]):
            results.append(DPBlockResult(block.block_id, block.start, block.stop,
                                         (), None, "NO_FEASIBLE_PATH"))
            continue
        path: list[int] = []
        right, phase = m - 1, final_phase
        while right:
            left = int(prior_node[right, phase])
            if left < 0:
                raise RuntimeError("DP backpointer is incomplete")
            labels[int(nodes[left]):int(nodes[right])] = PHASES[phase]
            if left:
                path.append(int(nodes[left]))
            phase = int(prior_phase[right, phase])
            right = left
        path.reverse()
        selected.extend(path)
        results.append(DPBlockResult(block.block_id, block.start, block.stop,
                                     tuple(path), float(dp[-1, final_phase]), None))
    return DPResult(labels.astype(str), tuple(selected), tuple(results),
                    flight.candidates.sha256)
