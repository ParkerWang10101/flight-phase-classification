"""Batch RF interval scores followed by BGDP, without truth or fitted-state changes."""

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from flight_phase.data.preprocessing.features import F1A_FEATURE_COLUMNS
from flight_phase.data.time_support import build_time_support
from flight_phase.decoding.bgdp import DPConfig, ScoredFlight, build_candidates, decode_scored_flight
from flight_phase.models.segment_rf import SegmentRandomForest, segment_feature_row


def decode_rf_segments(
    frame: pd.DataFrame,
    boundary_probability,
    classifier: SegmentRandomForest,
    transition_log_probability: np.ndarray,
    *,
    config: DPConfig = DPConfig(),
    block_ids=None,
    coverage_mask=None,
    gap_threshold_seconds=5.0,
    candidate_threshold=0.05,
    candidate_min_distance_seconds=20.0,
    max_candidates=160,
    min_segment_seconds=5.0,
):
    """Decode one already-featured flight with point-aligned boundary scores.

    frame requires flight_id, timestamp and F1A_FEATURE_COLUMNS in canonical
    row order. Supply the same rows used for boundary inference; 1 Hz TCN grid
    probabilities must be paired with that grid, not raw/resampled row numbers.
    If using prepare_boundary_sequences, pass its sequence_id as block_ids to
    preserve every network-context reset. No alignment or feature set is guessed.
    """
    if frame.empty or frame.flight_id.isna().any() or frame.flight_id.nunique() != 1:
        raise ValueError("segment decoding requires exactly one identified flight")
    classifier.validate()
    if not np.isfinite(min_segment_seconds) or min_segment_seconds <= 0:
        raise ValueError("minimum segment duration must be finite and positive")
    support = build_time_support(frame.timestamp, block_ids=block_ids, coverage_mask=coverage_mask,
                                 gap_threshold_seconds=gap_threshold_seconds)
    candidates = build_candidates(str(frame.flight_id.iloc[0]), boundary_probability, support,
                                  threshold=candidate_threshold, min_distance_seconds=candidate_min_distance_seconds,
                                  max_candidates=max_candidates)
    features = frame.loc[:, F1A_FEATURE_COLUMNS].to_numpy(float)
    tensors = []
    for block in candidates.blocks:
        nodes = block.nodes
        tensor = np.full((len(nodes), len(nodes), 5), -np.inf)
        intervals, rows = [], []
        for left in range(len(nodes) - 1):
            for right in range(left + 1, len(nodes)):
                start, stop = int(nodes[left]), int(nodes[right])
                if support.weights_seconds[start:stop].sum() < min_segment_seconds:
                    continue
                row = segment_feature_row(features, support, start, stop)
                if row is not None:
                    rows.append(row)
                    intervals.append((left, right))
        if rows:
            scores = classifier.predict_log_proba(pd.DataFrame(rows))
            scores -= logsumexp(scores, axis=1, keepdims=True)
            for (left, right), values in zip(intervals, scores, strict=True):
                tensor[left, right] = values
        tensors.append(tensor)
    scored = ScoredFlight(str(frame.flight_id.iloc[0]), candidates, support, tuple(tensors))
    return decode_scored_flight(scored, transition_log_probability, config)
