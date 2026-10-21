"""Synthetic candidate, exact-optimum and composed-pipeline regressions."""

from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from flight_phase.data.time_support import build_time_support
from flight_phase.decoding.bgdp import (
    DPConfig, LabeledFlight, build_candidates, decode_scored_flight,
    fit_transition_log_probability, score_candidate_intervals, scored_flight,
)
from flight_phase.schema import FIVE_STAGE_PHASES


def test_candidates_plateaus_stable_cap_and_time_support_binding():
    scores = np.zeros(30)
    scores[4:6], scores[10], scores[20] = .8, .9, .9
    support = build_time_support(np.arange(30))
    candidates = build_candidates("A", scores, support, min_distance_seconds=5, max_candidates=2)
    assert candidates.indices == (10, 20)
    altered = scores.copy(); altered[2] = .01
    assert build_candidates("A", altered, support, min_distance_seconds=5, max_candidates=2).sha256 != candidates.sha256
    times = np.arange(30, dtype=float); times[12] += .1
    with pytest.raises(ValueError, match="time support"):
        score_candidate_intervals(candidates, build_time_support(times), lambda a, b: np.zeros(5))


def test_dp_matches_exhaustive_path_score_and_boundaries():
    support = build_time_support(np.arange(21))
    probabilities = np.zeros(21); probabilities[[5, 10, 15]] = [.8, .7, .9]
    candidates = build_candidates("A", probabilities, support, min_distance_seconds=1)
    nodes = candidates.blocks[0].nodes
    rng = np.random.default_rng(42)
    raw = {(int(a), int(b)): rng.normal(size=5) for i, a in enumerate(nodes[:-1]) for b in nodes[i + 1:]}
    scored = scored_flight(candidates, support, lambda a, b: raw[a, b])
    transitions = rng.uniform(.1, 1, (5, 5)); transitions = np.log(transitions / transitions.sum(axis=1, keepdims=True))
    config = DPConfig(.5, .1, .05)
    actual = decode_scored_flight(scored, transitions, config)
    best = -np.inf
    for count in range(len(nodes) - 1):
        for internal in combinations(range(1, len(nodes) - 1), count):
            path = (0, *internal, len(nodes) - 1)
            for phases in product(range(5), repeat=len(path) - 1):
                if any(a == b for a, b in zip(phases[:-1], phases[1:])):
                    continue
                value = sum(scored.interval_log_probability[0][a, b, p] for a, b, p in zip(path[:-1], path[1:], phases))
                value += sum(config.transition_weight * transitions[a, b] for a, b in zip(phases[:-1], phases[1:]))
                value += sum(config.boundary_weight * np.log(probabilities[nodes[i]] / (1 - probabilities[nodes[i]])) - config.segment_penalty for i in internal)
                if value > best:
                    best, expected_indices = value, tuple(int(nodes[i]) for i in internal)
                    expected = np.empty(21, dtype="U7")
                    for a, b, p in zip(path[:-1], path[1:], phases):
                        expected[nodes[a]:nodes[b]] = FIVE_STAGE_PHASES[p]
    assert actual.blocks[0].path_score == pytest.approx(best, abs=1e-12)
    assert actual.selected_indices == expected_indices
    np.testing.assert_array_equal(actual.labels, expected)


def test_dp_stays_inside_gaps_and_reports_impossible_blocks():
    support = build_time_support(np.r_[np.arange(12), np.arange(30, 42)])
    probability = np.zeros(24); probability[[6, 18]] = .9
    candidates = build_candidates("A", probability, support)
    calls = []
    def score(start, stop):
        calls.append((start, stop))
        values = np.full(5, -8.)
        values[0 if start in (0, 12) else 1] = 8.
        return np.zeros(5) if (start, stop) in ((0, 12), (12, 24)) else values
    result = decode_scored_flight(scored_flight(candidates, support, score), np.log(np.full((5, 5), .2)))
    assert all(stop <= 12 or start >= 12 for start, stop in calls)
    assert result.selected_indices == (6, 18) and not result.errors
    assert result.labels.tolist() == (["GROUND"] * 6 + ["CLIMB"] * 6) * 2
    impossible = decode_scored_flight(scored_flight(candidates, support, lambda a, b: None), np.log(np.full((5, 5), .2)))
    assert np.all(impossible.labels == "UNKNOWN")
    assert [item.error for item in impossible.errors] == ["NO_FEASIBLE_PATH"] * 2
    with pytest.raises(ValueError, match="finite or -inf"):
        scored_flight(candidates, support, lambda a, b: [np.nan] * 5)


def test_zero_transition_weight_still_respects_forbidden_exits():
    support = build_time_support(np.arange(12))
    probability = np.zeros(12); probability[6] = .9
    candidates = build_candidates("A", probability, support)
    def score(start, stop):
        values = np.full(5, -np.inf)
        if (start, stop) == (0, 6): values[0] = 0
        if (start, stop) == (6, 12): values[1] = 0
        return values
    flight = scored_flight(candidates, support, score)
    transitions = np.full((5, 5), -np.inf)
    assert decode_scored_flight(flight, transitions, DPConfig(transition_weight=0)).errors
    transitions[0, 1] = 0
    assert not decode_scored_flight(flight, transitions, DPConfig(transition_weight=0)).errors


def test_transition_fitting_uses_only_changes_inside_each_training_flight():
    support = build_time_support([0, 1, 2, 20, 21, 22])
    flight = LabeledFlight("train", support, np.array(["GROUND", "CLIMB", "CLIMB", "DESCENT", "LVL", "CR"]))
    matrix = np.exp(fit_transition_log_probability([flight]))
    assert matrix[0, 1] == pytest.approx(2 / 6)
    assert matrix[1, 4] == pytest.approx(1 / 5)
    with pytest.raises(ValueError, match="unique"):
        fit_transition_log_probability([flight, flight])


def test_label_free_tcn_rf_bgdp_composition_and_score_alignment():
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from flight_phase.models.boundary_tcn import BoundaryTCN, BoundaryTCNBundle, BoundaryTCNConfig, RobustFeatureScaler, prepare_boundary_sequences
    from flight_phase.models.segment_rf import SEGMENT_FEATURE_COLUMNS, SegmentRFConfig, fit_segment_rf
    from flight_phase.models.segment_pipeline import decode_rf_segments
    torch.manual_seed(42)
    raw = pd.DataFrame({"flight_id": "A", "timestamp": np.arange(60), "altitude": np.arange(60) * 100.,
                        "vertrate": 600., "velocity": np.arange(60) + 100.})
    grid = prepare_boundary_sequences(raw)
    neural = BoundaryTCN(BoundaryTCNConfig(channels=4, kernel_size=3, dilations=(1,), dropout=0, mode="offline"))
    bundle = BoundaryTCNBundle(neural.config, RobustFeatureScaler.fit([grid]), neural.state_dict(), .5, 20, {"seed": 42})
    probabilities = bundle.predict_scores(raw).boundary_probability.to_numpy()
    train = pd.DataFrame(np.random.default_rng(42).normal(size=(50, 9)), columns=SEGMENT_FEATURE_COLUMNS)
    rf = fit_segment_rf(train, np.tile(FIVE_STAGE_PHASES, 10), config=SegmentRFConfig(n_estimators=8))
    result = decode_rf_segments(grid, probabilities, rf, np.log(np.full((5, 5), .2)), block_ids=grid.sequence_id)
    assert len(result.labels) == len(grid) and not result.errors
    assert set(result.labels) <= set(FIVE_STAGE_PHASES) and "truth_phase" not in grid
    with pytest.raises(ValueError, match="disagree"):
        decode_rf_segments(grid, probabilities[:-1], rf, np.log(np.full((5, 5), .2)))
