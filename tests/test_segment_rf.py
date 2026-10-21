"""Synthetic segment features, fold-local RF and explicit class-order checks."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from flight_phase.data.preprocessing.features import F1A_FEATURE_COLUMNS
from flight_phase.data.time_support import build_time_support
from flight_phase.models.segment_rf import (
    SEGMENT_FEATURE_COLUMNS, SegmentRFConfig, SegmentRandomForest, fit_segment_rf,
    segment_feature_row,
)
from flight_phase.schema import FIVE_STAGE_PHASES


def rows():
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(50, 9)), columns=SEGMENT_FEATURE_COLUMNS)
    frame.loc[0, "velocity"] = np.nan
    return frame


def test_interval_means_duration_and_gap_isolation():
    features = np.arange(84, dtype=float).reshape(12, 7)
    features[0, -1] = np.nan
    support = build_time_support(np.r_[np.arange(6), np.arange(20, 26)])
    row = segment_feature_row(features, support, 0, 6)
    np.testing.assert_allclose([row[name] for name in F1A_FEATURE_COLUMNS], np.nanmean(features[:6], axis=0))
    assert row["duration_seconds"] == 5 and row["log_duration"] == np.log1p(5)
    assert segment_feature_row(features, support, 5, 7) is None
    assert segment_feature_row(np.full((12, 7), np.nan), support, 0, 6) is None
    with pytest.raises(ValueError, match="half-open"):
        segment_feature_row(features, support, 0, 13)


def test_fold_local_fit_probability_order_and_trusted_roundtrip(tmp_path):
    train = rows()
    labels = np.tile(FIVE_STAGE_PHASES, 10)
    model = fit_segment_rf(train, labels, config=SegmentRFConfig(n_estimators=8))
    before = model.medians.copy()
    held_out = train.copy() + 10000
    probability = model.predict_proba(held_out)
    assert probability.shape == (50, 5)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    np.testing.assert_array_equal(before, model.medians)
    assert set(model.predict(train)) <= set(FIVE_STAGE_PHASES)
    model.dump(tmp_path / "rf.joblib")
    restored = SegmentRandomForest.load(tmp_path / "rf.joblib")
    np.testing.assert_array_equal(model.predict_proba(train), restored.predict_proba(train))
    with pytest.raises(FileExistsError):
        model.dump(tmp_path / "rf.joblib")
    with pytest.raises(ValueError, match="median dimensions"):
        replace(model, medians=np.zeros(7)).validate()


def test_missing_classes_keep_zero_columns_without_fabricated_samples():
    model = fit_segment_rf(rows(), ["GROUND", "DESCENT"] * 25, config=SegmentRFConfig(n_estimators=8))
    assert model.estimator.classes_.tolist() == [0, 4]
    probability = model.predict_proba(rows())
    assert not probability[:, 1:4].any()
    assert np.isneginf(model.predict_log_proba(rows())[:, 1:4]).all()


def test_training_rejects_label_mismatch_unknown_and_missing_feature():
    frame = rows()
    for labels in [["GROUND"], ["UNKNOWN"] * 50]:
        with pytest.raises(ValueError, match="aligned nonempty"):
            fit_segment_rf(frame, labels)
    frame["altitude"] = np.nan
    with pytest.raises(ValueError, match="all-missing feature"):
        fit_segment_rf(frame, ["GROUND"] * 50)
