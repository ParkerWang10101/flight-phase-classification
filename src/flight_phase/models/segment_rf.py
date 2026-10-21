"""Five-stage segment random forest with fold-local preprocessing."""

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from flight_phase.data.preprocessing.features import F1A_FEATURE_COLUMNS
from flight_phase.data.time_support import TimeSupport
from flight_phase.schema import FIVE_STAGE_PHASES

SEGMENT_FEATURE_COLUMNS = (*F1A_FEATURE_COLUMNS, "duration_seconds", "log_duration")


def segment_feature_row(features: np.ndarray, support: TimeSupport, start: int, stop: int) -> dict | None:
    """Aggregate one half-open [start, stop) interval of a single flight.

    features is a point matrix in F1A_FEATURE_COLUMNS order, aligned to support.
    Means omit NaNs; duration sums right-continuous support weights, excluding
    uncovered gaps and the final point's zero support. An uncovered, cross-block
    or wholly missing interval has no score (None). No labels are accessed.
    """
    values = np.asarray(features, dtype=float)
    if values.shape != (len(support.times), len(F1A_FEATURE_COLUMNS)):
        raise ValueError("expected one flight's aligned seven-column F1A matrix")
    if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)) or not 0 <= start < stop <= len(values):
        raise ValueError("segment bounds must be nonempty half-open point indices")
    blocks = support.block_ids[start:stop]
    if blocks[0] < 0 or np.any(blocks != blocks[0]):
        return None
    interval = values[start:stop]
    valid = np.isfinite(interval).any(axis=1)
    if not valid.any():
        return None
    # Avoid all-NaN warnings while retaining the original per-column nanmean.
    usable = np.where(valid[:, None], interval, np.nan)
    present = ~np.isnan(usable).all(axis=0)
    mean = np.full(len(F1A_FEATURE_COLUMNS), np.nan)
    mean[present] = np.nanmean(usable[:, present], axis=0)
    duration = float(support.weights_seconds[start:stop].sum())
    return {"start_index": int(start), "stop_index": int(stop),
            **dict(zip(F1A_FEATURE_COLUMNS, mean)), "duration_seconds": duration,
            "log_duration": float(np.log1p(max(duration, 0.0)))}


@dataclass(frozen=True)
class SegmentRFConfig:
    n_estimators: int = 400
    max_depth: int = 8
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "balanced_subsample"
    n_jobs: int = 1
    random_state: int = 42


@dataclass
class SegmentRandomForest:
    """Explicit fitted state; predict columns always follow FIVE_STAGE_PHASES."""

    medians: np.ndarray
    scaler: StandardScaler
    estimator: RandomForestClassifier

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        raw = rows.loc[:, SEGMENT_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return self.scaler.transform(np.where(np.isfinite(raw), raw, self.medians))

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        probability = np.zeros((len(rows), len(FIVE_STAGE_PHASES)))
        probability[:, self.estimator.classes_] = self.estimator.predict_proba(self.transform(rows))
        return probability

    def predict_log_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Clipped RF log scores; absent training classes remain -inf."""
        scores = np.full((len(rows), len(FIVE_STAGE_PHASES)), -np.inf)
        scores[:, self.estimator.classes_] = np.log(np.maximum(
            self.estimator.predict_proba(self.transform(rows)), 1e-300,
        ))
        return scores

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(FIVE_STAGE_PHASES)[self.predict_proba(rows).argmax(axis=1)]

    def validate(self) -> None:
        size = len(SEGMENT_FEATURE_COLUMNS)
        if np.asarray(self.medians).shape != (size,) or not np.isfinite(self.medians).all():
            raise ValueError("RF median dimensions or values do not match the features")
        if self.scaler.n_features_in_ != size or self.estimator.n_features_in_ != size:
            raise ValueError("RF estimator and scaler feature dimensions differ")
        classes = self.estimator.classes_
        if not np.issubdtype(classes.dtype, np.integer) or not np.isin(classes, np.arange(5)).all():
            raise ValueError("RF classes must be canonical five-stage indices")
        if self.scaler.mean_.shape != (size,) or self.scaler.scale_.shape != (size,):
            raise ValueError("RF scaler statistics must match the feature order")
        if not np.isfinite(self.scaler.mean_).all() or not np.isfinite(self.scaler.scale_).all() or np.any(self.scaler.scale_ <= 0):
            raise ValueError("RF scaler statistics must be finite with positive scales")

    def dump(self, path: str | Path) -> Path:
        """Save fitted state without overwriting an existing local artifact."""
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            joblib.dump({"format": "flight-phase-segment-rf-v1", "model": self,
                         "feature_columns": SEGMENT_FEATURE_COLUMNS, "phase_order": FIVE_STAGE_PHASES}, stream)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SegmentRandomForest":
        """Load only a trusted, locally trained joblib artifact (pickle format)."""
        payload = joblib.load(path)
        if payload["format"] != "flight-phase-segment-rf-v1" or payload["feature_columns"] != SEGMENT_FEATURE_COLUMNS or payload["phase_order"] != FIVE_STAGE_PHASES:
            raise ValueError("RF artifact feature or phase metadata mismatch")
        model = payload["model"]
        if not isinstance(model, cls):
            raise TypeError("RF artifact does not contain a SegmentRandomForest")
        model.validate()
        return model


def fit_segment_rf(rows: pd.DataFrame, labels, *, config: SegmentRFConfig = SegmentRFConfig()) -> SegmentRandomForest:
    """Fit medians, StandardScaler and RF on this training fold only.

    Rows contain the nine documented segment features; labels are explicit
    five-stage strings. No sample is copied to fill missing classes. Predict
    exposes absent classes as zero probability; a five-class decoder can require
    all classes before requesting finite interval log scores.
    """
    truth = pd.Series(labels)
    if rows.empty or len(truth) != len(rows) or not truth.isin(FIVE_STAGE_PHASES).all():
        raise ValueError("RF training requires aligned nonempty five-stage labels")
    raw = rows.loc[:, SEGMENT_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(raw).any(axis=0).all():
        raise ValueError("training segment matrix contains an all-missing feature")
    medians = np.nanmedian(raw, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(raw), raw, medians)
    scaler = StandardScaler().fit(filled)
    encoded = np.asarray([FIVE_STAGE_PHASES.index(label) for label in truth], dtype=int)
    estimator = RandomForestClassifier(**asdict(config)).fit(scaler.transform(filled), encoded)
    return SegmentRandomForest(medians, scaler, estimator)
