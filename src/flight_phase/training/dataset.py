"""Explicit training manifests, aligned interval labels and nested flight folds."""

from collections.abc import Iterator
import hashlib
from operator import index
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from flight_phase.data.io import load_trajectory
from flight_phase.schema import FIVE_STAGE_PHASES, FOUR_STAGE_PHASES


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read flight_id, split, trajectory_path and annotation_path from CSV.

    Splits are train, sealed or test. IDs are unique, explicit strings. Only
    train rows require paths; this function never resolves or opens those paths.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"flight_id", "split", "trajectory_path", "annotation_path"}
    if frame.empty or not required.issubset(frame):
        raise ValueError(f"a nonempty manifest requires {sorted(required)}")
    if frame.flight_id.str.strip().eq("").any() or frame.flight_id.duplicated().any():
        raise ValueError("manifest flight IDs must be nonempty and unique")
    if not frame.split.isin(["train", "sealed", "test"]).all():
        raise ValueError("split must be train, sealed or test")
    train = frame.loc[frame.split.eq("train")]
    if train.empty:
        raise ValueError("manifest has no training flights")
    if train[["trajectory_path", "annotation_path"]].apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ValueError("training rows require trajectory and annotation paths")
    return frame


def expand_labels(
    annotation: pd.DataFrame,
    flight_id: str,
    point_count: int,
    *,
    stages: Literal[4, 5] = 5,
) -> np.ndarray:
    """Expand ordered, inclusive intervals over canonical point_index positions.

    All points must be covered exactly once. Select four or five stages
    explicitly; UNKNOWN is never a training class. If status is supplied, every
    interval must be complete. No sorting, phase conversion or class filling
    is performed, and a flight need not contain every class.
    """
    point_count = index(point_count)
    if point_count < 1:
        raise ValueError("point_count must be positive")
    if stages not in (4, 5):
        raise ValueError("stages must be 4 or 5")
    required = {"flight_id", "phase", "start_index", "end_index"}
    if annotation.empty or not required.issubset(annotation):
        raise ValueError(f"nonempty labels require {sorted(required)}")
    if not annotation.flight_id.eq(flight_id).fillna(False).all():
        raise ValueError("annotation flight_id mismatch")
    phases = FOUR_STAGE_PHASES if stages == 4 else FIVE_STAGE_PHASES
    if not annotation.phase.isin(phases).all():
        raise ValueError(f"expected {stages}-stage labels from {phases}")
    if "status" in annotation and not annotation.status.eq("complete").fillna(False).all():
        raise ValueError("only complete annotations can be used for training")
    bounds = annotation[["start_index", "end_index"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(bounds).all() or np.any(bounds != np.floor(bounds)):
        raise ValueError("annotation indices must be finite integers")
    starts, ends = bounds.T
    if (
        starts[0] != 0 or ends[-1] != point_count - 1
        or np.any(starts > ends)
        or not np.array_equal(starts[1:], ends[:-1] + 1)
    ):
        raise ValueError("annotation intervals must cover every point exactly once")
    labels = np.empty(point_count, dtype="U16")
    for phase, (start, end) in zip(annotation.phase, bounds.astype(int), strict=True):
        # Select the canonical string, including for public str Enum inputs.
        labels[start:end + 1] = phases[phases.index(phase)]
    return labels


def _resolve(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"manifest path escapes data root: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_flights(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    stages: Literal[4, 5] = 5,
) -> Iterator[tuple[str, pd.DataFrame, np.ndarray, dict[str, str]]]:
    """Yield (flight_id, trajectory, labels, source_hashes) for train rows only.

    Paths resolve inside data_root. Each file contains one explicitly identified
    canonical flight. Annotation indices refer to the sorted, zero-based
    point_index returned by load_trajectory, not raw CSV row order. Sealed/test
    paths are neither resolved nor opened. No features or fitted state are built.
    """
    manifest = load_manifest(manifest_path)
    root = Path(data_root).resolve()
    train = manifest.loc[manifest.split.eq("train")].sort_values("flight_id")
    for row in train.itertuples(index=False):
        trajectory_path = _resolve(root, row.trajectory_path)
        label_path = _resolve(root, row.annotation_path)
        frame = load_trajectory(trajectory_path)
        if frame.flight_id.iloc[0] != row.flight_id:
            raise ValueError(f"{row.flight_id}: trajectory flight_id mismatch")
        annotation = pd.read_csv(label_path, dtype=str, keep_default_na=False)
        truth = expand_labels(annotation, row.flight_id, len(frame), stages=stages)
        yield row.flight_id, frame, truth, {
            "trajectory_sha256": _sha256(trajectory_path),
            "annotation_sha256": _sha256(label_path),
        }


def _kfold(size: int, count: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Match shuffled sklearn KFold membership and sorted output indices."""
    shuffled = np.arange(size)
    np.random.RandomState(seed).shuffle(shuffled)
    for selected in np.array_split(shuffled, count):
        validation = np.zeros(size, dtype=bool)
        validation[selected] = True
        yield np.flatnonzero(~validation), np.flatnonzero(validation)


def nested_folds(
    flight_ids: list[str], *, outer: int = 5, inner: int = 4, seed: int = 42,
) -> list[dict]:
    """Split development IDs into nested, shuffled folds of complete flights.

    IDs are sorted before shuffling; outer seeds are seed and inner seeds are
    seed + outer_fold + 1. Fold numbers start at zero. Only actual split sizes
    are required. Each inner split uses its outer training flights exclusively.
    Callers supply train IDs only and fit every learned transform within each
    training fold; this utility neither reads data nor fits transforms.
    """
    outer, inner = index(outer), index(inner)
    if any(not isinstance(fid, str) or not fid.strip() for fid in flight_ids):
        raise ValueError("flight IDs must be nonempty strings")
    ids = np.asarray(sorted(flight_ids), dtype=str)
    if len(ids) != len(set(ids)):
        raise ValueError("flight IDs must be unique")
    if not 2 <= outer <= len(ids) or inner < 2:
        raise ValueError("need at least two folds and one flight per outer fold")
    if len(ids) - (len(ids) + outer - 1) // outer < inner:
        raise ValueError("not enough outer training flights for the inner folds")
    folds = []
    for fold, (train, valid) in enumerate(_kfold(len(ids), outer, seed)):
        train_ids = ids[train]
        parts = [
            {"fold": j, "train": train_ids[a].tolist(), "validation": train_ids[b].tolist()}
            for j, (a, b) in enumerate(_kfold(len(train_ids), inner, seed + fold + 1))
        ]
        folds.append({"fold": fold, "train": train_ids.tolist(),
                      "validation": ids[valid].tolist(), "inner": parts})
    return folds
