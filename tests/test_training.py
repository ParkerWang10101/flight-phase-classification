"""Synthetic regressions for public training inputs and flight-level splits."""

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flight_phase.training import expand_labels, load_manifest, nested_folds, training_flights
from flight_phase.schema import FlightPhase, FiveStageFlightPhase


def intervals(fid="001", phases=("CLIMB", "LVL")):
    return pd.DataFrame({"flight_id": [fid, fid], "phase": phases,
                         "start_index": [0, 1], "end_index": [0, 2]})


@pytest.mark.parametrize("utc", [False, True])
def test_training_reader_aligns_canonical_points_and_never_resolves_holdouts(tmp_path, monkeypatch, utc):
    rows = []
    for fid in ("NA", "001"):
        times = [8, 0, 2]
        if utc:
            times = pd.to_datetime(times, unit="s", utc=True)
        pd.DataFrame({"flight_id": fid, "timestamp": times,
                      "altitude": [1080, 1000, 1020], "vertrate": 600,
                      "velocity": [108, 100, 102]}).to_csv(tmp_path / f"{fid}.csv", index=False)
        intervals(fid).to_csv(tmp_path / f"{fid}-labels.csv", index=False)
        rows.append([fid, "train", f"{fid}.csv", f"{fid}-labels.csv"])
    rows.extend([["S", "sealed", "../holdout-do-not-open/S.csv", ""],
                 ["T", "test", "", "../holdout-do-not-open/T.csv"]])
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=["flight_id", "split", "trajectory_path", "annotation_path"]).to_csv(
        manifest_path, index=False
    )
    original_resolve, original_open = Path.resolve, Path.open

    def guard(operation):
        def checked(path, *args, **kwargs):
            assert "holdout-do-not-open" not in path.parts
            return operation(path, *args, **kwargs)
        return checked

    monkeypatch.setattr(Path, "resolve", guard(original_resolve))
    monkeypatch.setattr(Path, "open", guard(original_open))
    assert load_manifest(manifest_path).flight_id.tolist() == ["NA", "001", "S", "T"]
    flights = list(training_flights(manifest_path, tmp_path))
    assert [item[0] for item in flights] == ["001", "NA"]
    for fid, frame, truth, hashes in flights:
        assert frame.flight_id.tolist() == [fid] * 3
        assert frame.point_index.tolist() == [0, 1, 2]
        assert frame.velocity.tolist() == [100, 102, 108]
        assert truth.tolist() == ["CLIMB", "LVL", "LVL"]
        assert frame.timestamp.is_monotonic_increasing
        for kind, filename in [("trajectory", f"{fid}.csv"), ("annotation", f"{fid}-labels.csv")]:
            assert hashes[f"{kind}_sha256"] == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()


@pytest.mark.parametrize("bad", ["duplicate_id", "split", "missing_path"])
def test_manifest_rejects_ambiguous_training_rows(tmp_path, bad):
    frame = pd.DataFrame({"flight_id": ["A", "B"], "split": ["train", "train"],
                          "trajectory_path": ["a.csv", "b.csv"], "annotation_path": ["a-label.csv", "b-label.csv"]})
    column, value = {"duplicate_id": ("flight_id", "A"), "split": ("split", "validation"),
                     "missing_path": ("annotation_path", " ")}[bad]
    frame.loc[1, column] = value
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "manifest.csv")


def test_labels_keep_four_and_five_stage_contracts_without_filling_classes():
    assert expand_labels(intervals(), "001", 3).tolist() == ["CLIMB", "LVL", "LVL"]
    four = intervals(phases=("GROUND", "CRUISE"))
    assert expand_labels(four, "001", 3, stages=4).tolist() == ["GROUND", "CRUISE", "CRUISE"]
    with pytest.raises(ValueError, match="5-stage"):
        expand_labels(four, "001", 3)
    with pytest.raises(ValueError, match="4-stage"):
        expand_labels(intervals(), "001", 3, stages=4)
    assert expand_labels(intervals(phases=(FiveStageFlightPhase.CLIMB, FiveStageFlightPhase.LEVEL)),
                         "001", 3).tolist() == ["CLIMB", "LVL", "LVL"]
    assert expand_labels(intervals(phases=(FlightPhase.GROUND, FlightPhase.CRUISE)),
                         "001", 3, stages=4).tolist() == ["GROUND", "CRUISE", "CRUISE"]


@pytest.mark.parametrize("column,row,value", [
    ("start_index", 1, 0), ("start_index", 1, 2), ("end_index", 1, 3),
    ("start_index", 1, 1.5), ("end_index", 1, np.nan),
    ("flight_id", 0, "other"), ("flight_id", 0, pd.NA),
    ("phase", 0, "UNKNOWN"), ("status", 0, "draft"), ("status", 0, pd.NA),
])
def test_labels_reject_misalignment_or_unusable_intervals(column, row, value):
    frame = intervals().astype({"start_index": float, "end_index": float})
    frame["status"] = "complete"
    frame = frame.astype({"flight_id": "string", "status": "string"})
    frame.loc[row, column] = value
    with pytest.raises(ValueError):
        expand_labels(frame, "001", 3)


@pytest.mark.parametrize("bad", ["escape", "flight_id"])
def test_training_reader_rejects_unsafe_path_or_wrong_flight(tmp_path, bad):
    outside = tmp_path / "outside.csv"
    pd.DataFrame({"flight_id": ["other"], "timestamp": [0], "altitude": [1000],
                  "vertrate": [600], "velocity": [100]}).to_csv(outside, index=False)
    intervals().to_csv(tmp_path / "labels.csv", index=False)
    root = tmp_path / "data"
    root.mkdir()
    if bad == "escape":
        (root / "flight.csv").symlink_to(outside)
    else:
        (root / "flight.csv").write_bytes(outside.read_bytes())
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"flight_id": ["001"], "split": ["train"],
                  "trajectory_path": ["flight.csv"], "annotation_path": ["labels.csv"]}).to_csv(manifest, index=False)
    (root / "labels.csv").write_bytes((tmp_path / "labels.csv").read_bytes())
    with pytest.raises(ValueError, match="escapes data root" if bad == "escape" else "trajectory flight_id mismatch"):
        list(training_flights(manifest, root))


def test_nested_folds_are_reproducible_complete_flight_partitions():
    ids = [f"F{i:02}" for i in range(7)]
    folds = nested_folds(ids, outer=3, inner=4)
    assert folds == nested_folds(ids[::-1], outer=3, inner=4)
    assert folds != nested_folds(ids, outer=3, inner=4, seed=43)
    assert Counter(fid for fold in folds for fid in fold["validation"]) == Counter(ids)
    for i, fold in enumerate(folds):
        train, valid = set(fold["train"]), set(fold["validation"])
        assert fold["fold"] == i and not train & valid and train | valid == set(ids)
        assert Counter(fid for part in fold["inner"] for fid in part["validation"]) == Counter(train)
        for j, part in enumerate(fold["inner"]):
            assert part["fold"] == j
            assert set(part["train"]) | set(part["validation"]) == train
            assert not set(part["train"]) & set(part["validation"])


def test_nested_folds_require_only_feasible_unique_flights():
    assert len(nested_folds(["A", "B", "C"], outer=3, inner=2)) == 3
    for ids, outer, inner in [(["A", "A", "B"], 3, 2), (["A", "B"], 3, 2),
                              (["A", "B", "C"], 2, 2)]:
        with pytest.raises(ValueError):
            nested_folds(ids, outer=outer, inner=inner)
