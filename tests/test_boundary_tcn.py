"""Synthetic boundary-network, gridding and portable-checkpoint regressions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flight_phase.models.boundary_tcn import (
    BOUNDARY_INPUT_COLUMNS, BoundaryTCN, BoundaryTCNBundle, BoundaryTCNConfig,
    RobustFeatureScaler, prepare_boundary_sequences,
)
from flight_phase.training.boundary import gaussian_boundary_target, sigmoid_focal_loss


def trajectory(fid="A", times=None):
    times = np.arange(40) if times is None else np.asarray(times)
    return pd.DataFrame({"flight_id": fid, "timestamp": times, "altitude": times * 100.,
                         "vertrate": 600., "velocity": times + 100.})


@pytest.mark.parametrize("mode", ["causal", "offline"])
def test_network_length_context_and_receptive_field(mode):
    torch.manual_seed(42)
    config = BoundaryTCNConfig(channels=4, kernel_size=3, dilations=(1, 2), dropout=0, mode=mode)
    model = BoundaryTCN(config).eval()
    inputs = torch.randn(2, len(BOUNDARY_INPUT_COLUMNS), 30)
    with torch.inference_mode():
        full, prefix = model(inputs), model(inputs[:, :, :20])
    assert full.shape == (2, 30) and config.receptive_field == 13
    assert config.lookahead_seconds == (0 if mode == "causal" else 6)
    if mode == "causal":
        torch.testing.assert_close(full[:, :20], prefix)
    else:
        torch.testing.assert_close(full[:, :14], prefix[:, :14])


def test_multiscale_auxiliary_head_respects_count_mask():
    torch.manual_seed(42)
    model = BoundaryTCN(BoundaryTCNConfig(channels=4, kernel_size=3, dilations=(1, 2),
                                        local_channels=2, local_dilations=(1,), count_classes=3, dropout=0))
    logits, counts = model.forward_with_count(torch.randn(2, 9, 20), torch.zeros(2, 20))
    assert logits.shape == (2, 20) and counts.shape == (2, 3)
    torch.testing.assert_close(counts, model.count_projection.bias.expand(2, -1))


def test_gridding_preserves_flight_gap_and_source_alignment():
    one = trajectory(times=[0, 1, 3, 10, 11])
    two = trajectory("B", [0, 1, 3, 10, 11])
    two["velocity"] += 200
    combined = prepare_boundary_sequences(pd.concat([two, one]).sample(frac=1, random_state=42))
    assert combined.groupby("sequence_id").size().tolist() == [4, 2, 4, 2]
    for fid, raw in [("A", one), ("B", two)]:
        actual = combined.loc[combined.flight_id.eq(fid)].reset_index(drop=True)
        separate = prepare_boundary_sequences(raw)
        pd.testing.assert_frame_equal(actual.drop(columns="sequence_id"), separate.drop(columns="sequence_id"))
        inserted = actual.loc[actual.timestamp.eq(2)].iloc[0]
        assert inserted.observation_present == 0 and inserted.observation_age_seconds == 1
        assert pd.isna(inserted.source_point_index)
        assert actual.loc[actual.timestamp.eq(10), "source_point_index"].item() == 3
    assert len(prepare_boundary_sequences(one, max_forward_fill_seconds=0)) == len(one)
    missing = trajectory(times=np.arange(10))
    missing.loc[1:6, ["altitude", "vertrate", "velocity"]] = np.nan
    split = prepare_boundary_sequences(missing)
    assert split.groupby("sequence_id").size().tolist() == [6, 3]
    restart = split.loc[split.timestamp.eq(7)].iloc[0]
    assert restart.velocity_10s_mean == 107 and np.isnan(restart.velocity_d1_10s_mean)


def test_targets_training_update_scaler_and_bundle_roundtrip(tmp_path):
    torch.manual_seed(42)
    raw = trajectory(times=np.r_[np.arange(20), np.arange(50, 70)])
    prepared = prepare_boundary_sequences(raw)
    scaler = RobustFeatureScaler.fit([prepared])
    matrix = scaler.transform(prepared)
    assert matrix.shape == (40, 9) and np.isfinite(matrix).all()
    assert np.abs(matrix).max() <= 8
    blocks = np.repeat([0, 1], 20)
    target = gaussian_boundary_target(raw.timestamp, [10, 30], blocks)
    assert target[10] == target[30] == 1 and target[19] == target[20] == 0
    model = BoundaryTCN(BoundaryTCNConfig(channels=4, kernel_size=3, dilations=(1,), dropout=0, mode="offline"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = model.output_projection.weight.detach().clone()
    inputs = torch.from_numpy(matrix[:20].T[None])
    logits = model(inputs)
    mask = torch.ones_like(logits)
    mask[:, :2] = 0
    loss = sigmoid_focal_loss(logits, torch.from_numpy(target[:20][None]), mask)
    loss.backward()
    optimizer.step()
    assert np.isfinite(loss.item()) and not torch.equal(before, model.output_projection.weight)
    bundle = BoundaryTCNBundle(model.config, scaler, model.state_dict(), 0.5, 20, {"seed": 42})
    expected = bundle.predict_scores(raw)
    restored = BoundaryTCNBundle.load(bundle.dump(tmp_path / "bundle"))
    pd.testing.assert_frame_equal(expected, restored.predict_scores(raw))
    assert len(expected) == 40 and expected.boundary_probability.between(0, 1).all()
    with pytest.raises(FileExistsError):
        bundle.dump(tmp_path / "bundle")
    with pytest.raises(ValueError, match="feature order"):
        replace(bundle, scaler=replace(scaler, columns=scaler.columns[::-1])).validate()
    checkpoint = tmp_path / "bundle/model.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        BoundaryTCNBundle.load(tmp_path / "bundle")
