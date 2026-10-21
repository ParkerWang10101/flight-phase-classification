"""ADS-B 五阶段边界的纯二元 TCN 模型与冻结 bundle。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover - 仅在未安装训练依赖时触发
    raise ImportError(
        "边界 TCN 需要 PyTorch；请安装项目的 tcn 可选依赖（pip install '.[tcn]'）"
    ) from error

from flight_phase.data.time import elapsed_seconds
from flight_phase.data.trajectory import normalize_trajectory
from flight_phase.data.preprocessing.features import F1A_FEATURE_COLUMNS, build_f1a_features
from flight_phase.schema import BASE_FEATURE_COLUMNS


BoundaryMode = Literal["causal", "offline"]
QUALITY_COLUMNS = ("observation_present", "observation_age_seconds")
BOUNDARY_INPUT_COLUMNS = (*F1A_FEATURE_COLUMNS, *QUALITY_COLUMNS)


@dataclass(frozen=True)
class BoundaryTCNConfig:
    channels: int = 64
    kernel_size: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    layers_per_block: int = 2
    dropout: float = 0.1
    mode: BoundaryMode = "causal"
    local_channels: int = 0
    local_dilations: tuple[int, ...] = ()
    count_classes: int = 0

    def validate(self) -> None:
        if self.channels < 1:
            raise ValueError("channels 必须为正整数")
        if self.kernel_size < 2 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为不小于 3 的奇数")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations 必须包含正整数")
        if self.layers_per_block < 1:
            raise ValueError("layers_per_block 必须为正整数")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1)")
        if self.mode not in {"causal", "offline"}:
            raise ValueError("mode 必须是 causal 或 offline")
        if self.local_channels < 0:
            raise ValueError("local_channels 不能为负数")
        if (self.local_channels == 0) != (len(self.local_dilations) == 0):
            raise ValueError("local_channels 与 local_dilations 必须同时启用或关闭")
        if any(value < 1 for value in self.local_dilations):
            raise ValueError("local_dilations 必须包含正整数")
        if self.count_classes not in {0, 3}:
            raise ValueError("count_classes 只能为 0 或 3（0/1/2+）")

    def _field_for(self, dilations: Sequence[int]) -> int:
        return 1 + self.layers_per_block * (self.kernel_size - 1) * sum(dilations)

    @property
    def global_receptive_field(self) -> int:
        return self._field_for(self.dilations)

    @property
    def local_receptive_field(self) -> int:
        return self._field_for(self.local_dilations) if self.local_dilations else 1

    @property
    def receptive_field(self) -> int:
        return max(self.global_receptive_field, self.local_receptive_field)

    @property
    def lookahead_seconds(self) -> int:
        return 0 if self.mode == "causal" else (self.receptive_field - 1) // 2


@dataclass(frozen=True)
class RobustFeatureScaler:
    columns: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float = 8.0

    @classmethod
    def fit(
        cls,
        frames: Iterable[pd.DataFrame],
        *,
        columns: Sequence[str] = BOUNDARY_INPUT_COLUMNS,
        clip: float = 8.0,
    ) -> "RobustFeatureScaler":
        names = tuple(columns)
        matrices = []
        for frame in frames:
            missing = [column for column in names if column not in frame]
            if missing:
                raise ValueError("缩放输入缺少字段: " + ", ".join(missing))
            matrices.append(
                frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            )
        if not matrices:
            raise ValueError("缩放器训练集不能为空")
        values = np.concatenate(matrices, axis=0)
        center = np.nanmedian(values, axis=0)
        q25 = np.nanquantile(values, 0.25, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        scale = q75 - q25
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        center[~np.isfinite(center)] = 0.0
        return cls(
            columns=names,
            center=tuple(map(float, center)),
            scale=tuple(map(float, scale)),
            clip=float(clip),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [column for column in self.columns if column not in frame]
        if missing:
            raise ValueError("推理输入缺少字段: " + ", ".join(missing))
        values = frame.loc[:, self.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        center = np.asarray(self.center, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        transformed = (values - center) / scale
        transformed[~np.isfinite(transformed)] = 0.0
        return np.clip(transformed, -self.clip, self.clip).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RobustFeatureScaler":
        return cls(
            columns=tuple(map(str, value["columns"])),
            center=tuple(map(float, value["center"])),
            scale=tuple(map(float, value["scale"])),
            clip=float(value["clip"]),
        )


class _TemporalConvolution(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        mode: BoundaryMode,
    ) -> None:
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.mode = mode
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, values: Tensor) -> Tensor:
        if self.mode == "causal":
            padded = functional.pad(values, (self.padding, 0))
        else:
            left = self.padding // 2
            padded = functional.pad(values, (left, self.padding - left))
        return self.convolution(padded)


class _TimewiseLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: Tensor) -> Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class _ResidualBlock(nn.Module):
    def __init__(self, config: BoundaryTCNConfig, dilation: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(config.layers_per_block):
            layers.extend(
                [
                    _TemporalConvolution(
                        config.channels,
                        kernel_size=config.kernel_size,
                        dilation=dilation,
                        mode=config.mode,
                    ),
                    _TimewiseLayerNorm(config.channels),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.network = nn.Sequential(*layers)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.network(values)


class BoundaryTCN(nn.Module):
    """保持输入长度不变的二元边界检测 TCN。"""

    def __init__(
        self,
        config: BoundaryTCNConfig,
        *,
        input_channels: int = len(BOUNDARY_INPUT_COLUMNS),
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.input_channels = int(input_channels)
        self.input_projection = nn.Conv1d(self.input_channels, config.channels, 1)
        self.blocks = nn.Sequential(
            *[_ResidualBlock(config, dilation) for dilation in config.dilations]
        )
        self.local_input_projection: nn.Module | None = None
        self.local_blocks: nn.Module | None = None
        self.fusion: nn.Module | None = None
        if config.local_channels:
            local_config = BoundaryTCNConfig(
                channels=config.local_channels,
                kernel_size=config.kernel_size,
                dilations=config.local_dilations,
                layers_per_block=config.layers_per_block,
                dropout=config.dropout,
                mode=config.mode,
            )
            self.local_input_projection = nn.Conv1d(
                self.input_channels, config.local_channels, 1
            )
            self.local_blocks = nn.Sequential(
                *[
                    _ResidualBlock(local_config, dilation)
                    for dilation in config.local_dilations
                ]
            )
            self.fusion = nn.Sequential(
                nn.Conv1d(config.channels + config.local_channels, config.channels, 1),
                nn.GELU(),
            )
        self.output_projection = nn.Conv1d(config.channels, 1, 1)
        self.count_projection: nn.Module | None = (
            nn.Linear(config.channels, config.count_classes)
            if config.count_classes
            else None
        )

    def _encode(self, values: Tensor) -> Tensor:
        hidden = self.blocks(self.input_projection(values))
        if self.local_input_projection is not None:
            assert self.local_blocks is not None and self.fusion is not None
            local = self.local_blocks(self.local_input_projection(values))
            hidden = self.fusion(torch.cat((hidden, local), dim=1))
        return hidden

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError(
                f"TCN 输入必须是 [batch, {self.input_channels}, time]"
            )
        hidden = self._encode(values)
        return self.output_projection(hidden).squeeze(1)

    def forward_with_count(
        self, values: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """同时输出二元边界热图和窗口内边界数 0/1/2+ 辅助任务。"""

        if self.count_projection is None:
            raise RuntimeError("当前配置没有启用 0/1/2+ 辅助计数头")
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError(
                f"TCN 输入必须是 [batch, {self.input_channels}, time]"
            )
        hidden = self._encode(values)
        logits = self.output_projection(hidden).squeeze(1)
        if mask is None:
            pooled = hidden.mean(dim=2)
        else:
            if mask.shape != logits.shape:
                raise ValueError("计数池化 mask 必须与边界输出同形")
            weights = mask.to(hidden.dtype).unsqueeze(1)
            pooled = (hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return logits, self.count_projection(pooled)


def prepare_boundary_sequences(
    data: pd.DataFrame, *, max_forward_fill_seconds: int = 5,
) -> pd.DataFrame:
    """Prepare explicit 1 Hz grids independently for canonical flights and gaps.

    Times are seconds relative to each flight's first observation, rounded to
    the nearest second (NumPy ties-to-even); collisions keep the last sample.
    This historical gridding convention can shift subsecond observations by
    up to half a second. Causal mode describes the network on this grid.

    Each gap greater than max_forward_fill_seconds starts a new sequence.
    Partial fields are forward-filled within that limit; rows still missing
    any base field are omitted. F1A history restarts after any remaining gap.
    point_index denotes grid seconds; source_point_index identifies the retained
    canonical observation at an occupied cell, and is missing on inserted cells.
    sequence_id is unique across the returned flights. No labels are resampled.
    """
    if not isinstance(max_forward_fill_seconds, int) or max_forward_fill_seconds < 0:
        raise ValueError("max_forward_fill_seconds must be a nonnegative integer")
    canonical = normalize_trajectory(data)
    sequences = []
    sequence_id = 0
    for fid, flight in canonical.groupby("flight_id", sort=False):
        flight = flight.copy()
        flight["_elapsed"] = elapsed_seconds(flight.timestamp)
        groups = flight._elapsed.diff().fillna(0).gt(max_forward_fill_seconds).cumsum()
        for _, group in flight.groupby(groups, sort=False):
            seconds = np.rint(group._elapsed.to_numpy()).astype(int)
            indexed = group.assign(_second=seconds).drop_duplicates("_second", keep="last").set_index("_second")
            grid = np.arange(seconds.min(), seconds.max() + 1)
            result = indexed.loc[:, BASE_FEATURE_COLUMNS].reindex(grid)
            observed = result.notna().all(axis=1)
            if max_forward_fill_seconds:
                result = result.ffill(limit=max_forward_fill_seconds)
            result = result.dropna(subset=list(BASE_FEATURE_COLUMNS))
            if result.empty:
                continue
            observed = observed.reindex(result.index)
            last_observed = np.maximum.accumulate(
                np.where(observed.to_numpy(), result.index.to_numpy(), -10**9)
            )
            result.insert(0, "timestamp", result.index.to_numpy(dtype=float))
            result.insert(0, "flight_id", fid)
            # Any absent grid cell breaks 1 Hz context, even a one-cell hole.
            featured = build_f1a_features(result, max_gap_seconds=1.0)
            featured["point_index"] = result.index.to_numpy(dtype=int)
            featured["source_point_index"] = pd.array(indexed.point_index.reindex(result.index), dtype="Int64")
            featured["observation_present"] = observed.to_numpy(dtype=float)
            featured["observation_age_seconds"] = result.index.to_numpy() - last_observed
            # A removed all-missing run must also reset the network context.
            for _, block in featured.groupby("block_id", sort=False):
                block = block.copy()
                block["sequence_id"] = sequence_id
                sequences.append(block)
                sequence_id += 1
    if not sequences:
        raise ValueError("boundary input has no usable sequences")
    return pd.concat(sequences, ignore_index=True)


@dataclass
class BoundaryTCNBundle:
    """A model and its exact nine-column scaler, with explicit peak settings.

    Fit scaler/model on training folds only. dump refuses existing artifacts;
    load verifies the checkpoint hash before a weights-only Torch load.
    """

    config: BoundaryTCNConfig
    scaler: RobustFeatureScaler
    state_dict: dict[str, Tensor]
    threshold: float
    min_distance_seconds: int
    metadata: dict[str, Any]

    def validate(self) -> None:
        self.config.validate()
        if self.scaler.columns != BOUNDARY_INPUT_COLUMNS:
            raise ValueError("bundle requires the exact boundary input feature order")
        center, scale = np.asarray(self.scaler.center), np.asarray(self.scaler.scale)
        if center.shape != (len(BOUNDARY_INPUT_COLUMNS),) or scale.shape != center.shape:
            raise ValueError("bundle scaler dimensions do not match its features")
        if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("bundle scaler requires finite centers and positive scales")
        if not np.isfinite(self.scaler.clip) or self.scaler.clip <= 0:
            raise ValueError("bundle scaler clip must be finite and positive")
        if not 0 < self.threshold < 1 or self.min_distance_seconds < 1:
            raise ValueError("invalid boundary threshold or minimum peak distance")

    def build_model(self, *, device: str | torch.device = "cpu") -> BoundaryTCN:
        self.validate()
        model = BoundaryTCN(self.config)
        model.load_state_dict(self.state_dict)
        return model.to(device).eval()

    def predict_scores(self, data: pd.DataFrame, *, device: str | torch.device = "cpu") -> pd.DataFrame:
        """Return probabilities on the prepared grid, not raw source rows.

        Every flight/gap sequence is evaluated separately. Use timestamp and
        source_point_index for alignment; inserted grid cells have no source ID.
        """
        prepared = prepare_boundary_sequences(data)
        model = self.build_model(device=device)
        outputs = []
        with torch.inference_mode():
            for _, sequence in prepared.groupby("sequence_id", sort=False):
                matrix = self.scaler.transform(sequence)
                scores = torch.sigmoid(model(torch.from_numpy(matrix.T[None]).to(device)))[0].cpu().numpy()
                result = sequence[["flight_id", "timestamp", "point_index", "source_point_index", "sequence_id"]].copy()
                result["boundary_probability"] = scores
                outputs.append(result)
        return pd.concat(outputs, ignore_index=True)

    def dump(self, directory: str | Path) -> Path:
        self.validate()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        model_path, manifest_path = root / "model.pt", root / "manifest.json"
        if model_path.exists() or manifest_path.exists():
            raise FileExistsError("refusing to overwrite an existing boundary bundle")
        torch.save({
            "config": asdict(self.config), "scaler": asdict(self.scaler),
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict.items()},
            "threshold": self.threshold, "min_distance_seconds": self.min_distance_seconds,
            "metadata": self.metadata,
        }, model_path)
        manifest_path.write_text(json.dumps({"format": "flight-phase-boundary-tcn-v1",
                                             "sha256": _sha256(model_path)}, indent=2) + "\n")
        return root

    @classmethod
    def load(cls, directory: str | Path, *, map_location: str | torch.device = "cpu") -> "BoundaryTCNBundle":
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["format"] != "flight-phase-boundary-tcn-v1":
            raise ValueError("unsupported boundary bundle format")
        model_path = root / "model.pt"
        if _sha256(model_path) != manifest["sha256"]:
            raise ValueError("boundary checkpoint hash mismatch")
        payload = torch.load(model_path, map_location=map_location, weights_only=True)
        bundle = cls(BoundaryTCNConfig(**payload["config"]),
                     RobustFeatureScaler.from_dict(payload["scaler"]), payload["state_dict"],
                     payload["threshold"], payload["min_distance_seconds"], payload["metadata"])
        bundle.validate()
        return bundle


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
