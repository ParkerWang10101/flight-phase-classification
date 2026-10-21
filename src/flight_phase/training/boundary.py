"""Boundary-target and masked focal-loss primitives for fold-local training."""

import numpy as np
import torch
from torch import Tensor

from flight_phase.data.time import elapsed_seconds


def gaussian_boundary_target(timestamps, boundary_indices, block_ids, *, sigma_seconds=2.0, radius_seconds=6.0):
    """Combine Gaussian targets by maximum, never spreading across time blocks.

    Supply indices of label changes inside a flight's continuous blocks; a
    block start is not itself a phase transition. No labels or files are read.
    """
    times = elapsed_seconds(timestamps)
    blocks = np.asarray(block_ids)
    indices = np.asarray(boundary_indices)
    if blocks.shape != times.shape or indices.ndim != 1:
        raise ValueError("block IDs and boundary indices must align with timestamps")
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0 or not np.isfinite(radius_seconds) or radius_seconds < 0:
        raise ValueError("Gaussian width must be positive and radius nonnegative")
    if not np.issubdtype(indices.dtype, np.integer) and indices.size:
        raise ValueError("boundary indices must be integers")
    if np.any(indices < 0) or np.any(indices >= len(times)):
        raise ValueError("boundary indices exceed the trajectory")
    result = np.zeros(len(times), dtype=np.float32)
    for point in indices.astype(int):
        if point == 0 or blocks[point] < 0 or blocks[point] != blocks[point - 1]:
            raise ValueError("a boundary must be inside a covered continuous block")
        distance = np.abs(times - times[point])
        valid = (distance <= radius_seconds) & (blocks == blocks[point])
        result[valid] = np.maximum(result[valid], np.exp(-0.5 * (distance[valid] / sigma_seconds) ** 2))
    return result


def sigmoid_focal_loss(logits: Tensor, target: Tensor, mask: Tensor, *, alpha=0.75, gamma=2.0) -> Tensor:
    """Mean masked sigmoid focal loss, matching the retained boundary trainer."""
    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets and mask must have identical shapes")
    probability = torch.sigmoid(logits)
    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability_true = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_weight = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_weight * (1.0 - probability_true).pow(gamma) * cross_entropy
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
