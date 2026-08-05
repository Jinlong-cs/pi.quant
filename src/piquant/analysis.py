"""Numerical and action-level comparisons for VLA optimization evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from piquant.contracts import ActionMetric, ComparisonReport, TensorMetric


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _safe_float(value: np.ndarray | float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite comparison metric: {result}")
    return result


def compare_tensor(reference: Any, candidate: Any) -> TensorMetric:
    """Compare one named activation while retaining shape and finite gates."""

    reference_array = _array(reference)
    candidate_array = _array(candidate)
    shape_match = reference_array.shape == candidate_array.shape
    finite = bool(np.isfinite(reference_array).all() and np.isfinite(candidate_array).all())
    if not shape_match or not finite:
        return TensorMetric(
            reference_shape=list(reference_array.shape),
            candidate_shape=list(candidate_array.shape),
            shape_match=shape_match,
            finite=finite,
        )

    delta = candidate_array - reference_array
    reference_norm = float(np.linalg.norm(reference_array.ravel()))
    delta_norm = float(np.linalg.norm(delta.ravel()))
    candidate_norm = float(np.linalg.norm(candidate_array.ravel()))
    denominator = max(reference_norm, 1e-12)
    cosine_denominator = max(reference_norm * candidate_norm, 1e-12)
    signal_power = float(np.sum(reference_array * reference_array))
    noise_power = float(np.sum(delta * delta))
    sqnr_db = None if noise_power == 0.0 else _safe_float(10.0 * np.log10(signal_power / noise_power))
    return TensorMetric(
        reference_shape=list(reference_array.shape),
        candidate_shape=list(candidate_array.shape),
        shape_match=True,
        finite=True,
        max_abs=_safe_float(np.max(np.abs(delta))),
        relative_l2=_safe_float(delta_norm / denominator),
        cosine=_safe_float(np.dot(reference_array.ravel(), candidate_array.ravel()) / cosine_denominator),
        sqnr_db=sqnr_db,
    )


def compare_action(
    reference: Any,
    candidate: Any,
    *,
    gripper_index: int | None = None,
    gripper_threshold: float = 0.5,
) -> ActionMetric:
    """Compare action vectors, direction, and optional gripper state."""

    reference_array = _array(reference)
    candidate_array = _array(candidate)
    shape_match = reference_array.shape == candidate_array.shape
    finite = bool(np.isfinite(reference_array).all() and np.isfinite(candidate_array).all())
    if not shape_match or not finite:
        return ActionMetric(shape_match=shape_match, finite=finite)

    if reference_array.ndim == 1:
        reference_array = reference_array[None, :]
        candidate_array = candidate_array[None, :]
    delta = candidate_array - reference_array
    reference_norm = np.linalg.norm(reference_array, axis=-1)
    candidate_norm = np.linalg.norm(candidate_array, axis=-1)
    denominator = np.maximum(reference_norm * candidate_norm, 1e-12)
    directions = np.sum(reference_array * candidate_array, axis=-1) / denominator
    mismatch_rate: float | None = None
    if gripper_index is not None:
        if gripper_index < 0 or gripper_index >= reference_array.shape[-1]:
            raise ValueError(f"gripper_index {gripper_index} is outside action dimension {reference_array.shape[-1]}")
        reference_state = reference_array[:, gripper_index] > gripper_threshold
        candidate_state = candidate_array[:, gripper_index] > gripper_threshold
        mismatch_rate = _safe_float(np.mean(reference_state != candidate_state))
    return ActionMetric(
        shape_match=True,
        finite=True,
        l1_mean=_safe_float(np.mean(np.abs(delta))),
        l2_mean=_safe_float(np.mean(np.linalg.norm(delta, axis=-1))),
        direction_cosine_mean=_safe_float(np.mean(directions)),
        gripper_mismatch_rate=mismatch_rate,
    )


class NumpyNumericalAnalyzer:
    """Reference analyzer usable by PyTorch hooks and ORT output arrays."""

    def __init__(self, gripper_index: int | None = None, gripper_threshold: float = 0.5) -> None:
        self.gripper_index = gripper_index
        self.gripper_threshold = gripper_threshold

    def compare(self, reference: Mapping[str, Any], candidate: Mapping[str, Any], action_name: str) -> ComparisonReport:
        missing = sorted(set(reference) - set(candidate))
        if missing:
            raise ValueError(f"candidate capture is missing tensors: {missing}")
        tensor_metrics = {name: compare_tensor(reference[name], candidate[name]) for name in sorted(reference) if name != action_name}
        if action_name not in reference or action_name not in candidate:
            raise ValueError(f"action tensor {action_name!r} must be present in both captures")
        return ComparisonReport(
            tensors=tensor_metrics,
            action=compare_action(
                reference[action_name],
                candidate[action_name],
                gripper_index=self.gripper_index,
                gripper_threshold=self.gripper_threshold,
            ),
        )
