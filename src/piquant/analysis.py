"""Numerical and action-level comparisons for VLA optimization evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from piquant.contracts import (
    ActionMetric,
    ActionMetricSummary,
    ActionSchema,
    ComparisonReport,
    DiagnosticBucket,
    MetricDistribution,
    SensitivityDiagnostics,
    TensorMetric,
    TensorMetricSummary,
)


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
    sqnr_db = None if noise_power == 0.0 or signal_power == 0.0 else _safe_float(10.0 * np.log10(signal_power / noise_power))
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
        reference_state = reference_array[..., gripper_index] > gripper_threshold
        candidate_state = candidate_array[..., gripper_index] > gripper_threshold
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


def summarize_distribution(values: Sequence[float], *, bootstrap_samples: int = 256, seed: int = 0) -> MetricDistribution:
    """Summarize paired sample metrics with a deterministic bootstrap interval for the mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return MetricDistribution(count=0)
    if not np.isfinite(array).all():
        raise ValueError("metric distribution contains non-finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(bootstrap_samples, array.size))
    bootstrap_means = np.mean(array[indices], axis=1)
    return MetricDistribution(
        count=int(array.size),
        mean=float(np.mean(array)),
        std=float(np.std(array)),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        p50=float(np.quantile(array, 0.50)),
        p95=float(np.quantile(array, 0.95)),
        bootstrap_low=float(np.quantile(bootstrap_means, 0.025)),
        bootstrap_high=float(np.quantile(bootstrap_means, 0.975)),
    )


@dataclass
class _TensorAccumulator:
    reference_shape: list[int] = field(default_factory=list)
    candidate_shape: list[int] = field(default_factory=list)
    shape_match: bool = True
    finite: bool = True
    max_abs: list[float] = field(default_factory=list)
    relative_l2: list[float] = field(default_factory=list)
    cosine: list[float] = field(default_factory=list)
    sqnr_db: list[float] = field(default_factory=list)

    def add(self, metric: TensorMetric) -> None:
        if not self.reference_shape:
            self.reference_shape = metric.reference_shape
            self.candidate_shape = metric.candidate_shape
        self.shape_match = self.shape_match and metric.shape_match
        self.finite = self.finite and metric.finite
        for values, value in (
            (self.max_abs, metric.max_abs),
            (self.relative_l2, metric.relative_l2),
            (self.cosine, metric.cosine),
            (self.sqnr_db, metric.sqnr_db),
        ):
            if value is not None:
                values.append(value)

    def summary(self, seed: int) -> TensorMetricSummary:
        return TensorMetricSummary(
            reference_shape=self.reference_shape,
            candidate_shape=self.candidate_shape,
            shape_match=self.shape_match,
            finite=self.finite,
            max_abs=summarize_distribution(self.max_abs, seed=seed),
            relative_l2=summarize_distribution(self.relative_l2, seed=seed + 1),
            cosine=summarize_distribution(self.cosine, seed=seed + 2),
            sqnr_db=summarize_distribution(self.sqnr_db, seed=seed + 3),
        )


def _vector_metrics(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float, float]:
    delta = candidate - reference
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    denominator = max(float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)), 1e-12)
    return (
        float(np.mean(np.abs(delta))),
        float(np.mean(np.linalg.norm(delta, axis=-1))),
        float(np.dot(reference_flat, candidate_flat) / denominator),
    )


@dataclass
class _ActionAccumulator:
    shape_match: bool = True
    finite: bool = True
    l1: list[float] = field(default_factory=list)
    l2: list[float] = field(default_factory=list)
    direction: list[float] = field(default_factory=list)
    translation_l1: list[float] = field(default_factory=list)
    translation_l2: list[float] = field(default_factory=list)
    translation_direction: list[float] = field(default_factory=list)
    rotation_l1: list[float] = field(default_factory=list)
    rotation_l2: list[float] = field(default_factory=list)
    rotation_direction: list[float] = field(default_factory=list)
    gripper_mismatch: list[float] = field(default_factory=list)
    horizon_l2: list[list[float]] = field(default_factory=list)

    def add(self, reference: Any, candidate: Any, schema: ActionSchema) -> None:
        reference_array = _array(reference)
        candidate_array = _array(candidate)
        self.shape_match = self.shape_match and reference_array.shape == candidate_array.shape
        self.finite = self.finite and bool(np.isfinite(reference_array).all() and np.isfinite(candidate_array).all())
        if reference_array.shape != candidate_array.shape or not self.finite:
            return
        if reference_array.ndim == 1:
            reference_array = reference_array[None, :]
            candidate_array = candidate_array[None, :]
        if reference_array.shape[-2] != schema.horizon:
            raise ValueError(f"action horizon {reference_array.shape[-2]} differs from schema horizon {schema.horizon}")
        if reference_array.shape[-1] < schema.output_action_dim:
            raise ValueError(f"action dimension {reference_array.shape[-1]} is smaller than schema output dim {schema.output_action_dim}")
        overall = _vector_metrics(reference_array, candidate_array)
        translation = _vector_metrics(reference_array[..., schema.translation_indices], candidate_array[..., schema.translation_indices])
        rotation = _vector_metrics(reference_array[..., schema.rotation_indices], candidate_array[..., schema.rotation_indices])
        self.l1.append(overall[0])
        self.l2.append(overall[1])
        self.direction.append(overall[2])
        self.translation_l1.append(translation[0])
        self.translation_l2.append(translation[1])
        self.translation_direction.append(translation[2])
        self.rotation_l1.append(rotation[0])
        self.rotation_l2.append(rotation[1])
        self.rotation_direction.append(rotation[2])
        reference_gripper = reference_array[..., schema.gripper_index] > schema.gripper_threshold
        candidate_gripper = candidate_array[..., schema.gripper_index] > schema.gripper_threshold
        self.gripper_mismatch.append(float(np.mean(reference_gripper != candidate_gripper)))
        horizon = reference_array.shape[-2]
        if not self.horizon_l2:
            self.horizon_l2 = [[] for _ in range(horizon)]
        if len(self.horizon_l2) != horizon:
            raise ValueError(f"action horizon changed within one study: {len(self.horizon_l2)} != {horizon}")
        delta = candidate_array - reference_array
        for index in range(horizon):
            self.horizon_l2[index].append(float(np.mean(np.linalg.norm(delta[..., index, :], axis=-1))))

    def summary(self, seed: int) -> ActionMetricSummary:
        return ActionMetricSummary(
            shape_match=self.shape_match,
            finite=self.finite,
            l1=summarize_distribution(self.l1, seed=seed),
            l2=summarize_distribution(self.l2, seed=seed + 1),
            direction_cosine=summarize_distribution(self.direction, seed=seed + 2),
            translation_l1=summarize_distribution(self.translation_l1, seed=seed + 3),
            translation_l2=summarize_distribution(self.translation_l2, seed=seed + 4),
            translation_direction_cosine=summarize_distribution(self.translation_direction, seed=seed + 5),
            rotation_l1=summarize_distribution(self.rotation_l1, seed=seed + 6),
            rotation_l2=summarize_distribution(self.rotation_l2, seed=seed + 7),
            rotation_direction_cosine=summarize_distribution(self.rotation_direction, seed=seed + 8),
            gripper_mismatch=summarize_distribution(self.gripper_mismatch, seed=seed + 9),
            horizon_l2=[summarize_distribution(values, seed=seed + 10 + index) for index, values in enumerate(self.horizon_l2)],
        )


@dataclass
class _BucketAccumulator:
    tensors: dict[str, _TensorAccumulator] = field(default_factory=dict)
    action: _ActionAccumulator = field(default_factory=_ActionAccumulator)
    flow: _ActionAccumulator | None = None
    sample_count: int = 0

    def add(
        self,
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        sample_index: int,
        *,
        action_name: str,
        flow_name: str | None,
        action_schema: ActionSchema,
    ) -> None:
        self.sample_count += 1
        for name in sorted(reference):
            if name in {action_name, flow_name}:
                continue
            if name not in candidate:
                raise ValueError(f"candidate capture is missing tensor {name!r}")
            metric = compare_tensor(np.asarray(reference[name])[sample_index], np.asarray(candidate[name])[sample_index])
            self.tensors.setdefault(name, _TensorAccumulator()).add(metric)
        if action_name not in reference or action_name not in candidate:
            raise ValueError(f"action tensor {action_name!r} must be present in both captures")
        self.action.add(np.asarray(reference[action_name])[sample_index], np.asarray(candidate[action_name])[sample_index], action_schema)
        if flow_name is not None:
            if flow_name not in reference or flow_name not in candidate:
                raise ValueError(f"flow tensor {flow_name!r} must be present in both captures")
            if self.flow is None:
                self.flow = _ActionAccumulator()
            self.flow.add(np.asarray(reference[flow_name])[sample_index], np.asarray(candidate[flow_name])[sample_index], action_schema)

    def summary(self, seed: int) -> DiagnosticBucket:
        return DiagnosticBucket(
            sample_count=self.sample_count,
            tensors={
                name: accumulator.summary(seed + index * 17) for index, (name, accumulator) in enumerate(sorted(self.tensors.items()))
            },
            action=self.action.summary(seed + 1000),
            flow=None if self.flow is None else self.flow.summary(seed + 2000),
        )


class StreamingDiagnosticAnalyzer(NumpyNumericalAnalyzer):
    """Aggregate sample-paired activation, action, stage, timestep, and horizon diagnostics."""

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self._overall = _BucketAccumulator()
        self._by_stage: dict[str, _BucketAccumulator] = {}
        self._by_timestep: dict[str, _BucketAccumulator] = {}

    def add(
        self,
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        sample_metadata: Sequence[Mapping[str, Any]],
        *,
        action_name: str,
        flow_name: str | None = None,
        action_schema: ActionSchema,
    ) -> None:
        batch_size = np.asarray(reference[action_name]).shape[0]
        if batch_size != len(sample_metadata):
            raise ValueError(f"sample metadata count {len(sample_metadata)} differs from capture batch {batch_size}")
        for index, metadata in enumerate(sample_metadata):
            stage = str(metadata["stage"])
            timestep = f"{float(metadata['timestep']):.3f}"
            buckets = (
                self._overall,
                self._by_stage.setdefault(stage, _BucketAccumulator()),
                self._by_timestep.setdefault(timestep, _BucketAccumulator()),
            )
            for bucket in buckets:
                bucket.add(reference, candidate, index, action_name=action_name, flow_name=flow_name, action_schema=action_schema)

    def finalize(self) -> SensitivityDiagnostics:
        if self._overall.sample_count == 0:
            raise ValueError("cannot finalize diagnostics without samples")
        return SensitivityDiagnostics(
            overall=self._overall.summary(self.seed),
            by_stage={name: bucket.summary(self.seed + index * 100) for index, (name, bucket) in enumerate(sorted(self._by_stage.items()))},
            by_timestep={
                name: bucket.summary(self.seed + 10_000 + index * 100)
                for index, (name, bucket) in enumerate(sorted(self._by_timestep.items()))
            },
        )
