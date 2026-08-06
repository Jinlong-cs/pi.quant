"""Temporal manifests, calibration batches, and axis-aware diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from piquant.analysis import compare_action, compare_tensor, summarize_distribution
from piquant.contracts import (
    ActionMetricSummary,
    ActionSchema,
    MetricDistribution,
    RolloutDivergenceReport,
    TemporalCalibrationManifest,
    TemporalCaptureSpec,
    TemporalMetricReport,
    TemporalMode,
    TensorMetricSummary,
    fingerprint,
)

TEMPORAL_SEQUENCE_REFS_KEY = "__piquant_sequence_refs__"
TemporalBatchLoader = Callable[[Sequence[Any]], Mapping[str, Any]]


def temporal_episode_keys(manifest: TemporalCalibrationManifest) -> set[tuple[str, int]]:
    return {(sequence.suite, sequence.episode_index) for sequence in manifest.sequences}


def require_temporal_episode_disjoint(*manifests: TemporalCalibrationManifest) -> None:
    for index, left in enumerate(manifests):
        left_keys = temporal_episode_keys(left)
        for right in manifests[index + 1 :]:
            overlap = sorted(left_keys & temporal_episode_keys(right))
            if overlap:
                raise ValueError(f"episode overlap between {left.manifest_id!r} and {right.manifest_id!r}: {overlap[:20]!r}")


def save_temporal_manifest(manifest: TemporalCalibrationManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_temporal_manifest(path: str | Path) -> TemporalCalibrationManifest:
    return TemporalCalibrationManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


class BatchedTemporalCalibrationProvider:
    """Load explicit sequence references in deterministic batches through an injected loader."""

    def __init__(self, manifest: TemporalCalibrationManifest, loader: TemporalBatchLoader, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._manifest = manifest
        self._loader = loader
        self._batch_size = batch_size

    @property
    def manifest(self) -> TemporalCalibrationManifest:
        return self._manifest

    @property
    def fingerprint(self) -> str:
        return fingerprint(self._manifest)

    def batches(self) -> Iterable[Mapping[str, Any]]:
        for start in range(0, len(self._manifest.sequences), self._batch_size):
            refs = self._manifest.sequences[start : start + self._batch_size]
            batch = dict(self._loader(refs))
            if TEMPORAL_SEQUENCE_REFS_KEY in batch:
                raise ValueError(f"batch loader must not populate reserved key {TEMPORAL_SEQUENCE_REFS_KEY!r}")
            batch[TEMPORAL_SEQUENCE_REFS_KEY] = refs
            yield batch

    def forward_loop(self, model: Any) -> None:
        for batch in self.batches():
            model(batch)


def _distribution(values: Sequence[float], seed: int) -> MetricDistribution:
    return summarize_distribution(values, seed=seed)


def _action_trajectories(value: Any, axes: Sequence[str], schema: ActionSchema) -> tuple[np.ndarray, list[str]]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != len(axes):
        raise ValueError(f"capture axes length {len(axes)} differs from tensor rank {array.ndim}")
    if axes.count("batch") != 1 or axes.index("batch") != 0:
        raise ValueError("temporal captures must place the unique batch axis at dimension zero")
    action_name = "action" if "action" in axes else "horizon" if "horizon" in axes else None
    dim_name = "action_dim" if "action_dim" in axes else "dim" if "dim" in axes else None
    if action_name is None or dim_name is None:
        raise ValueError(f"action capture axes must include action/horizon and action_dim/dim, got {list(axes)!r}")
    action_axis = axes.index(action_name)
    dim_axis = axes.index(dim_name)
    if array.shape[action_axis] != schema.horizon:
        raise ValueError(f"action axis has length {array.shape[action_axis]}, expected {schema.horizon}")
    if array.shape[dim_axis] < schema.output_action_dim:
        raise ValueError(f"action dimension {array.shape[dim_axis]} is smaller than {schema.output_action_dim}")
    prefix_axes = [index for index in range(1, array.ndim) if index not in {action_axis, dim_axis}]
    order = [0, *prefix_axes, action_axis, dim_axis]
    transposed = np.transpose(array, order)
    prefix_names = [axes[index] for index in prefix_axes]
    return transposed, prefix_names


@dataclass(frozen=True)
class _ActionMetrics:
    l1: float
    l2: float
    direction: float
    translation_l1: float
    translation_l2: float
    translation_direction: float
    rotation_l1: float
    rotation_l2: float
    rotation_direction: float
    gripper_mismatch: float
    horizon_l2: tuple[float, ...]


def _direction(reference: np.ndarray, candidate: np.ndarray) -> float:
    metric = compare_action(reference, candidate).direction_cosine_mean
    if metric is None:
        raise ValueError("finite action slice did not produce a direction metric")
    return metric


def _one_action_metrics(reference: np.ndarray, candidate: np.ndarray, schema: ActionSchema) -> _ActionMetrics:
    metric = compare_action(
        reference,
        candidate,
        gripper_index=schema.gripper_index,
        gripper_threshold=schema.gripper_threshold,
    )
    if metric.l1_mean is None or metric.l2_mean is None or metric.direction_cosine_mean is None:
        raise ValueError("finite action capture did not produce action metrics")
    translation_l1 = float(np.mean(np.abs(reference[..., schema.translation_indices] - candidate[..., schema.translation_indices])))
    translation_l2 = float(
        np.mean(np.linalg.norm(reference[..., schema.translation_indices] - candidate[..., schema.translation_indices], axis=-1))
    )
    rotation_l1 = float(np.mean(np.abs(reference[..., schema.rotation_indices] - candidate[..., schema.rotation_indices])))
    rotation_l2 = float(np.mean(np.linalg.norm(reference[..., schema.rotation_indices] - candidate[..., schema.rotation_indices], axis=-1)))
    return _ActionMetrics(
        l1=metric.l1_mean,
        l2=metric.l2_mean,
        direction=metric.direction_cosine_mean,
        translation_l1=translation_l1,
        translation_l2=translation_l2,
        translation_direction=_direction(reference[..., schema.translation_indices], candidate[..., schema.translation_indices]),
        rotation_l1=rotation_l1,
        rotation_l2=rotation_l2,
        rotation_direction=_direction(reference[..., schema.rotation_indices], candidate[..., schema.rotation_indices]),
        gripper_mismatch=float(
            np.mean(
                (reference[..., schema.gripper_index] > schema.gripper_threshold)
                != (candidate[..., schema.gripper_index] > schema.gripper_threshold)
            )
        ),
        horizon_l2=tuple(
            float(np.mean(np.linalg.norm(reference[..., index, :] - candidate[..., index, :], axis=-1))) for index in range(schema.horizon)
        ),
    )


@dataclass
class _TensorAccumulator:
    sample_count: int = 0
    reference_shape: list[int] = field(default_factory=list)
    candidate_shape: list[int] = field(default_factory=list)
    shape_match: bool = True
    finite: bool = True
    max_abs: list[float] = field(default_factory=list)
    relative_l2: list[float] = field(default_factory=list)
    cosine: list[float] = field(default_factory=list)
    sqnr_db: list[float] = field(default_factory=list)
    by_denoise_step: dict[str, list[float]] = field(default_factory=dict)
    by_stage: dict[str, list[float]] = field(default_factory=dict)
    by_timestep: dict[str, list[float]] = field(default_factory=dict)

    def add(self, reference: Any, candidate: Any, axes: Sequence[str], *, stage: str, timestep: str) -> None:
        self.sample_count += 1
        metric = compare_tensor(reference, candidate)
        if not self.reference_shape:
            self.reference_shape = metric.reference_shape
            self.candidate_shape = metric.candidate_shape
        self.shape_match = self.shape_match and metric.shape_match
        self.finite = self.finite and metric.finite
        if metric.max_abs is None:
            return
        self.max_abs.append(metric.max_abs)
        self.relative_l2.append(metric.relative_l2 or 0.0)
        self.cosine.append(metric.cosine or 0.0)
        if metric.sqnr_db is not None:
            self.sqnr_db.append(metric.sqnr_db)
        self.by_stage.setdefault(stage, []).append(metric.relative_l2 or 0.0)
        self.by_timestep.setdefault(timestep, []).append(metric.relative_l2 or 0.0)
        if "denoise_step" in axes:
            denoise_axis = axes.index("denoise_step") - 1
            reference_array = np.asarray(reference)
            candidate_array = np.asarray(candidate)
            for index in range(reference_array.shape[denoise_axis]):
                step_reference = np.take(reference_array, index, axis=denoise_axis)
                step_candidate = np.take(candidate_array, index, axis=denoise_axis)
                step_metric = compare_tensor(step_reference, step_candidate)
                if step_metric.relative_l2 is not None:
                    self.by_denoise_step.setdefault(str(index), []).append(step_metric.relative_l2)

    def report(self, capture: TemporalCaptureSpec, mode: TemporalMode, seed: int) -> TemporalMetricReport:
        if self.sample_count == 0:
            raise ValueError(f"capture {capture.logical_id!r} has no temporal samples")
        return TemporalMetricReport(
            capture_id=capture.logical_id,
            kind=capture.kind,
            mode=mode,
            axes=capture.axes,
            sample_count=self.sample_count,
            tensor=TensorMetricSummary(
                reference_shape=self.reference_shape,
                candidate_shape=self.candidate_shape,
                shape_match=self.shape_match,
                finite=self.finite,
                max_abs=_distribution(self.max_abs, seed),
                relative_l2=_distribution(self.relative_l2, seed + 1),
                cosine=_distribution(self.cosine, seed + 2),
                sqnr_db=_distribution(self.sqnr_db, seed + 3),
            ),
            by_denoise_step={
                name: _distribution(values, seed + 50 + index) for index, (name, values) in enumerate(sorted(self.by_denoise_step.items()))
            },
            by_stage={
                name: _distribution(values, seed + 100 + index) for index, (name, values) in enumerate(sorted(self.by_stage.items()))
            },
            by_timestep={
                name: _distribution(values, seed + 200 + index) for index, (name, values) in enumerate(sorted(self.by_timestep.items()))
            },
        )


@dataclass
class _ActionAccumulator:
    sample_count: int = 0
    shape_match: bool = True
    finite: bool = True
    values: dict[str, list[float]] = field(
        default_factory=lambda: {
            name: []
            for name in (
                "l1",
                "l2",
                "direction",
                "translation_l1",
                "translation_l2",
                "translation_direction",
                "rotation_l1",
                "rotation_l2",
                "rotation_direction",
                "gripper_mismatch",
            )
        }
    )
    horizon_l2: list[list[float]] = field(default_factory=list)
    by_denoise_step: dict[str, list[float]] = field(default_factory=dict)
    by_stage: dict[str, list[float]] = field(default_factory=dict)
    by_timestep: dict[str, list[float]] = field(default_factory=dict)
    reference_shape: list[int] = field(default_factory=list)
    candidate_shape: list[int] = field(default_factory=list)

    def add(self, reference: Any, candidate: Any, axes: Sequence[str], schema: ActionSchema, *, stage: str, timestep: str) -> None:
        self.sample_count += 1
        reference_array = np.asarray(reference, dtype=np.float64)
        candidate_array = np.asarray(candidate, dtype=np.float64)
        shape_match = reference_array.shape == candidate_array.shape
        finite = bool(np.isfinite(reference_array).all() and np.isfinite(candidate_array).all())
        self.shape_match = self.shape_match and shape_match
        self.finite = self.finite and finite
        if not self.reference_shape:
            self.reference_shape = list(reference_array.shape)
            self.candidate_shape = list(candidate_array.shape)
        if not shape_match or not finite:
            return
        ref, prefix_names = _action_trajectories(reference_array[None, ...], axes, schema)
        cand, _ = _action_trajectories(candidate_array[None, ...], axes, schema)
        ref = ref[0]
        cand = cand[0]
        metrics = _one_action_metrics(
            ref.reshape(-1, schema.horizon, ref.shape[-1]),
            cand.reshape(-1, schema.horizon, cand.shape[-1]),
            schema,
        )
        metric_values = {
            "l1": metrics.l1,
            "l2": metrics.l2,
            "direction": metrics.direction,
            "translation_l1": metrics.translation_l1,
            "translation_l2": metrics.translation_l2,
            "translation_direction": metrics.translation_direction,
            "rotation_l1": metrics.rotation_l1,
            "rotation_l2": metrics.rotation_l2,
            "rotation_direction": metrics.rotation_direction,
            "gripper_mismatch": metrics.gripper_mismatch,
        }
        for name, value in metric_values.items():
            self.values[name].append(value)
        if not self.horizon_l2:
            self.horizon_l2 = [[] for _ in range(schema.horizon)]
        for index, value in enumerate(metrics.horizon_l2):
            self.horizon_l2[index].append(value)
        self.by_stage.setdefault(stage, []).append(metrics.l2)
        self.by_timestep.setdefault(timestep, []).append(metrics.l2)
        if "denoise_step" in prefix_names:
            denoise_index = prefix_names.index("denoise_step")
            prefix_shape = ref.shape[:-2]
            denoise_count = prefix_shape[denoise_index]
            for index in range(denoise_count):
                slicer: list[int | slice] = [slice(None)] * len(prefix_shape)
                slicer[denoise_index] = index
                step_ref = ref[tuple(slicer)].reshape(-1, schema.horizon, ref.shape[-1])
                step_cand = cand[tuple(slicer)].reshape(-1, schema.horizon, cand.shape[-1])
                self.by_denoise_step.setdefault(str(index), []).append(_one_action_metrics(step_ref, step_cand, schema).l2)

    def report(self, capture: TemporalCaptureSpec, mode: TemporalMode, schema: ActionSchema, seed: int) -> TemporalMetricReport:
        if self.sample_count == 0:
            raise ValueError(f"capture {capture.logical_id!r} has no temporal samples")

        def dist(name: str, offset: int) -> MetricDistribution:
            return _distribution(self.values[name], seed + offset)

        action = ActionMetricSummary(
            shape_match=self.shape_match,
            finite=self.finite,
            l1=dist("l1", 0),
            l2=dist("l2", 1),
            direction_cosine=dist("direction", 2),
            translation_l1=dist("translation_l1", 3),
            translation_l2=dist("translation_l2", 4),
            translation_direction_cosine=dist("translation_direction", 5),
            rotation_l1=dist("rotation_l1", 6),
            rotation_l2=dist("rotation_l2", 7),
            rotation_direction_cosine=dist("rotation_direction", 8),
            gripper_mismatch=dist("gripper_mismatch", 9),
            horizon_l2=[_distribution(values, seed + 10 + index) for index, values in enumerate(self.horizon_l2)],
        )
        return TemporalMetricReport(
            capture_id=capture.logical_id,
            kind=capture.kind,
            mode=mode,
            axes=capture.axes,
            sample_count=self.sample_count,
            action=action,
            by_denoise_step={
                name: _distribution(values, seed + 100 + index) for index, (name, values) in enumerate(sorted(self.by_denoise_step.items()))
            },
            by_stage={
                name: _distribution(values, seed + 200 + index) for index, (name, values) in enumerate(sorted(self.by_stage.items()))
            },
            by_timestep={
                name: _distribution(values, seed + 300 + index) for index, (name, values) in enumerate(sorted(self.by_timestep.items()))
            },
        )


class TemporalStreamingAnalyzer:
    """Aggregate temporal captures without retaining raw tensors in process memory."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._action_schema: ActionSchema | None = None
        self._accumulators: dict[str, _TensorAccumulator | _ActionAccumulator] = {}
        self._specs: dict[str, TemporalCaptureSpec] = {}
        self._mode: TemporalMode | None = None

    def add(
        self,
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        sequence_metadata: Sequence[Mapping[str, Any]],
        capture_specs: Sequence[TemporalCaptureSpec],
        *,
        mode: TemporalMode,
        action_schema: ActionSchema,
    ) -> None:
        if not sequence_metadata:
            raise ValueError("temporal diagnostics require sequence metadata")
        if self._mode is not None and self._mode != mode:
            raise ValueError("teacher-forced and iterative captures require separate analyzers")
        if self._action_schema is not None and self._action_schema != action_schema:
            raise ValueError("action schema changed while aggregating temporal diagnostics")
        self._mode = mode
        self._action_schema = action_schema
        missing = sorted(set(reference) - set(candidate))
        if missing:
            raise ValueError(f"candidate temporal capture is missing tensors: {missing}")
        for spec in capture_specs:
            if spec.mode != "both" and spec.mode != mode:
                raise ValueError(f"capture {spec.logical_id!r} does not support temporal mode {mode!r}")
            if spec.logical_id not in reference:
                raise ValueError(f"reference temporal capture is missing {spec.logical_id!r}")
            reference_array = np.asarray(reference[spec.logical_id])
            candidate_array = np.asarray(candidate[spec.logical_id])
            if reference_array.ndim != len(spec.axes) or candidate_array.ndim != len(spec.axes):
                raise ValueError(f"capture {spec.logical_id!r} rank does not match declared axes {spec.axes!r}")
            if reference_array.shape[0] != len(sequence_metadata) or candidate_array.shape[0] != len(sequence_metadata):
                raise ValueError(f"capture {spec.logical_id!r} batch axis does not match sequence metadata")
            accumulator = self._accumulators.get(spec.logical_id)
            if accumulator is None:
                accumulator = _ActionAccumulator() if spec.kind in {"action", "flow"} else _TensorAccumulator()
                self._accumulators[spec.logical_id] = accumulator
                self._specs[spec.logical_id] = spec
            elif self._specs[spec.logical_id] != spec:
                raise ValueError(f"capture spec changed while aggregating {spec.logical_id!r}")
            if isinstance(accumulator, _ActionAccumulator) != (spec.kind in {"action", "flow"}):
                raise ValueError(f"capture kind changed for {spec.logical_id!r}")
            for index, metadata in enumerate(sequence_metadata):
                stage = str(metadata["stage"])
                timestep = f"{float(metadata['timestep']):.3f}" if "timestep" in metadata else "schedule"
                if isinstance(accumulator, _ActionAccumulator):
                    accumulator.add(
                        reference_array[index],
                        candidate_array[index],
                        spec.axes,
                        action_schema,
                        stage=stage,
                        timestep=timestep,
                    )
                else:
                    accumulator.add(reference_array[index], candidate_array[index], spec.axes, stage=stage, timestep=timestep)

    def finalize(self) -> list[TemporalMetricReport]:
        if not self._accumulators:
            raise ValueError("cannot finalize temporal diagnostics without captures")
        if self._mode is None:
            raise ValueError("temporal mode is missing")
        reports: list[TemporalMetricReport] = []
        for index, (logical_id, accumulator) in enumerate(sorted(self._accumulators.items())):
            spec = self._specs[logical_id]
            if isinstance(accumulator, _ActionAccumulator):
                if self._action_schema is None:
                    raise ValueError("action schema is required before finalizing temporal action metrics")
                reports.append(accumulator.report(spec, self._mode, self._action_schema, self.seed + index))
            else:
                reports.append(accumulator.report(spec, self._mode, self.seed + index))
        return reports


class RolloutDivergenceAccumulator:
    """Aggregate final-action and optional world-latent drift by rollout horizon."""

    def __init__(
        self,
        action_capture: TemporalCaptureSpec,
        latent_captures: Sequence[TemporalCaptureSpec],
        *,
        mode: TemporalMode,
        exceedance_threshold: float,
        seed: int = 0,
    ) -> None:
        if action_capture.kind != "rollout" or "rollout_horizon" not in action_capture.axes:
            raise ValueError("rollout divergence requires a kind=rollout capture with a rollout_horizon axis")
        if exceedance_threshold < 0.0:
            raise ValueError("rollout exceedance threshold must be non-negative")
        self.action_capture = action_capture
        self.latent_captures = [capture for capture in latent_captures if "rollout_horizon" in capture.axes]
        self.mode = mode
        self.exceedance_threshold = exceedance_threshold
        self.seed = seed
        self.sample_count = 0
        self.shape_match = True
        self.finite = True
        self._rollout_horizon_steps: list[int] = []
        self._latent_horizon_steps: list[int] = []
        self._action_l2: dict[str, list[float]] = {}
        self._action_direction: dict[str, list[float]] = {}
        self._latent_relative_l2: dict[str, list[float]] = {}
        self._first_exceedance: list[float] = []

    @staticmethod
    def _action_arrays(value: Any, spec: TemporalCaptureSpec, schema: ActionSchema) -> np.ndarray:
        array, prefix_names = _action_trajectories(value, spec.axes, schema)
        if prefix_names != ["rollout_horizon"]:
            raise ValueError(f"rollout action capture must have only rollout_horizon as its prefix axis, found {prefix_names!r}")
        return array[..., : schema.output_action_dim]

    @staticmethod
    def _latent_arrays(value: Any, spec: TemporalCaptureSpec) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != len(spec.axes):
            raise ValueError(f"latent capture {spec.logical_id!r} rank does not match axes")
        if spec.axes[0] != "batch" or "rollout_horizon" not in spec.axes:
            raise ValueError(f"latent capture {spec.logical_id!r} must declare batch first and rollout_horizon")
        horizon_axis = spec.axes.index("rollout_horizon")
        return cast(np.ndarray, np.moveaxis(array, horizon_axis, 1))

    @staticmethod
    def _direction(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
        reference_norm = np.linalg.norm(reference, axis=-1)
        candidate_norm = np.linalg.norm(candidate, axis=-1)
        denominator = np.maximum(reference_norm * candidate_norm, 1e-12)
        return np.asarray(np.sum(reference * candidate, axis=-1) / denominator, dtype=np.float64)

    def add(
        self,
        reference: Mapping[str, Any],
        candidate: Mapping[str, Any],
        action_schema: ActionSchema,
    ) -> None:
        required = {self.action_capture.logical_id, *(capture.logical_id for capture in self.latent_captures)}
        missing = sorted(required - set(reference))
        if missing:
            raise ValueError(f"reference rollout captures are missing: {missing!r}")
        missing = sorted(required - set(candidate))
        if missing:
            raise ValueError(f"candidate rollout captures are missing: {missing!r}")
        reference_action = self._action_arrays(reference[self.action_capture.logical_id], self.action_capture, action_schema)
        candidate_action = self._action_arrays(candidate[self.action_capture.logical_id], self.action_capture, action_schema)
        batch_size = int(reference_action.shape[0])
        self.sample_count += batch_size
        shape_match = reference_action.shape == candidate_action.shape
        finite = bool(np.isfinite(reference_action).all() and np.isfinite(candidate_action).all())
        self.shape_match = self.shape_match and shape_match
        self.finite = self.finite and finite
        if not shape_match or not finite:
            return
        horizon = int(reference_action.shape[1])
        horizon_steps = list(range(1, horizon + 1))
        if self._rollout_horizon_steps and self._rollout_horizon_steps != horizon_steps:
            raise ValueError("rollout horizon changed while aggregating")
        self._rollout_horizon_steps = horizon_steps
        action_l2 = np.mean(np.linalg.norm(candidate_action - reference_action, axis=-1), axis=-1)
        action_direction = self._direction(
            reference_action.reshape(reference_action.shape[0], reference_action.shape[1], -1),
            candidate_action.reshape(candidate_action.shape[0], candidate_action.shape[1], -1),
        )
        for index, step in enumerate(horizon_steps):
            self._action_l2.setdefault(str(step), []).extend(float(value) for value in action_l2[:, index])
            self._action_direction.setdefault(str(step), []).extend(float(value) for value in action_direction[:, index])
        for sample in action_l2:
            exceeded = np.flatnonzero(sample > self.exceedance_threshold)
            if exceeded.size:
                self._first_exceedance.append(float(int(exceeded[0]) + 1))

        for capture in self.latent_captures:
            reference_latent = self._latent_arrays(reference[capture.logical_id], capture)
            candidate_latent = self._latent_arrays(candidate[capture.logical_id], capture)
            latent_shape_match = reference_latent.shape == candidate_latent.shape and reference_latent.shape[0] == batch_size
            latent_finite = bool(np.isfinite(reference_latent).all() and np.isfinite(candidate_latent).all())
            self.shape_match = self.shape_match and latent_shape_match
            self.finite = self.finite and latent_finite
            if not latent_shape_match or not latent_finite:
                continue
            latent_horizon = int(reference_latent.shape[1])
            latent_horizon_steps = list(range(1, latent_horizon + 1))
            if self._latent_horizon_steps and self._latent_horizon_steps != latent_horizon_steps:
                raise ValueError("rollout latent horizon changed while aggregating")
            self._latent_horizon_steps = latent_horizon_steps
            reference_flat = reference_latent.reshape(batch_size, latent_horizon, -1)
            candidate_flat = candidate_latent.reshape(batch_size, latent_horizon, -1)
            relative_l2 = np.linalg.norm(candidate_flat - reference_flat, axis=-1) / np.maximum(
                np.linalg.norm(reference_flat, axis=-1), 1e-12
            )
            for index, step in enumerate(latent_horizon_steps):
                self._latent_relative_l2.setdefault(str(step), []).extend(float(value) for value in relative_l2[:, index])

    def finalize(self) -> RolloutDivergenceReport:
        if self.sample_count == 0 or not self._rollout_horizon_steps:
            raise ValueError("cannot finalize rollout divergence without valid action captures")
        return RolloutDivergenceReport(
            report_id=f"rollout-{self.mode}-{self.action_capture.logical_id}",
            mode=self.mode,
            sample_count=self.sample_count,
            shape_match=self.shape_match,
            finite=self.finite,
            action_capture_id=self.action_capture.logical_id,
            latent_capture_ids=[capture.logical_id for capture in self.latent_captures],
            rollout_horizon_steps=self._rollout_horizon_steps,
            latent_horizon_steps=self._latent_horizon_steps,
            exceedance_threshold=self.exceedance_threshold,
            exceedance_rate=len(self._first_exceedance) / self.sample_count,
            action_l2_by_horizon={
                step: _distribution(values, self.seed + index) for index, (step, values) in enumerate(sorted(self._action_l2.items()))
            },
            action_direction_by_horizon={
                step: _distribution(values, self.seed + 100 + index)
                for index, (step, values) in enumerate(sorted(self._action_direction.items()))
            },
            latent_relative_l2_by_horizon={
                step: _distribution(values, self.seed + 200 + index)
                for index, (step, values) in enumerate(sorted(self._latent_relative_l2.items()))
            },
            first_exceedance_horizon=_distribution(self._first_exceedance, self.seed + 300),
        )


def __getattr__(name: str) -> Any:
    if name == "TemporalSensitivityRunner":
        from piquant.temporal_study import TemporalSensitivityRunner

        return TemporalSensitivityRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
