"""Versioned, serializable data contracts shared by pi.quant components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Reject undocumented fields so recipe drift fails at the boundary."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelSpec(Contract):
    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    framework: str = Field(min_length=1)
    revision: str = "local"
    task: Literal["vla", "wam", "flow_action"] = "vla"
    action_dim: int = Field(gt=0)
    action_horizon: int = Field(gt=0)


class ActionSchema(Contract):
    """Explicit model-to-task action contract used by real VLA studies."""

    schema_version: Literal[1] = 1
    model_action_dim: int = Field(gt=0)
    output_action_dim: int = Field(gt=0)
    horizon: int = Field(gt=0)
    denoise_steps: int = Field(gt=0)
    translation_indices: list[int] = Field(min_length=1)
    rotation_indices: list[int] = Field(min_length=1)
    gripper_index: int = Field(ge=0)
    gripper_threshold: float = 0.0
    postprocess: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_indices(self) -> ActionSchema:
        indices = [*self.translation_indices, *self.rotation_indices, self.gripper_index]
        if len(indices) != len(set(indices)):
            raise ValueError("action component indices must be unique")
        if max(indices) >= self.output_action_dim:
            raise ValueError("action component index exceeds output_action_dim")
        if self.output_action_dim > self.model_action_dim:
            raise ValueError("output_action_dim cannot exceed model_action_dim")
        return self


TemporalSplit = Literal["calibration", "diagnostic_holdout", "static_control", "promotion_reserved"]
TemporalMode = Literal["teacher_forced", "iterative"]
EvidenceStatus = Literal["pending", "measured", "accepted", "rejected"]
PrecisionMode = Literal["fp32", "fp16", "bf16", "int8", "fp8", "nvfp4"]
SupportStatus = Literal["supported", "unsupported", "pending"]


class SequenceRef(Contract):
    """Identity-bound temporal sample used by WAM calibration and diagnostics."""

    schema_version: Literal[1] = 1
    sequence_id: str = Field(min_length=1)
    split: TemporalSplit
    suite: str = Field(min_length=1)
    task: str = Field(min_length=1)
    task_index: int = Field(ge=0)
    episode_index: int = Field(ge=0)
    frame_start: int = Field(ge=0)
    frame_end: int = Field(gt=0)
    observation_indices: list[int] = Field(min_length=1)
    camera_keys: list[str] = Field(min_length=1)
    instruction_sha256: str = Field(min_length=64, max_length=64)
    action_target_sha256: str = Field(min_length=64, max_length=64)
    normalization_revision: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    seed: int
    flow_noise_seed: int
    timestep_schedule: list[float] = Field(min_length=1)
    denoise_steps: int = Field(gt=0)
    action_horizon: int = Field(gt=0)
    action_dim: int = Field(gt=0)
    sampling_mode: Literal["temporal_balanced", "static_frame"]

    @model_validator(mode="after")
    def validate_temporal_window(self) -> SequenceRef:
        if self.frame_end <= self.frame_start:
            raise ValueError("frame_end must be greater than frame_start")
        if self.observation_indices != sorted(set(self.observation_indices)):
            raise ValueError("observation_indices must be sorted and unique")
        if any(index < self.frame_start or index >= self.frame_end for index in self.observation_indices):
            raise ValueError("observation_indices must be inside [frame_start, frame_end)")
        if any(timestep < 0.0 or timestep > 1.0 for timestep in self.timestep_schedule):
            raise ValueError("timestep_schedule values must be inside [0, 1]")
        if len(self.timestep_schedule) != self.denoise_steps:
            raise ValueError("timestep_schedule length must equal denoise_steps")
        return self


class TemporalCalibrationManifest(Contract):
    """Episode-aware sequence manifest with an explicit calibration sampling mode."""

    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    split: TemporalSplit
    preprocess_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    stage_rule: str = Field(min_length=1)
    sampling_mode: Literal["temporal_balanced", "static_frame"]
    denoise_steps: int = Field(gt=0)
    action_horizon: int = Field(gt=0)
    sequences: list[SequenceRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sequences(self) -> TemporalCalibrationManifest:
        sequence_ids = [sequence.sequence_id for sequence in self.sequences]
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("temporal manifest sequence_id values must be unique")
        if any(sequence.split != self.split for sequence in self.sequences):
            raise ValueError("sequence split must match temporal manifest split")
        if any(sequence.normalization_revision != self.normalization_revision for sequence in self.sequences):
            raise ValueError("sequence normalization revision must match temporal manifest")
        if any(sequence.sampling_mode != self.sampling_mode for sequence in self.sequences):
            raise ValueError("sequence sampling mode must match temporal manifest")
        if any(sequence.denoise_steps != self.denoise_steps for sequence in self.sequences):
            raise ValueError("sequence denoise steps must match temporal manifest")
        if any(sequence.action_horizon != self.action_horizon for sequence in self.sequences):
            raise ValueError("sequence action horizon must match temporal manifest")
        return self


class ModuleDescriptor(Contract):
    schema_version: Literal[1] = 1
    logical_id: str = Field(min_length=1)
    backend_path: str = Field(min_length=1)
    component: str = Field(min_length=1)
    block_index: int | None = Field(default=None, ge=0)
    op_family: str = Field(min_length=1)
    parameter_count: int = Field(ge=0)
    quantizable: bool
    tags: list[str] = Field(default_factory=list)


class CaptureSpec(Contract):
    schema_version: Literal[1] = 1
    logical_id: str = Field(min_length=1)
    backend_path: str = Field(min_length=1)
    component: str = Field(min_length=1)
    kind: Literal["activation", "cache", "flow", "action", "gripper"]
    block_index: int | None = Field(default=None, ge=0)
    output_index: int | None = Field(default=None, ge=0)


class TemporalCaptureSpec(Contract):
    """Named WAM capture with explicit tensor axes and execution mode."""

    schema_version: Literal[1] = 1
    logical_id: str = Field(min_length=1)
    backend_path: str = Field(min_length=1)
    component: str = Field(min_length=1)
    kind: Literal["activation", "cache", "latent", "flow", "action", "gripper", "rollout"]
    axes: list[str] = Field(min_length=1)
    mode: Literal["teacher_forced", "iterative", "both"] = "both"
    block_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_axes(self) -> TemporalCaptureSpec:
        if len(self.axes) != len(set(self.axes)):
            raise ValueError("temporal capture axes must be unique")
        if self.axes[0] != "batch":
            raise ValueError("temporal capture axes must place batch first")
        return self


class SampleRef(Contract):
    schema_version: Literal[1] = 1
    sample_id: str = Field(min_length=1)
    split: Literal["calibration", "diagnostic_holdout", "random_control", "promotion_reserved"]
    suite: str = Field(min_length=1)
    task: str = Field(min_length=1)
    task_index: int = Field(ge=0)
    episode_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    window_start: int = Field(ge=0)
    window_end: int = Field(gt=0)
    camera_keys: list[str] = Field(min_length=1)
    instruction_sha256: str = Field(min_length=64, max_length=64)
    history_window: list[int] = Field(default_factory=list)
    action_target_sha256: str = Field(min_length=64, max_length=64)
    normalization_revision: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(min_length=64, max_length=64)
    seed: int
    flow_noise_seed: int
    timestep: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_window(self) -> SampleRef:
        if not self.window_start <= self.frame_index < self.window_end:
            raise ValueError("frame_index must be inside [window_start, window_end)")
        return self


class CalibrationManifest(Contract):
    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    split: Literal["calibration", "diagnostic_holdout", "random_control", "promotion_reserved"]
    preprocess_revision: str = Field(min_length=1)
    normalization_revision: str = Field(min_length=1)
    stage_rule: str = Field(min_length=1)
    samples: list[SampleRef] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_samples(self) -> CalibrationManifest:
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("calibration manifest sample_id values must be unique")
        if any(sample.split != self.split for sample in self.samples):
            raise ValueError("sample split must match calibration manifest split")
        if any(sample.normalization_revision != self.normalization_revision for sample in self.samples):
            raise ValueError("sample normalization revision must match calibration manifest")
        return self


class CalibrationSpec(Contract):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = "local"
    sample_count: int = Field(gt=0)
    seed: int
    stages: list[str] = Field(min_length=1)
    input_fields: list[str] = Field(min_length=1)


class OptimizationPlan(Contract):
    schema_version: Literal[1] = 1
    plan_id: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    quant_format: Literal["int8", "none"] = "int8"
    representation: Literal["fake_quant", "real_quant", "fp_control"] = "fake_quant"
    module_include: list[str] = Field(default_factory=lambda: ["*"])
    module_exclude: list[str] = Field(default_factory=list)
    calibration: CalibrationSpec
    seed: int
    capture_points: list[str] = Field(min_length=1)
    timing_boundary: str = Field(min_length=1)


class TargetFingerprint(Contract):
    schema_version: Literal[1] = 1
    platform: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    device: str = Field(min_length=1)
    gpu_name: str = "unavailable"
    compute_capability: str = "unavailable"
    driver_version: str = "unavailable"
    cuda_version: str = "unavailable"
    tensorrt_version: str = "unavailable"
    torch_version: str = "unavailable"
    modelopt_version: str = "unavailable"
    onnx_version: str = "unavailable"
    onnxruntime_version: str = "unavailable"
    memory_total_mib: int | None = Field(default=None, ge=0)
    power_mode: str = "unavailable"
    clock_policy: str = "unavailable"
    container_image: str = "host"


class ModuleCoverage(Contract):
    schema_version: Literal[1] = 1
    candidate_count: int = Field(ge=0)
    matched_count: int = Field(ge=0)
    excluded_count: int = Field(ge=0)
    candidate_names: list[str] = Field(default_factory=list)
    matched_names: list[str] = Field(default_factory=list)
    excluded_names: list[str] = Field(default_factory=list)
    matched_logical_ids: list[str] = Field(default_factory=list)
    resolved_backend_names: list[str] = Field(default_factory=list)
    candidate_parameter_count: int = Field(default=0, ge=0)
    matched_parameter_count: int = Field(default=0, ge=0)


class QuantizationResult(Contract):
    schema_version: Literal[1] = 1
    backend: str = Field(min_length=1)
    representation: Literal["fake_quant", "real_quant", "reference_qdq", "fp_control"]
    status: Literal["measured", "pending", "rejected"]
    quant_format: str = Field(min_length=1)
    module_coverage: ModuleCoverage
    metadata: dict[str, Any] = Field(default_factory=dict)


class TensorMetric(Contract):
    schema_version: Literal[1] = 1
    reference_shape: list[int]
    candidate_shape: list[int]
    shape_match: bool
    finite: bool
    max_abs: float | None = None
    relative_l2: float | None = None
    cosine: float | None = None
    sqnr_db: float | None = None


class ActionMetric(Contract):
    schema_version: Literal[1] = 1
    shape_match: bool
    finite: bool
    l1_mean: float | None = None
    l2_mean: float | None = None
    direction_cosine_mean: float | None = None
    gripper_mismatch_rate: float | None = None


class ComparisonReport(Contract):
    schema_version: Literal[1] = 1
    tensors: dict[str, TensorMetric] = Field(default_factory=dict)
    action: ActionMetric


class MetricDistribution(Contract):
    schema_version: Literal[1] = 1
    count: int = Field(ge=0)
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    p50: float | None = None
    p95: float | None = None
    bootstrap_low: float | None = None
    bootstrap_high: float | None = None


class TensorMetricSummary(Contract):
    schema_version: Literal[1] = 1
    reference_shape: list[int]
    candidate_shape: list[int]
    shape_match: bool
    finite: bool
    max_abs: MetricDistribution
    relative_l2: MetricDistribution
    cosine: MetricDistribution
    sqnr_db: MetricDistribution


class ActionMetricSummary(Contract):
    schema_version: Literal[1] = 1
    shape_match: bool
    finite: bool
    l1: MetricDistribution
    l2: MetricDistribution
    direction_cosine: MetricDistribution
    translation_l1: MetricDistribution
    translation_l2: MetricDistribution
    translation_direction_cosine: MetricDistribution
    rotation_l1: MetricDistribution
    rotation_l2: MetricDistribution
    rotation_direction_cosine: MetricDistribution
    gripper_mismatch: MetricDistribution
    horizon_l2: list[MetricDistribution] = Field(default_factory=list)


class TemporalMetricReport(Contract):
    """Streaming numerical summary for one explicitly-axis-labelled temporal capture."""

    schema_version: Literal[1] = 1
    capture_id: str = Field(min_length=1)
    kind: Literal["activation", "cache", "latent", "flow", "action", "gripper", "rollout"]
    mode: TemporalMode
    axes: list[str] = Field(min_length=1)
    sample_count: int = Field(gt=0)
    tensor: TensorMetricSummary | None = None
    action: ActionMetricSummary | None = None
    by_denoise_step: dict[str, MetricDistribution] = Field(default_factory=dict)
    by_rollout_horizon: dict[str, MetricDistribution] = Field(default_factory=dict)
    by_stage: dict[str, MetricDistribution] = Field(default_factory=dict)
    by_timestep: dict[str, MetricDistribution] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metric_payload(self) -> TemporalMetricReport:
        if (self.tensor is None) == (self.action is None):
            raise ValueError("temporal metric report requires exactly one tensor or action summary")
        if len(self.axes) != len(set(self.axes)) or self.axes[0] != "batch":
            raise ValueError("temporal metric report axes must be unique and place batch first")
        return self


class RolloutDivergenceReport(Contract):
    """Action/world drift across an explicitly labelled rollout boundary."""

    schema_version: Literal[1] = 1
    report_id: str = Field(min_length=1)
    mode: TemporalMode
    sample_count: int = Field(gt=0)
    shape_match: bool
    finite: bool
    action_capture_id: str = Field(min_length=1)
    latent_capture_ids: list[str] = Field(default_factory=list)
    rollout_horizon_steps: list[int] = Field(min_length=1)
    latent_horizon_steps: list[int] = Field(default_factory=list)
    exceedance_threshold: float = Field(ge=0.0)
    exceedance_rate: float = Field(ge=0.0, le=1.0)
    action_l2_by_horizon: dict[str, MetricDistribution] = Field(default_factory=dict)
    action_direction_by_horizon: dict[str, MetricDistribution] = Field(default_factory=dict)
    latent_relative_l2_by_horizon: dict[str, MetricDistribution] = Field(default_factory=dict)
    first_exceedance_horizon: MetricDistribution | None = None

    @model_validator(mode="after")
    def validate_horizons(self) -> RolloutDivergenceReport:
        for name, steps in (("rollout", self.rollout_horizon_steps), ("latent", self.latent_horizon_steps)):
            if steps != sorted(set(steps)) or any(step < 0 for step in steps):
                raise ValueError(f"{name} horizon steps must be sorted, unique, and non-negative")
        rollout_allowed = {str(step) for step in self.rollout_horizon_steps}
        rollout_distributions = (
            ("action_l2_by_horizon", self.action_l2_by_horizon),
            ("action_direction_by_horizon", self.action_direction_by_horizon),
        )
        for name, values in rollout_distributions:
            if unknown := sorted(set(values) - rollout_allowed):
                raise ValueError(f"{name} contains undeclared rollout horizons: {unknown!r}")
        latent_allowed = {str(step) for step in self.latent_horizon_steps}
        if unknown := sorted(set(self.latent_relative_l2_by_horizon) - latent_allowed):
            raise ValueError(f"latent_relative_l2_by_horizon contains undeclared latent horizons: {unknown!r}")
        return self


class DiagnosticBucket(Contract):
    schema_version: Literal[1] = 1
    sample_count: int = Field(ge=0)
    tensors: dict[str, TensorMetricSummary] = Field(default_factory=dict)
    action: ActionMetricSummary
    flow: ActionMetricSummary | None = None


class SensitivityDiagnostics(Contract):
    schema_version: Literal[1] = 1
    overall: DiagnosticBucket
    by_stage: dict[str, DiagnosticBucket] = Field(default_factory=dict)
    by_timestep: dict[str, DiagnosticBucket] = Field(default_factory=dict)


class SensitivityTrial(Contract):
    schema_version: Literal[1] = 1
    trial_id: str = Field(min_length=1)
    kind: Literal[
        "fp_control",
        "broad",
        "component_only",
        "rollback_component",
        "rollback_block_group",
        "rollback_block",
        "rollback_layer",
        "calibration_control",
        "calibration_rollback",
    ]
    quantized_components: list[str] = Field(default_factory=list)
    rollback_components: list[str] = Field(default_factory=list)
    parent_trial_id: str | None = None
    calibration_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    resolved_plan_hash: str = Field(min_length=64, max_length=64)
    seed: int
    notes: list[str] = Field(default_factory=list)


class CandidateEvidenceRef(Contract):
    schema_version: Literal[1] = 1
    trial_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    status: Literal["pending", "measured", "rejected"]


class SensitivityRank(Contract):
    schema_version: Literal[1] = 1
    component: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    broad_error: float = Field(ge=0.0)
    rollback_error: float = Field(ge=0.0)
    recovery: float
    quantized_parameter_coverage: float = Field(ge=0.0, le=1.0)
    evidence_record_id: str = Field(min_length=1)


class CalibrationRankChange(Contract):
    schema_version: Literal[1] = 1
    component: str = Field(min_length=1)
    baseline_trial_id: str = Field(min_length=1)
    control_trial_id: str = Field(min_length=1)
    baseline_error: float = Field(ge=0.0)
    control_error: float = Field(ge=0.0)
    baseline_recovery: float
    control_recovery: float
    baseline_rank: int = Field(gt=0)
    control_rank: int = Field(gt=0)
    rank_delta: int


class CalibrationAblation(Contract):
    schema_version: Literal[1] = 1
    baseline_trial_id: str = Field(min_length=1)
    control_trial_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    baseline_error: float = Field(ge=0.0)
    control_error: float = Field(ge=0.0)
    relative_error_change: float
    rank_metric: str | None = None
    rank_correlation: float | None = Field(default=None, ge=-1.0, le=1.0)
    rank_changes: list[CalibrationRankChange] = Field(default_factory=list)


class ArtifactRef(Contract):
    schema_version: Literal[1] = 1
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)


class TargetCapability(Contract):
    """One target feature probe with an explicit support state."""

    schema_version: Literal[1] = 1
    capability_id: str = Field(min_length=1)
    target: TargetFingerprint
    feature: str = Field(min_length=1)
    precision: PrecisionMode | None = None
    status: SupportStatus
    reason_code: str | None = None
    evidence: list[ArtifactRef] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reason(self) -> TargetCapability:
        if self.status in {"unsupported", "pending"} and not self.reason_code:
            raise ValueError("unsupported or pending target capabilities require reason_code")
        return self


class ShapeProfile(Contract):
    schema_version: Literal[1] = 1
    input_name: str = Field(min_length=1)
    minimum: list[int] = Field(min_length=1)
    optimum: list[int] = Field(min_length=1)
    maximum: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profile(self) -> ShapeProfile:
        if not (len(self.minimum) == len(self.optimum) == len(self.maximum)):
            raise ValueError("shape profile min/opt/max ranks must match")
        for index, (minimum, optimum, maximum) in enumerate(zip(self.minimum, self.optimum, self.maximum, strict=True)):
            if minimum <= 0 or optimum <= 0 or maximum <= 0 or not minimum <= optimum <= maximum:
                raise ValueError(f"shape profile dimension {index} must satisfy 0 < min <= opt <= max")
        return self


class CompilationPlan(Contract):
    """A target compiler invocation plan for one frozen candidate artifact."""

    schema_version: Literal[1] = 1
    plan_id: str = Field(min_length=1)
    compiler: str = Field(min_length=1)
    source_artifact: ArtifactRef
    target: TargetFingerprint
    precision: PrecisionMode
    shape_profiles: list[ShapeProfile] = Field(default_factory=list)
    strongly_typed: bool = False
    workspace_mib: int | None = Field(default=None, gt=0)
    builder_optimization_level: int | None = Field(default=None, ge=0, le=5)
    timing_cache: ArtifactRef | None = None
    flags: list[str] = Field(default_factory=list)
    timing_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source(self) -> CompilationPlan:
        if self.source_artifact.kind not in {"onnx", "candidate-onnx"}:
            raise ValueError("CompilationPlan source_artifact must reference an ONNX candidate")
        return self


class OperatorNodeInfo(Contract):
    schema_version: Literal[1] = 1
    name: str
    op_type: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class OperatorGraphReport(Contract):
    schema_version: Literal[1] = 1
    graph_id: str = Field(min_length=1)
    source: ArtifactRef | None = None
    producer: str = "unknown"
    opset: dict[str, int] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    nodes: list[OperatorNodeInfo] = Field(default_factory=list)
    op_counts: dict[str, int] = Field(default_factory=dict)
    initializer_count: int = Field(ge=0)
    initializer_names: list[str] = Field(default_factory=list)
    dtype_counts: dict[str, int] = Field(default_factory=dict)
    external_data: bool = False
    constant_weight_candidates: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> OperatorGraphReport:
        if sum(self.op_counts.values()) != len(self.nodes):
            raise ValueError("operator report op_counts must sum to node count")
        if self.initializer_count != len(self.initializer_names):
            raise ValueError("operator report initializer_count must match initializer_names")
        return self


class TensorRTLayerReport(Contract):
    schema_version: Literal[1] = 1
    layer_count: int = Field(ge=0)
    dtype_counts: dict[str, int] = Field(default_factory=dict)
    tactic_count: int = Field(ge=0)
    qdq_layer_count: int = Field(ge=0)
    reformat_layer_count: int = Field(ge=0)
    copy_layer_count: int = Field(ge=0)
    fused_layer_count: int = Field(ge=0)
    unsupported_layers: list[str] = Field(default_factory=list)
    source: ArtifactRef | None = None


class CompiledArtifactRef(Contract):
    schema_version: Literal[1] = 1
    artifact_id: str = Field(min_length=1)
    artifact: ArtifactRef
    compiler: str = Field(min_length=1)
    precision: PrecisionMode
    target: TargetFingerprint
    source_sha256: str = Field(min_length=64, max_length=64)
    status: Literal["pending", "measured", "rejected", "unsupported"]
    reason_code: str | None = None

    @model_validator(mode="after")
    def validate_reason(self) -> CompiledArtifactRef:
        if self.status in {"pending", "rejected", "unsupported"} and not self.reason_code:
            raise ValueError("non-measured compiled artifacts require reason_code")
        return self


class CompilerEvidenceRecord(Contract):
    schema_version: Literal[1] = 1
    record_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "rejected", "unsupported"]
    model: ModelSpec
    compilation: CompilationPlan
    artifacts: list[CompiledArtifactRef] = Field(default_factory=list)
    graph: OperatorGraphReport | None = None
    layer_report: TensorRTLayerReport | None = None
    build_time_seconds: float | None = Field(default=None, ge=0.0)
    commands: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    reason_code: str | None = None

    @model_validator(mode="after")
    def validate_status_and_artifacts(self) -> CompilerEvidenceRecord:
        if self.status in {"pending", "rejected", "unsupported"} and not self.reason_code:
            raise ValueError("non-measured compiler evidence requires reason_code")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("compiler evidence artifact IDs must be unique")
        return self


class LatencyDistribution(Contract):
    schema_version: Literal[1] = 1
    count: int = Field(gt=0)
    mean_ms: float = Field(ge=0.0)
    min_ms: float = Field(ge=0.0)
    max_ms: float = Field(ge=0.0)
    std_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_order(self) -> LatencyDistribution:
        if not self.min_ms <= self.p50_ms <= self.p95_ms <= self.p99_ms <= self.max_ms:
            raise ValueError("latency percentiles must satisfy min <= p50 <= p95 <= p99 <= max")
        return self


class BenchmarkProtocol(Contract):
    schema_version: Literal[1] = 1
    protocol_id: str = Field(min_length=1)
    timing_boundary: Literal["build", "engine_stage", "standalone", "server_inference", "client_roundtrip", "closed_loop"]
    warmup: int = Field(ge=0)
    repeat: int = Field(gt=0)
    synchronization: str = Field(min_length=1)
    includes_h2d: bool = False
    includes_d2h: bool = False
    includes_preprocess: bool = False
    includes_postprocess: bool = False
    shape_profiles: list[ShapeProfile] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class StageTimingReport(Contract):
    schema_version: Literal[1] = 1
    report_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "rejected"]
    stage_name: str = Field(min_length=1)
    target: TargetFingerprint
    protocol: BenchmarkProtocol
    latency: LatencyDistribution | None = None
    peak_memory_mib: int | None = Field(default=None, ge=0)
    average_power_w: float | None = Field(default=None, ge=0.0)
    max_temperature_c: float | None = None
    commands: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    reason_code: str | None = None

    @model_validator(mode="after")
    def validate_timing_status(self) -> StageTimingReport:
        if self.status == "measured" and self.latency is None:
            raise ValueError("measured timing reports require latency")
        if self.status != "measured" and not self.reason_code:
            raise ValueError("pending or rejected timing reports require reason_code")
        return self


class DeploymentCandidateManifest(Contract):
    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "rejected", "accepted"]
    model: ModelSpec
    action_schema: ActionSchema | None = None
    target: TargetFingerprint
    source_model: ArtifactRef | None = None
    optimization_plan_hash: str | None = Field(default=None, min_length=64, max_length=64)
    calibration_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    precision_map: dict[str, PrecisionMode] = Field(default_factory=dict)
    input_abi: dict[str, Any] = Field(default_factory=dict)
    output_abi: dict[str, Any] = Field(default_factory=dict)
    compiler_records: list[CompilerEvidenceRecord] = Field(default_factory=list)
    timing_reports: list[StageTimingReport] = Field(default_factory=list)
    evidence_boundary: Literal["target_compiler"] = "target_compiler"
    pi_cpp_integration_status: Literal["pending", "measured", "rejected"] = "pending"
    human_acceptance: Literal["pending", "accepted", "rejected"] = "pending"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_manifest(self) -> DeploymentCandidateManifest:
        record_ids = [record.record_id for record in self.compiler_records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("deployment manifest compiler record IDs must be unique")
        timing_ids = [report.report_id for report in self.timing_reports]
        if len(timing_ids) != len(set(timing_ids)):
            raise ValueError("deployment manifest timing report IDs must be unique")
        if self.status == "accepted" and self.human_acceptance != "accepted":
            raise ValueError("accepted deployment manifests require human_acceptance=accepted")
        if self.status == "measured" and not self.compiler_records:
            raise ValueError("measured deployment manifests require compiler records")
        return self


class CaptureChunkRef(Contract):
    schema_version: Literal[1] = 1
    chunk_id: str = Field(min_length=1)
    sample_ids: list[str] = Field(min_length=1)
    artifact: ArtifactRef


class GoldenCaptureManifest(Contract):
    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1)
    model: ModelSpec
    action_schema: ActionSchema
    holdout_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    capture_specs: list[CaptureSpec] = Field(min_length=1)
    chunks: list[CaptureChunkRef] = Field(min_length=1)
    seed: int
    status: Literal["measured", "rejected"]

    @model_validator(mode="after")
    def validate_capture_lineage(self) -> GoldenCaptureManifest:
        capture_ids = [capture.logical_id for capture in self.capture_specs]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("golden capture logical IDs must be unique")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("golden chunk IDs must be unique")
        sample_ids = [sample_id for chunk in self.chunks for sample_id in chunk.sample_ids]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("golden sample IDs must be unique across chunks")
        return self


class TemporalGoldenCaptureManifest(Contract):
    schema_version: Literal[1] = 1
    manifest_id: str = Field(min_length=1)
    model: ModelSpec
    action_schema: ActionSchema
    holdout_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    capture_specs: list[TemporalCaptureSpec] = Field(min_length=1)
    chunks: list[CaptureChunkRef] = Field(min_length=1)
    modes: list[TemporalMode] = Field(min_length=1)
    seed: int
    status: Literal["measured", "rejected"]

    @model_validator(mode="after")
    def validate_capture_lineage(self) -> TemporalGoldenCaptureManifest:
        capture_ids = [capture.logical_id for capture in self.capture_specs]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("temporal golden capture logical IDs must be unique")
        if self.modes != sorted(set(self.modes)):
            raise ValueError("temporal golden modes must be sorted and unique")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("temporal golden chunk IDs must be unique")
        sequence_ids = [sequence_id for chunk in self.chunks for sequence_id in chunk.sample_ids]
        if len(sequence_ids) != len(set(sequence_ids)):
            raise ValueError("temporal golden sequence IDs must be unique across chunks")
        return self


class EvidenceRecord(Contract):
    schema_version: Literal[1] = 1
    record_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "accepted", "rejected"]
    model: ModelSpec
    target: TargetFingerprint
    plan: OptimizationPlan
    backend: str = Field(min_length=1)
    representation: Literal["fake_quant", "real_quant", "reference_qdq", "fp_control"]
    calibration_fingerprint: str = Field(min_length=64, max_length=64)
    module_coverage: ModuleCoverage
    quantization: QuantizationResult | None = None
    comparison: ComparisonReport
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    timing_boundary: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)
    trial: SensitivityTrial | None = None
    diagnostics: SensitivityDiagnostics | None = None
    temporal_metrics: list[TemporalMetricReport] = Field(default_factory=list)
    rollout_divergence: list[RolloutDivergenceReport] = Field(default_factory=list)
    evaluation: dict[str, float] = Field(default_factory=dict)


class SensitivityStudyRecord(Contract):
    schema_version: Literal[1] = 1
    study_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "rejected"]
    model: ModelSpec
    action_schema: ActionSchema
    module_inventory_sha256: str = Field(min_length=64, max_length=64)
    calibration_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    calibration_manifest_fingerprints: list[str] = Field(default_factory=list)
    holdout_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    golden_manifest: ArtifactRef
    trials: list[SensitivityTrial] = Field(min_length=1)
    candidates: list[CandidateEvidenceRef] = Field(default_factory=list)
    ranking: list[SensitivityRank] = Field(default_factory=list)
    calibration_ablation: list[CalibrationAblation] = Field(default_factory=list)
    evidence_boundary: Literal["source_offline"] = "source_offline"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lineage(self) -> SensitivityStudyRecord:
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("sensitivity study trial IDs must be unique")
        candidate_trial_ids = [candidate.trial_id for candidate in self.candidates]
        if len(candidate_trial_ids) != len(set(candidate_trial_ids)):
            raise ValueError("sensitivity study candidate trial IDs must be unique")
        if unknown := sorted(set(candidate_trial_ids) - set(trial_ids)):
            raise ValueError(f"sensitivity study candidates reference unknown trials: {unknown!r}")
        record_ids = [candidate.record_id for candidate in self.candidates]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("sensitivity study candidate record IDs must be unique")
        if unknown := sorted({rank.evidence_record_id for rank in self.ranking} - set(record_ids)):
            raise ValueError(f"sensitivity ranking references unknown evidence records: {unknown!r}")
        calibration_trial_ids = {
            trial_id for comparison in self.calibration_ablation for trial_id in (comparison.baseline_trial_id, comparison.control_trial_id)
        }
        if unknown := sorted(calibration_trial_ids - set(trial_ids)):
            raise ValueError(f"calibration ablation references unknown trials: {unknown!r}")
        rank_change_trial_ids = {
            trial_id
            for comparison in self.calibration_ablation
            for change in comparison.rank_changes
            for trial_id in (change.baseline_trial_id, change.control_trial_id)
        }
        if unknown := sorted(rank_change_trial_ids - set(trial_ids)):
            raise ValueError(f"calibration rank changes reference unknown trials: {unknown!r}")
        fingerprints = self.calibration_manifest_fingerprints or [self.calibration_manifest_fingerprint]
        if self.calibration_manifest_fingerprint not in fingerprints:
            raise ValueError("primary calibration fingerprint must be included in calibration_manifest_fingerprints")
        return self


class TemporalStudyRecord(Contract):
    schema_version: Literal[1] = 1
    study_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "rejected"]
    model: ModelSpec
    action_schema: ActionSchema
    module_inventory_sha256: str = Field(min_length=64, max_length=64)
    calibration_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    static_control_manifest_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    holdout_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    promotion_reserved_manifest_fingerprint: str = Field(min_length=64, max_length=64)
    golden_manifest: ArtifactRef
    capture_specs: list[TemporalCaptureSpec] = Field(min_length=1)
    trials: list[SensitivityTrial] = Field(min_length=1)
    candidates: list[CandidateEvidenceRef] = Field(default_factory=list)
    ranking: list[SensitivityRank] = Field(default_factory=list)
    calibration_ablation: list[CalibrationAblation] = Field(default_factory=list)
    evidence_boundary: Literal["source_offline_temporal"] = "source_offline_temporal"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_lineage(self) -> TemporalStudyRecord:
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("temporal study trial IDs must be unique")
        candidate_trial_ids = [candidate.trial_id for candidate in self.candidates]
        if len(candidate_trial_ids) != len(set(candidate_trial_ids)):
            raise ValueError("temporal study candidate trial IDs must be unique")
        if unknown := sorted(set(candidate_trial_ids) - set(trial_ids)):
            raise ValueError(f"temporal study candidates reference unknown trials: {unknown!r}")
        record_ids = [candidate.record_id for candidate in self.candidates]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("temporal study candidate record IDs must be unique")
        if unknown := sorted({rank.evidence_record_id for rank in self.ranking} - set(record_ids)):
            raise ValueError(f"temporal ranking references unknown evidence records: {unknown!r}")
        capture_ids = [capture.logical_id for capture in self.capture_specs]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("temporal study capture IDs must be unique")
        calibration_trial_ids = {
            trial_id for comparison in self.calibration_ablation for trial_id in (comparison.baseline_trial_id, comparison.control_trial_id)
        }
        if unknown := sorted(calibration_trial_ids - set(trial_ids)):
            raise ValueError(f"temporal calibration ablation references unknown trials: {unknown!r}")
        if self.calibration_ablation and self.static_control_manifest_fingerprint is None:
            raise ValueError("temporal calibration ablation requires a static control manifest fingerprint")
        return self


def load_plan(path: str | Path) -> OptimizationPlan:
    """Load JSON or YAML and validate it through the single public schema."""

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"optimization plan must be a mapping: {source}")
    return OptimizationPlan.model_validate(data)


def load_compilation_plan(path: str | Path) -> CompilationPlan:
    """Load JSON or YAML and validate it through the target compiler schema."""

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"compilation plan must be a mapping: {source}")
    return CompilationPlan.model_validate(data)


def load_deployment_manifest(path: str | Path) -> DeploymentCandidateManifest:
    """Load a target compiler handoff manifest without importing target runtimes."""

    return DeploymentCandidateManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def fingerprint(value: Any) -> str:
    """Hash a JSON-compatible contract or metadata object deterministically."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")

    def encode_model(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"fingerprint value contains unsupported type {type(item).__name__}")

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=encode_model).encode()
    return hashlib.sha256(encoded).hexdigest()
