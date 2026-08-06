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
    torch_version: str = "unavailable"
    modelopt_version: str = "unavailable"
    onnx_version: str = "unavailable"
    onnxruntime_version: str = "unavailable"


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


def load_plan(path: str | Path) -> OptimizationPlan:
    """Load JSON or YAML and validate it through the single public schema."""

    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"optimization plan must be a mapping: {source}")
    return OptimizationPlan.model_validate(data)


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
