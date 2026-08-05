"""Versioned, serializable data contracts shared by pi.quant components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


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
    quant_format: Literal["int8"] = "int8"
    representation: Literal["fake_quant", "real_quant"] = "fake_quant"
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


class QuantizationResult(Contract):
    schema_version: Literal[1] = 1
    backend: str = Field(min_length=1)
    representation: Literal["fake_quant", "real_quant", "reference_qdq"]
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


class ArtifactRef(Contract):
    schema_version: Literal[1] = 1
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)


class EvidenceRecord(Contract):
    schema_version: Literal[1] = 1
    record_id: str = Field(min_length=1)
    status: Literal["pending", "measured", "accepted", "rejected"]
    model: ModelSpec
    target: TargetFingerprint
    plan: OptimizationPlan
    backend: str = Field(min_length=1)
    representation: Literal["fake_quant", "real_quant", "reference_qdq"]
    calibration_fingerprint: str = Field(min_length=64, max_length=64)
    module_coverage: ModuleCoverage
    comparison: ComparisonReport
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    timing_boundary: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


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
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()
