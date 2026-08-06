from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from piquant.analysis import StreamingDiagnosticAnalyzer
from piquant.backends import ModelOptBackend
from piquant.contracts import (
    ActionSchema,
    ArtifactRef,
    CalibrationAblation,
    CalibrationRankChange,
    CandidateEvidenceRef,
    CaptureChunkRef,
    CaptureSpec,
    EvidenceRecord,
    GoldenCaptureManifest,
    ModelSpec,
    ModuleDescriptor,
    OptimizationPlan,
    SensitivityStudyRecord,
    SensitivityTrial,
    fingerprint,
)
from piquant.evidence import JsonEvidenceStore, validate_sensitivity_study
from piquant.matching import require_matches, select_modules
from piquant.sensitivity import SensitivityRunner


def test_module_selection_reports_include_and_exclude() -> None:
    coverage = select_modules(
        {"vision.patch": object(), "action_encoder": object(), "head": object()},
        ["*"],
        ["head"],
    )
    assert coverage.candidate_count == 3
    assert coverage.matched_names == ["action_encoder", "vision.patch"]
    assert coverage.excluded_names == ["head"]


def test_zero_match_fails_fast() -> None:
    coverage = select_modules({"head": object()}, ["vision.*"], [])
    with pytest.raises(ValueError, match="matched zero"):
        require_matches(coverage, ["vision.*"])


def test_evidence_store_round_trip(tmp_path: Path, synthetic_record: EvidenceRecord) -> None:
    path = tmp_path / "evidence.json"
    store = JsonEvidenceStore()
    store.write(synthetic_record, str(path))
    loaded = store.read(str(path))
    assert loaded.record_id == synthetic_record.record_id
    assert loaded.module_coverage.matched_count == synthetic_record.module_coverage.matched_count


def test_fingerprint_serializes_contracts_nested_in_metadata() -> None:
    model = ModelSpec(model_id="nested", family="test", framework="test", action_dim=7, action_horizon=1)
    assert fingerprint({"model": model, "seed": 7}) == fingerprint({"model": model.model_dump(mode="json"), "seed": 7})


def test_study_validation_hashes_golden_capture_chunks(tmp_path: Path, synthetic_record: EvidenceRecord) -> None:
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    capture_path = tmp_path / "capture.bin"
    capture_path.write_bytes(b"golden")
    action_schema = ActionSchema(
        model_action_dim=6,
        output_action_dim=6,
        horizon=1,
        denoise_steps=1,
        translation_indices=[0, 1, 2],
        rotation_indices=[3, 4],
        gripper_index=5,
        postprocess="identity",
    )
    golden = GoldenCaptureManifest(
        manifest_id="golden",
        model=synthetic_record.model,
        action_schema=action_schema,
        holdout_manifest_fingerprint="0" * 64,
        capture_specs=[CaptureSpec(logical_id="action", backend_path="$action", component="head", kind="action")],
        chunks=[
            CaptureChunkRef(
                chunk_id="batch-0000",
                sample_ids=["sample-0"],
                artifact=ArtifactRef(kind="capture", path=str(capture_path), sha256=sha256(capture_path)),
            )
        ],
        seed=synthetic_record.plan.seed,
        status="measured",
    )
    store = JsonEvidenceStore()
    golden_path = tmp_path / "golden.json"
    store.write(golden, str(golden_path))
    trial = SensitivityTrial(
        trial_id="candidate",
        kind="broad",
        quantized_components=["all"],
        calibration_manifest_fingerprint=synthetic_record.calibration_fingerprint,
        resolved_plan_hash=fingerprint(synthetic_record.plan),
        seed=synthetic_record.plan.seed,
    )
    record = synthetic_record.model_copy(update={"trial": trial})
    candidate_path = tmp_path / "candidate.json"
    store.write(record, str(candidate_path))
    study = SensitivityStudyRecord(
        study_id="study",
        status="measured",
        model=synthetic_record.model,
        action_schema=action_schema,
        module_inventory_sha256="1" * 64,
        calibration_manifest_fingerprint=synthetic_record.calibration_fingerprint,
        holdout_manifest_fingerprint=golden.holdout_manifest_fingerprint,
        golden_manifest=ArtifactRef(kind="golden", path=str(golden_path), sha256=sha256(golden_path)),
        trials=[trial],
        candidates=[
            CandidateEvidenceRef(
                trial_id=trial.trial_id,
                record_id=record.record_id,
                path=str(candidate_path),
                sha256=sha256(candidate_path),
                status="measured",
            )
        ],
    )
    study_path = tmp_path / "study.json"
    store.write(study, str(study_path))
    assert validate_sensitivity_study(study_path)["candidate_count"] == 1
    capture_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="golden capture chunk SHA256 differs"):
        validate_sensitivity_study(study_path)


class _SemanticAdapter:
    spec = ModelSpec(model_id="semantic", family="test", framework="test", action_dim=7, action_horizon=1)
    action_schema = ActionSchema(
        model_action_dim=7,
        output_action_dim=7,
        horizon=1,
        denoise_steps=1,
        translation_indices=[0, 1, 2],
        rotation_indices=[3, 4, 5],
        gripper_index=6,
        postprocess="identity",
    )

    def named_modules(self) -> dict[str, object]:
        return {"backend.vision": object(), "backend.head": object(), "backend.norm": object()}

    def module_inventory(self) -> list[ModuleDescriptor]:
        return [
            ModuleDescriptor(
                logical_id="vision.block.00.mlp.up",
                backend_path="backend.vision",
                component="vision",
                block_index=0,
                op_family="linear",
                parameter_count=100,
                quantizable=True,
            ),
            ModuleDescriptor(
                logical_id="action.head",
                backend_path="backend.head",
                component="action_head",
                op_family="linear",
                parameter_count=20,
                quantizable=True,
            ),
            ModuleDescriptor(
                logical_id="action.norm",
                backend_path="backend.norm",
                component="action_norm",
                op_family="linear",
                parameter_count=5,
                quantizable=False,
            ),
        ]

    def capture_specs(self) -> list[CaptureSpec]:
        return [CaptureSpec(logical_id="action", backend_path="$action", component="action_head", kind="action")]

    def fresh(self) -> _SemanticAdapter:
        return self

    def forward(self, batch: dict[str, object], capture_points: list[str]) -> dict[str, object]:
        return {}


def test_semantic_selection_resolves_logical_ids_to_backend_paths() -> None:
    plan = OptimizationPlan(
        plan_id="semantic",
        backend="modelopt",
        module_include=["vision.*", "action.*"],
        module_exclude=["action.head"],
        calibration={
            "dataset_id": "test",
            "sample_count": 1,
            "seed": 1,
            "stages": ["test"],
            "input_fields": ["input"],
        },
        seed=1,
        capture_points=["action"],
        timing_boundary="offline",
    )
    coverage = ModelOptBackend().inspect(_SemanticAdapter(), plan)
    assert coverage.candidate_count == 2
    assert coverage.candidate_parameter_count == 120
    assert coverage.matched_logical_ids == ["vision.block.00.mlp.up"]
    assert coverage.resolved_backend_names == ["backend.vision"]
    assert coverage.matched_parameter_count == 100


def test_calibration_ablation_compares_matched_component_rankings(synthetic_record: EvidenceRecord) -> None:
    action_schema = ActionSchema(
        model_action_dim=6,
        output_action_dim=6,
        horizon=1,
        denoise_steps=1,
        translation_indices=[0, 1, 2],
        rotation_indices=[3, 4],
        gripper_index=5,
        postprocess="identity",
    )

    def trial(trial_id: str, kind: str, fingerprint_value: str, *, parent: str | None = None, component: str | None = None):
        return SensitivityTrial(
            trial_id=trial_id,
            kind=kind,  # type: ignore[arg-type]
            rollback_components=[] if component is None else [component],
            parent_trial_id=parent,
            calibration_manifest_fingerprint=fingerprint_value,
            resolved_plan_hash="0" * 64,
            seed=1,
        )

    def record(candidate_trial: SensitivityTrial, error: float) -> EvidenceRecord:
        analyzer = StreamingDiagnosticAnalyzer(seed=1)
        reference = {"action": np.zeros((1, 1, 6), dtype=np.float32)}
        candidate = {"action": reference["action"].copy()}
        candidate["action"][0, 0, 0] = error
        analyzer.add(
            reference,
            candidate,
            [{"stage": "test", "timestep": 0.5}],
            action_name="action",
            action_schema=action_schema,
        )
        return synthetic_record.model_copy(
            update={
                "record_id": candidate_trial.trial_id,
                "trial": candidate_trial,
                "calibration_fingerprint": candidate_trial.calibration_manifest_fingerprint,
                "diagnostics": analyzer.finalize(),
            }
        )

    baseline_fingerprint = "a" * 64
    control_fingerprint = "b" * 64
    records = [
        record(trial("broad", "broad", baseline_fingerprint), 10.0),
        record(trial("rollback-a", "rollback_component", baseline_fingerprint, parent="broad", component="a"), 5.0),
        record(trial("rollback-b", "rollback_component", baseline_fingerprint, parent="broad", component="b"), 8.0),
        record(trial("random", "calibration_control", control_fingerprint, parent="broad"), 20.0),
        record(trial("random-a", "calibration_rollback", control_fingerprint, parent="random", component="a"), 19.0),
        record(trial("random-b", "calibration_rollback", control_fingerprint, parent="random", component="b"), 10.0),
    ]

    comparison = SensitivityRunner.calibration_ablation(records)[0]
    assert comparison.relative_error_change == pytest.approx(1.0)
    assert comparison.rank_correlation == pytest.approx(-1.0)
    changes = {change.component: change for change in comparison.rank_changes}
    assert (changes["a"].baseline_rank, changes["a"].control_rank, changes["a"].rank_delta) == (1, 2, 1)
    assert (changes["b"].baseline_rank, changes["b"].control_rank, changes["b"].rank_delta) == (2, 1, -1)


def test_study_lineage_rejects_unknown_calibration_trials(synthetic_record: EvidenceRecord) -> None:
    action_schema = ActionSchema(
        model_action_dim=6,
        output_action_dim=6,
        horizon=1,
        denoise_steps=1,
        translation_indices=[0, 1, 2],
        rotation_indices=[3, 4],
        gripper_index=5,
        postprocess="identity",
    )
    broad = SensitivityTrial(
        trial_id="broad",
        kind="broad",
        calibration_manifest_fingerprint="a" * 64,
        resolved_plan_hash="0" * 64,
        seed=1,
    )

    def build_study(ablation: CalibrationAblation) -> SensitivityStudyRecord:
        return SensitivityStudyRecord(
            study_id="study",
            status="measured",
            model=synthetic_record.model,
            action_schema=action_schema,
            module_inventory_sha256="1" * 64,
            calibration_manifest_fingerprint="a" * 64,
            holdout_manifest_fingerprint="b" * 64,
            golden_manifest=ArtifactRef(kind="golden", path="/golden.json", sha256="c" * 64),
            trials=[broad],
            calibration_ablation=[ablation],
        )

    with pytest.raises(ValueError, match="calibration ablation references unknown trials"):
        build_study(
            CalibrationAblation(
                baseline_trial_id="missing",
                control_trial_id="broad",
                metric="action_l2_mean",
                baseline_error=1.0,
                control_error=1.0,
                relative_error_change=0.0,
            )
        )
    with pytest.raises(ValueError, match="calibration rank changes reference unknown trials"):
        build_study(
            CalibrationAblation(
                baseline_trial_id="broad",
                control_trial_id="broad",
                metric="action_l2_mean",
                baseline_error=1.0,
                control_error=1.0,
                relative_error_change=0.0,
                rank_changes=[
                    CalibrationRankChange(
                        component="component",
                        baseline_trial_id="missing",
                        control_trial_id="broad",
                        baseline_error=1.0,
                        control_error=1.0,
                        baseline_recovery=0.0,
                        control_recovery=0.0,
                        baseline_rank=1,
                        control_rank=1,
                        rank_delta=0,
                    )
                ],
            )
        )
