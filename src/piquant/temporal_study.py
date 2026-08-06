"""Temporal study execution, ranking, and evidence orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from piquant.contracts import (
    ActionMetric,
    ActionSchema,
    ArtifactRef,
    CalibrationAblation,
    CalibrationRankChange,
    CandidateEvidenceRef,
    CaptureChunkRef,
    ComparisonReport,
    EvidenceRecord,
    ModuleCoverage,
    OptimizationPlan,
    QuantizationResult,
    RolloutDivergenceReport,
    SensitivityRank,
    SensitivityTrial,
    TemporalGoldenCaptureManifest,
    TemporalMetricReport,
    TemporalMode,
    TemporalStudyRecord,
    TensorMetric,
    fingerprint,
)
from piquant.evidence import target_fingerprint
from piquant.interfaces import (
    EvidenceStore,
    QuantizationBackend,
    TemporalAdapterFactory,
    TemporalManifestCalibrationProvider,
    TemporalModelAdapter,
    TemporalTorchQuantizableAdapter,
)
from piquant.temporal import (
    TEMPORAL_SEQUENCE_REFS_KEY,
    RolloutDivergenceAccumulator,
    TemporalStreamingAnalyzer,
    require_temporal_episode_disjoint,
)


def _sequence_id(value: Any) -> str:
    if hasattr(value, "sequence_id"):
        return str(value.sequence_id)
    if isinstance(value, Mapping) and "sequence_id" in value:
        return str(value["sequence_id"])
    raise TypeError(f"temporal sequence reference has no sequence_id: {type(value).__name__}")


def _comparison_from_report(reports: Sequence[TemporalMetricReport], action_schema: ActionSchema) -> ComparisonReport:
    tensors: dict[str, TensorMetric] = {}
    action_metric: ActionMetric | None = None
    for report in reports:
        if report.tensor is not None:
            tensors[f"{report.mode}:{report.capture_id}"] = TensorMetric(
                reference_shape=report.tensor.reference_shape,
                candidate_shape=report.tensor.candidate_shape,
                shape_match=report.tensor.shape_match,
                finite=report.tensor.finite,
                max_abs=report.tensor.max_abs.maximum,
                relative_l2=report.tensor.relative_l2.mean,
                cosine=report.tensor.cosine.mean,
                sqnr_db=report.tensor.sqnr_db.mean,
            )
        if report.action is not None and action_metric is None:
            action_metric = ActionMetric(
                shape_match=report.action.shape_match,
                finite=report.action.finite,
                l1_mean=report.action.l1.mean,
                l2_mean=report.action.l2.mean,
                direction_cosine_mean=report.action.direction_cosine.mean,
                gripper_mismatch_rate=report.action.gripper_mismatch.mean,
            )
    if action_metric is None:
        raise ValueError(f"temporal study requires an action capture; expected schema {action_schema}")
    return ComparisonReport(tensors=tensors, action=action_metric)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class TemporalSensitivityRunner:
    """Run fresh temporal trials against one external chunked FP golden."""

    def __init__(
        self,
        *,
        adapter_factory: TemporalAdapterFactory,
        backend: QuantizationBackend,
        evidence_store: EvidenceStore,
        artifact_root: str | Path,
        command: str = "",
        source_device: str = "source-device",
        seed: int = 0,
        rollout_exceedance_threshold: float = 0.0,
    ) -> None:
        self.adapter_factory = adapter_factory
        self.backend = backend
        self.evidence_store = evidence_store
        self.artifact_root = Path(artifact_root).resolve()
        self.command = command
        self.source_device = source_device
        self.seed = seed
        self.rollout_exceedance_threshold = rollout_exceedance_threshold

    def freeze_golden(
        self,
        holdout: TemporalManifestCalibrationProvider,
        *,
        seed: int,
        modes: Sequence[TemporalMode] = ("teacher_forced", "iterative"),
    ) -> TemporalGoldenCaptureManifest:
        adapter = self.adapter_factory()
        specs = list(adapter.temporal_capture_specs())
        if not specs:
            raise ValueError("temporal adapter must expose at least one capture spec")
        output_dir = self.artifact_root / "temporal-golden"
        if output_dir.exists():
            raise FileExistsError(f"golden directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        chunks: list[CaptureChunkRef] = []
        for mode in modes:
            for index, batch in enumerate(holdout.batches()):
                refs = batch.get(TEMPORAL_SEQUENCE_REFS_KEY)
                if not isinstance(refs, Sequence) or not refs:
                    raise ValueError(f"temporal batch must contain {TEMPORAL_SEQUENCE_REFS_KEY!r}")
                captures = adapter.forward_temporal(batch, [spec.logical_id for spec in specs], mode=mode)
                missing = sorted(set(spec.logical_id for spec in specs) - set(captures))
                if missing:
                    raise ValueError(f"temporal golden capture missing logical ids: {missing}")
                path = output_dir / f"{mode}-batch-{index:04d}.npz"
                arrays = {name: np.asarray(captures[name]) for name in captures}
                np.savez_compressed(path, **cast(Any, arrays))
                chunks.append(
                    CaptureChunkRef(
                        chunk_id=f"{mode}-batch-{index:04d}",
                        sample_ids=[f"{mode}:{_sequence_id(ref)}" for ref in refs],
                        artifact=ArtifactRef(kind=f"temporal-fp-golden-{mode}", path=str(path), sha256=_sha256(path)),
                    )
                )
        manifest = TemporalGoldenCaptureManifest(
            manifest_id=f"temporal-golden-{fingerprint({'model': adapter.spec, 'holdout': holdout.fingerprint, 'seed': seed})[:16]}",
            model=adapter.spec,
            action_schema=adapter.action_schema,
            holdout_manifest_fingerprint=holdout.fingerprint,
            capture_specs=specs,
            chunks=chunks,
            modes=sorted(set(modes)),
            seed=seed,
            status="measured",
        )
        self.evidence_store.write(manifest, str(output_dir / "manifest.json"))
        return manifest

    @staticmethod
    def _fp_coverage(adapter: TemporalModelAdapter) -> ModuleCoverage:
        inventory = [module for module in adapter.module_inventory() if module.quantizable]
        return ModuleCoverage(
            candidate_count=len(inventory),
            matched_count=0,
            excluded_count=0,
            candidate_names=[module.logical_id for module in inventory],
            candidate_parameter_count=sum(module.parameter_count for module in inventory),
        )

    def run_trial(
        self,
        trial: SensitivityTrial,
        plan: OptimizationPlan,
        calibration: TemporalManifestCalibrationProvider,
        holdout: TemporalManifestCalibrationProvider,
        golden: TemporalGoldenCaptureManifest,
    ) -> tuple[EvidenceRecord, CandidateEvidenceRef]:
        if calibration.fingerprint != trial.calibration_manifest_fingerprint:
            raise ValueError("trial calibration fingerprint differs from provider")
        if fingerprint(plan) != trial.resolved_plan_hash:
            raise ValueError("trial resolved_plan_hash differs from OptimizationPlan")
        if plan.seed != trial.seed or golden.seed != trial.seed:
            raise ValueError("trial, plan, and golden seeds must match")
        if holdout.fingerprint != golden.holdout_manifest_fingerprint:
            raise ValueError("holdout fingerprint differs from frozen temporal golden")
        manifest = calibration.manifest
        if plan.calibration.dataset_id != manifest.dataset_id or plan.calibration.dataset_revision != manifest.dataset_revision:
            raise ValueError("OptimizationPlan calibration dataset identity differs from temporal manifest")
        if plan.calibration.sample_count != len(manifest.sequences):
            raise ValueError("OptimizationPlan calibration sample count differs from temporal manifest")
        if set(plan.calibration.stages) != {sequence.stage for sequence in manifest.sequences}:
            raise ValueError("OptimizationPlan calibration stages differ from temporal manifest")
        adapter = self.adapter_factory()
        if trial.kind == "fp_control":
            if plan.backend != "none" or plan.quant_format != "none" or plan.representation != "fp_control":
                raise ValueError("fp_control trial requires backend=none, quant_format=none, and representation=fp_control")
            quantization = QuantizationResult(
                backend="fp-control",
                representation="fp_control",
                status="measured",
                quant_format="none",
                module_coverage=self._fp_coverage(adapter),
            )
        else:
            if plan.backend != self.backend.name or plan.quant_format != "int8" or plan.representation != "fake_quant":
                raise ValueError("quantized temporal trial requires the injected INT8 fake-quant backend")
            if not isinstance(adapter, TemporalTorchQuantizableAdapter):
                raise TypeError("quantized temporal trial requires a TemporalTorchQuantizableAdapter")
            model, quantization = self.backend.quantize(adapter, plan, calibration)
            adapter = adapter.with_backend_model(model)
        specs = list(adapter.temporal_capture_specs())
        if specs != golden.capture_specs:
            raise ValueError("candidate temporal capture specs differ from frozen golden")
        modes = golden.modes
        reports: list[TemporalMetricReport] = []
        rollout_reports: list[RolloutDivergenceReport] = []
        action_specs = [spec for spec in specs if spec.kind == "action"]
        if len(action_specs) != 1:
            raise ValueError("temporal study requires exactly one final action capture")
        rollout_specs = [spec for spec in specs if spec.kind == "rollout"]
        if len(rollout_specs) > 1:
            raise ValueError("temporal study supports at most one rollout action capture")
        latent_specs = [spec for spec in specs if spec.kind == "latent"]
        for mode in modes:
            analyzer = TemporalStreamingAnalyzer(seed=self.seed)
            rollout = (
                None
                if not rollout_specs
                else RolloutDivergenceAccumulator(
                    rollout_specs[0],
                    latent_specs,
                    mode=mode,
                    exceedance_threshold=self.rollout_exceedance_threshold,
                    seed=self.seed,
                )
            )
            chunks = [chunk for chunk in golden.chunks if chunk.chunk_id.startswith(f"{mode}-")]
            for chunk, batch in zip(chunks, holdout.batches(), strict=True):
                refs = batch.get(TEMPORAL_SEQUENCE_REFS_KEY)
                if not isinstance(refs, Sequence) or [f"{mode}:{_sequence_id(ref)}" for ref in refs] != chunk.sample_ids:
                    raise ValueError(f"temporal golden chunk sequence order differs for {chunk.chunk_id!r}")
                with np.load(chunk.artifact.path, allow_pickle=False) as archive:
                    reference = {name: np.asarray(archive[name]) for name in archive.files}
                candidate = adapter.forward_temporal(batch, [spec.logical_id for spec in specs], mode=mode)
                analyzer.add(
                    reference,
                    candidate,
                    [ref.model_dump(mode="json") for ref in refs],
                    specs,
                    mode=mode,
                    action_schema=adapter.action_schema,
                )
                if rollout is not None:
                    rollout.add(reference, candidate, adapter.action_schema)
            reports.extend(analyzer.finalize())
            if rollout is not None:
                rollout_reports.append(rollout.finalize())
        comparison = _comparison_from_report(reports, adapter.action_schema)
        evidence_path = self.artifact_root / "candidates" / trial.trial_id / "evidence.json"
        if evidence_path.exists():
            raise FileExistsError(f"candidate evidence already exists: {evidence_path}")
        record = EvidenceRecord(
            record_id=f"{trial.trial_id}-{fingerprint({'trial': trial, 'model': adapter.spec})[:16]}",
            status="measured",
            model=adapter.spec,
            target=target_fingerprint(device=self.source_device),
            plan=plan,
            backend=quantization.backend,
            representation=quantization.representation,
            calibration_fingerprint=calibration.fingerprint,
            module_coverage=quantization.module_coverage,
            quantization=quantization,
            comparison=comparison,
            commands=[] if not self.command else [self.command],
            timing_boundary="source/offline temporal diagnostics; no target latency",
            notes=["Temporal evidence is not TensorRT, hardware latency, closed-loop success, or accepted deployment evidence"],
            trial=trial,
            temporal_metrics=reports,
            rollout_divergence=rollout_reports,
        )
        self.evidence_store.write(record, str(evidence_path))
        evidence_ref = CandidateEvidenceRef(
            trial_id=trial.trial_id,
            record_id=record.record_id,
            path=str(evidence_path),
            sha256=_sha256(evidence_path),
            status="measured",
        )
        return record, evidence_ref

    @staticmethod
    def rank_records(records: Sequence[EvidenceRecord]) -> list[SensitivityRank]:
        broad = next((record for record in records if record.trial is not None and record.trial.kind == "broad"), None)
        if broad is None:
            return []
        broad_action = next(
            (report for report in broad.temporal_metrics if report.action is not None and report.mode == "iterative"),
            next((report for report in broad.temporal_metrics if report.action is not None), None),
        )
        if broad_action is None or broad_action.action is None or broad_action.action.l2.mean is None:
            return []
        broad_error = broad_action.action.l2.mean
        ranking: list[SensitivityRank] = []
        for record in records:
            if record.trial is None or record.trial.kind not in {
                "rollback_component",
                "rollback_block_group",
                "rollback_block",
                "rollback_layer",
            }:
                continue
            action_report = next(
                (report for report in record.temporal_metrics if report.action is not None and report.mode == "iterative"),
                next((report for report in record.temporal_metrics if report.action is not None), None),
            )
            if action_report is None or action_report.action is None or action_report.action.l2.mean is None:
                continue
            rollback_error = action_report.action.l2.mean
            coverage = record.module_coverage
            denominator = coverage.candidate_parameter_count or coverage.candidate_count or 1
            numerator = coverage.matched_parameter_count or coverage.matched_count
            ranking.append(
                SensitivityRank(
                    component=",".join(record.trial.rollback_components),
                    metric="temporal_action_l2_mean",
                    broad_error=broad_error,
                    rollback_error=rollback_error,
                    recovery=(broad_error - rollback_error) / max(broad_error, 1e-12),
                    quantized_parameter_coverage=numerator / denominator,
                    evidence_record_id=record.record_id,
                )
            )
        ranking.sort(key=lambda item: (-item.recovery, item.component))
        return ranking

    @staticmethod
    def _iterative_action_error(record: EvidenceRecord) -> float | None:
        reports = [report for report in record.temporal_metrics if report.action is not None]
        report = next((item for item in reports if item.mode == "iterative"), reports[0] if reports else None)
        return None if report is None or report.action is None else report.action.l2.mean

    @classmethod
    def calibration_ablation(cls, records: Sequence[EvidenceRecord]) -> list[CalibrationAblation]:
        broad = next((record for record in records if record.trial is not None and record.trial.kind == "broad"), None)
        if broad is None or broad.trial is None:
            return []
        baseline_error = cls._iterative_action_error(broad)
        if baseline_error is None:
            return []

        def rollbacks(kind: str, parent: str, calibration: str, parent_error: float) -> dict[str, tuple[EvidenceRecord, float, float]]:
            result: dict[str, tuple[EvidenceRecord, float, float]] = {}
            for record in records:
                trial = record.trial
                if trial is None or trial.kind != kind or trial.parent_trial_id != parent:
                    continue
                if trial.calibration_manifest_fingerprint != calibration or len(trial.rollback_components) != 1:
                    raise ValueError(f"temporal calibration rollback {trial.trial_id!r} does not match its parent")
                error = cls._iterative_action_error(record)
                if error is None:
                    raise ValueError(f"temporal calibration rollback {trial.trial_id!r} is missing iterative action L2")
                component = trial.rollback_components[0]
                if component in result:
                    raise ValueError(f"duplicate temporal calibration ranking component {component!r}")
                result[component] = (record, error, (parent_error - error) / max(parent_error, 1e-12))
            return result

        baseline_rollbacks = rollbacks(
            "rollback_component", broad.trial.trial_id, broad.trial.calibration_manifest_fingerprint, baseline_error
        )
        comparisons: list[CalibrationAblation] = []
        for control in records:
            trial = control.trial
            if trial is None or trial.kind != "calibration_control":
                continue
            control_error = cls._iterative_action_error(control)
            if control_error is None:
                continue
            control_rollbacks = rollbacks("calibration_rollback", trial.trial_id, trial.calibration_manifest_fingerprint, control_error)
            if control_rollbacks and set(control_rollbacks) != set(baseline_rollbacks):
                raise ValueError("temporal calibration control must cover the same rollback components as broad")
            baseline_order = sorted(baseline_rollbacks, key=lambda name: (-baseline_rollbacks[name][2], name))
            control_order = sorted(control_rollbacks, key=lambda name: (-control_rollbacks[name][2], name))
            baseline_ranks = {name: index + 1 for index, name in enumerate(baseline_order)}
            control_ranks = {name: index + 1 for index, name in enumerate(control_order)}
            rank_changes = [
                CalibrationRankChange(
                    component=name,
                    baseline_trial_id=baseline_rollbacks[name][0].trial.trial_id,  # type: ignore[union-attr]
                    control_trial_id=control_rollbacks[name][0].trial.trial_id,  # type: ignore[union-attr]
                    baseline_error=baseline_rollbacks[name][1],
                    control_error=control_rollbacks[name][1],
                    baseline_recovery=baseline_rollbacks[name][2],
                    control_recovery=control_rollbacks[name][2],
                    baseline_rank=baseline_ranks[name],
                    control_rank=control_ranks[name],
                    rank_delta=control_ranks[name] - baseline_ranks[name],
                )
                for name in baseline_order
                if name in control_rollbacks
            ]
            correlation = None
            if len(rank_changes) > 1:
                squared_delta = sum(change.rank_delta**2 for change in rank_changes)
                count = len(rank_changes)
                correlation = 1.0 - 6.0 * squared_delta / (count * (count**2 - 1))
            comparisons.append(
                CalibrationAblation(
                    baseline_trial_id=broad.trial.trial_id,
                    control_trial_id=trial.trial_id,
                    metric="iterative_action_l2_mean",
                    baseline_error=baseline_error,
                    control_error=control_error,
                    relative_error_change=(control_error - baseline_error) / max(baseline_error, 1e-12),
                    rank_metric="iterative_action_l2_recovery" if rank_changes else None,
                    rank_correlation=correlation,
                    rank_changes=rank_changes,
                )
            )
        return comparisons

    def finalize_study(
        self,
        *,
        study_id: str,
        trials: Sequence[SensitivityTrial],
        records: Sequence[EvidenceRecord],
        references: Sequence[CandidateEvidenceRef],
        calibration: TemporalManifestCalibrationProvider,
        static_control: TemporalManifestCalibrationProvider | None,
        holdout: TemporalManifestCalibrationProvider,
        promotion_reserved: TemporalManifestCalibrationProvider,
        golden: TemporalGoldenCaptureManifest,
    ) -> TemporalStudyRecord:
        if not trials or len(trials) != len(records) or len(records) != len(references):
            raise ValueError("temporal study requires aligned non-empty trials, records, and references")
        study_path = self.artifact_root / "study.json"
        if study_path.exists():
            raise FileExistsError(f"temporal study evidence already exists: {study_path}")
        manifests = [calibration.manifest, holdout.manifest, promotion_reserved.manifest]
        if static_control is not None:
            manifests.append(static_control.manifest)
        require_temporal_episode_disjoint(*manifests)
        if calibration.manifest.split != "calibration":
            raise ValueError("temporal study calibration manifest must use split=calibration")
        if holdout.manifest.split != "diagnostic_holdout":
            raise ValueError("temporal study holdout manifest must use split=diagnostic_holdout")
        if promotion_reserved.manifest.split != "promotion_reserved":
            raise ValueError("temporal study promotion manifest must use split=promotion_reserved")
        if static_control is not None and static_control.manifest.split != "static_control":
            raise ValueError("temporal study static control manifest must use split=static_control")
        if static_control is None and any(trial.kind == "calibration_control" for trial in trials):
            raise ValueError("temporal calibration control requires static_control_fingerprint")
        study = TemporalStudyRecord(
            study_id=study_id,
            status="measured",
            model=records[0].model,
            action_schema=golden.action_schema,
            module_inventory_sha256=fingerprint(self.adapter_factory().module_inventory()),
            calibration_manifest_fingerprint=calibration.fingerprint,
            static_control_manifest_fingerprint=None if static_control is None else static_control.fingerprint,
            holdout_manifest_fingerprint=holdout.fingerprint,
            promotion_reserved_manifest_fingerprint=promotion_reserved.fingerprint,
            golden_manifest=ArtifactRef(
                kind="temporal-golden-manifest",
                path=str(self.artifact_root / "temporal-golden" / "manifest.json"),
                sha256=_sha256(self.artifact_root / "temporal-golden" / "manifest.json"),
            ),
            capture_specs=golden.capture_specs,
            trials=list(trials),
            candidates=list(references),
            ranking=self.rank_records(records),
            calibration_ablation=self.calibration_ablation(records),
            notes=[
                "Candidate metrics and rollout divergence remain in each EvidenceRecord; the study stores only lineage and ranking",
                "Source/offline temporal evidence only; target compiler, server/client, closed-loop, and human promotion are pending",
            ],
        )
        if study.golden_manifest.sha256 != _sha256(Path(study.golden_manifest.path)):
            raise ValueError("temporal golden manifest identity differs while finalizing study")
        self.evidence_store.write(study, str(study_path))
        return study
