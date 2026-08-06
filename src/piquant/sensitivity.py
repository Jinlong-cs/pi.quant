"""Reusable golden-capture and hierarchical quantization sensitivity orchestration."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from piquant.calibration import SAMPLE_REFS_KEY
from piquant.contracts import (
    ActionMetric,
    ArtifactRef,
    CalibrationAblation,
    CalibrationRankChange,
    CandidateEvidenceRef,
    CaptureChunkRef,
    ComparisonReport,
    EvidenceRecord,
    GoldenCaptureManifest,
    ModuleCoverage,
    OptimizationPlan,
    QuantizationResult,
    SensitivityDiagnostics,
    SensitivityRank,
    SensitivityStudyRecord,
    SensitivityTrial,
    TensorMetric,
    fingerprint,
)
from piquant.evidence import target_fingerprint
from piquant.interfaces import (
    AdapterFactory,
    EvidenceStore,
    ManifestCalibrationProvider,
    QuantizationBackend,
    StreamingNumericalAnalyzer,
    TaskEvaluator,
    TorchQuantizableAdapter,
)

AnalyzerFactory = Callable[[], StreamingNumericalAnalyzer]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_metadata(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = batch.get(SAMPLE_REFS_KEY)
    if not isinstance(refs, Sequence) or not refs:
        raise ValueError(f"calibration batch must contain non-empty {SAMPLE_REFS_KEY!r}")
    return [ref.model_dump(mode="json") for ref in refs]


def _legacy_comparison(diagnostics: SensitivityDiagnostics) -> ComparisonReport:
    """Populate the v0.1 aggregate field while v0.2 retains full paired distributions."""

    bucket = diagnostics.overall
    tensors = {
        name: TensorMetric(
            reference_shape=summary.reference_shape,
            candidate_shape=summary.candidate_shape,
            shape_match=summary.shape_match,
            finite=summary.finite,
            max_abs=summary.max_abs.maximum,
            relative_l2=summary.relative_l2.mean,
            cosine=summary.cosine.mean,
            sqnr_db=summary.sqnr_db.mean,
        )
        for name, summary in bucket.tensors.items()
    }
    action = bucket.action
    return ComparisonReport(
        tensors=tensors,
        action=ActionMetric(
            shape_match=action.shape_match,
            finite=action.finite,
            l1_mean=action.l1.mean,
            l2_mean=action.l2.mean,
            direction_cosine_mean=action.direction_cosine.mean,
            gripper_mismatch_rate=action.gripper_mismatch.mean,
        ),
    )


def _coverage_fraction(coverage: ModuleCoverage) -> float:
    if coverage.candidate_parameter_count:
        return coverage.matched_parameter_count / coverage.candidate_parameter_count
    if coverage.candidate_count:
        return coverage.matched_count / coverage.candidate_count
    return 0.0


class SensitivityRunner:
    """Run fresh-model trials against one chunked FP golden without retaining large activations in memory."""

    def __init__(
        self,
        *,
        adapter_factory: AdapterFactory,
        backend: QuantizationBackend,
        analyzer_factory: AnalyzerFactory,
        evidence_store: EvidenceStore,
        artifact_root: str | Path,
        action_name: str,
        flow_name: str | None = None,
        evaluator: TaskEvaluator | None = None,
        command: str = "",
        source_device: str = "source-device",
    ) -> None:
        self.adapter_factory = adapter_factory
        self.backend = backend
        self.analyzer_factory = analyzer_factory
        self.evidence_store = evidence_store
        self.artifact_root = Path(artifact_root).resolve()
        self.action_name = action_name
        self.flow_name = flow_name
        self.evaluator = evaluator
        self.command = command
        self.source_device = source_device

    def freeze_golden(self, holdout: ManifestCalibrationProvider, *, seed: int) -> GoldenCaptureManifest:
        adapter = self.adapter_factory()
        capture_specs = list(adapter.capture_specs())
        capture_points = [capture.logical_id for capture in capture_specs]
        manifest_id = f"golden-{fingerprint({'model': adapter.spec, 'holdout': holdout.fingerprint, 'seed': seed})[:16]}"
        chunks: list[CaptureChunkRef] = []
        output_dir = self.artifact_root / "golden"
        if output_dir.exists():
            raise FileExistsError(f"golden directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        for index, batch in enumerate(holdout.batches()):
            metadata = _sample_metadata(batch)
            captures = adapter.forward(batch, capture_points)
            missing = sorted(set(capture_points) - set(captures))
            if missing:
                raise ValueError(f"golden capture missing logical ids: {missing}")
            path = output_dir / f"batch-{index:04d}.npz"
            capture_arrays: dict[str, Any] = {name: np.asarray(value) for name, value in captures.items()}
            np.savez_compressed(path, **capture_arrays)
            chunks.append(
                CaptureChunkRef(
                    chunk_id=f"batch-{index:04d}",
                    sample_ids=[str(item["sample_id"]) for item in metadata],
                    artifact=ArtifactRef(kind="fp-golden-chunk", path=str(path), sha256=_sha256(path)),
                )
            )
        manifest = GoldenCaptureManifest(
            manifest_id=manifest_id,
            model=adapter.spec,
            action_schema=adapter.action_schema,
            holdout_manifest_fingerprint=holdout.fingerprint,
            capture_specs=capture_specs,
            chunks=chunks,
            seed=seed,
            status="measured",
        )
        self.evidence_store.write(manifest, str(output_dir / "manifest.json"))
        return manifest

    def _candidate_adapter(
        self,
        trial: SensitivityTrial,
        plan: OptimizationPlan,
        calibration: ManifestCalibrationProvider,
    ) -> tuple[Any, QuantizationResult]:
        adapter = self.adapter_factory()
        if trial.kind == "fp_control":
            inventory = [module for module in adapter.module_inventory() if module.quantizable]
            coverage = ModuleCoverage(
                candidate_count=len(inventory),
                matched_count=0,
                excluded_count=0,
                candidate_names=[module.backend_path for module in inventory],
                candidate_parameter_count=sum(module.parameter_count for module in inventory),
            )
            return adapter, QuantizationResult(
                backend="fp-control",
                representation="fp_control",
                status="measured",
                quant_format="none",
                module_coverage=coverage,
            )
        if not isinstance(adapter, TorchQuantizableAdapter):
            raise TypeError("selected quantization backend requires a TorchQuantizableAdapter")
        model, result = self.backend.quantize(adapter, plan, calibration)
        return adapter.with_backend_model(model), result

    def run_trial(
        self,
        trial: SensitivityTrial,
        plan: OptimizationPlan,
        calibration: ManifestCalibrationProvider,
        holdout: ManifestCalibrationProvider,
        golden: GoldenCaptureManifest,
    ) -> tuple[EvidenceRecord, CandidateEvidenceRef]:
        if calibration.fingerprint != trial.calibration_manifest_fingerprint:
            raise ValueError("trial calibration fingerprint differs from provider")
        if fingerprint(plan) != trial.resolved_plan_hash:
            raise ValueError("trial resolved_plan_hash differs from OptimizationPlan")
        if plan.seed != trial.seed or plan.calibration.seed != trial.seed or golden.seed != trial.seed:
            raise ValueError("trial, plan, calibration spec, and golden seeds must match")
        manifest = calibration.manifest
        if plan.calibration.dataset_id != manifest.dataset_id or plan.calibration.dataset_revision != manifest.dataset_revision:
            raise ValueError("OptimizationPlan calibration dataset identity differs from manifest")
        if plan.calibration.sample_count != len(manifest.samples):
            raise ValueError("OptimizationPlan calibration sample count differs from manifest")
        if set(plan.calibration.stages) != {sample.stage for sample in manifest.samples}:
            raise ValueError("OptimizationPlan calibration stages differ from manifest")
        if trial.kind == "fp_control":
            if plan.backend != "none" or plan.quant_format != "none" or plan.representation != "fp_control":
                raise ValueError("fp_control trial requires backend=none, quant_format=none, and representation=fp_control")
        elif plan.backend != self.backend.name or plan.quant_format != "int8" or plan.representation != "fake_quant":
            raise ValueError("quantized sensitivity trial requires the injected backend and an INT8 fake-quant plan")
        if holdout.fingerprint != golden.holdout_manifest_fingerprint:
            raise ValueError("holdout fingerprint differs from frozen golden")
        capture_points = [capture.logical_id for capture in golden.capture_specs]
        if set(plan.capture_points) != set(capture_points):
            raise ValueError("OptimizationPlan capture points differ from frozen golden")
        evidence_path = self.artifact_root / "candidates" / trial.trial_id / "evidence.json"
        if evidence_path.exists():
            raise FileExistsError(f"candidate evidence already exists: {evidence_path}")
        adapter, quantization = self._candidate_adapter(trial, plan, calibration)
        analyzer = self.analyzer_factory()
        for chunk, batch in zip(golden.chunks, holdout.batches(), strict=True):
            metadata = _sample_metadata(batch)
            sample_ids = [str(item["sample_id"]) for item in metadata]
            if sample_ids != chunk.sample_ids:
                raise ValueError(f"golden chunk sample order differs: {sample_ids!r} != {chunk.sample_ids!r}")
            with np.load(chunk.artifact.path, allow_pickle=False) as archive:
                reference = {name: np.asarray(archive[name]) for name in archive.files}
            candidate = adapter.forward(batch, capture_points)
            analyzer.add(
                reference,
                candidate,
                metadata,
                action_name=self.action_name,
                flow_name=self.flow_name,
                action_schema=adapter.action_schema,
            )
        diagnostics = analyzer.finalize()
        evaluation = {} if self.evaluator is None else self.evaluator.evaluate(adapter, holdout)
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
            comparison=_legacy_comparison(diagnostics),
            commands=[] if not self.command else [self.command],
            timing_boundary="source/offline numerical diagnostics; no target latency",
            notes=["Offline diagnostics are not TensorRT, hardware latency, closed-loop success, or accepted deployment evidence"],
            trial=trial,
            diagnostics=diagnostics,
            evaluation=evaluation,
        )
        self.evidence_store.write(record, str(evidence_path))
        evidence_reference = CandidateEvidenceRef(
            trial_id=trial.trial_id,
            record_id=record.record_id,
            path=str(evidence_path),
            sha256=_sha256(evidence_path),
            status="measured",
        )
        return record, evidence_reference

    @staticmethod
    def rank_records(records: Sequence[EvidenceRecord]) -> list[SensitivityRank]:
        broad = next((record for record in records if record.trial is not None and record.trial.kind == "broad"), None)
        ranking: list[SensitivityRank] = []
        if broad is None or broad.diagnostics is None or broad.diagnostics.overall.action.l2.mean is None:
            return ranking
        broad_metrics = {"action_l2_mean": broad.diagnostics.overall.action.l2.mean}
        if broad.diagnostics.overall.flow is not None and broad.diagnostics.overall.flow.l2.mean is not None:
            broad_metrics["flow_l2_mean"] = broad.diagnostics.overall.flow.l2.mean
        for record in records:
            if (
                record.trial is None
                or record.trial.kind not in {"rollback_component", "rollback_block_group", "rollback_block", "rollback_layer"}
                or record.diagnostics is None
            ):
                continue
            rollback_metrics = {"action_l2_mean": record.diagnostics.overall.action.l2.mean}
            if record.diagnostics.overall.flow is not None:
                rollback_metrics["flow_l2_mean"] = record.diagnostics.overall.flow.l2.mean
            for metric, broad_error in broad_metrics.items():
                rollback_error = rollback_metrics.get(metric)
                if rollback_error is None:
                    continue
                ranking.append(
                    SensitivityRank(
                        component=",".join(record.trial.rollback_components),
                        metric=metric,
                        broad_error=broad_error,
                        rollback_error=rollback_error,
                        recovery=(broad_error - rollback_error) / max(broad_error, 1e-12),
                        quantized_parameter_coverage=_coverage_fraction(record.module_coverage),
                        evidence_record_id=record.record_id,
                    )
                )
        ranking.sort(key=lambda item: (item.metric, -item.recovery, item.component))
        return ranking

    @staticmethod
    def calibration_ablation(records: Sequence[EvidenceRecord]) -> list[CalibrationAblation]:
        broad = next((record for record in records if record.trial is not None and record.trial.kind == "broad"), None)
        if broad is None or broad.diagnostics is None or broad.diagnostics.overall.action.l2.mean is None:
            return []
        if broad.trial is None:
            raise ValueError("broad evidence record is missing its trial contract")
        baseline_error = broad.diagnostics.overall.action.l2.mean
        baseline_fingerprint = broad.trial.calibration_manifest_fingerprint

        def component_rollbacks(
            *, kind: str, parent_trial_id: str, calibration_fingerprint: str, parent_error: float
        ) -> dict[str, tuple[EvidenceRecord, float, float]]:
            result: dict[str, tuple[EvidenceRecord, float, float]] = {}
            for candidate in records:
                trial = candidate.trial
                if trial is None or trial.kind != kind or trial.parent_trial_id != parent_trial_id:
                    continue
                if trial.calibration_manifest_fingerprint != calibration_fingerprint:
                    raise ValueError(f"calibration rollback {trial.trial_id!r} differs from its parent calibration")
                if len(trial.rollback_components) != 1 or candidate.diagnostics is None:
                    raise ValueError(f"calibration ranking trial {trial.trial_id!r} must restore exactly one component")
                error = candidate.diagnostics.overall.action.l2.mean
                if error is None:
                    raise ValueError(f"calibration ranking trial {trial.trial_id!r} is missing action L2")
                component = trial.rollback_components[0]
                if component in result:
                    raise ValueError(f"duplicate calibration ranking component {component!r}")
                result[component] = (candidate, error, (parent_error - error) / max(parent_error, 1e-12))
            return result

        baseline_rollbacks = component_rollbacks(
            kind="rollback_component",
            parent_trial_id=broad.trial.trial_id,
            calibration_fingerprint=baseline_fingerprint,
            parent_error=baseline_error,
        )
        comparisons = []
        for record in records:
            if record.trial is None or record.trial.kind != "calibration_control" or record.diagnostics is None:
                continue
            control_error = record.diagnostics.overall.action.l2.mean
            if control_error is None:
                continue
            control_rollbacks = component_rollbacks(
                kind="calibration_rollback",
                parent_trial_id=record.trial.trial_id,
                calibration_fingerprint=record.trial.calibration_manifest_fingerprint,
                parent_error=control_error,
            )
            if control_rollbacks and set(control_rollbacks) != set(baseline_rollbacks):
                raise ValueError("calibration ranking control must cover the same rollback components as the baseline")
            baseline_order = sorted(baseline_rollbacks, key=lambda component: (-baseline_rollbacks[component][2], component))
            control_order = sorted(control_rollbacks, key=lambda component: (-control_rollbacks[component][2], component))
            baseline_ranks = {component: index + 1 for index, component in enumerate(baseline_order)}
            control_ranks = {component: index + 1 for index, component in enumerate(control_order)}
            rank_changes = [
                CalibrationRankChange(
                    component=component,
                    baseline_trial_id=baseline_rollbacks[component][0].trial.trial_id,  # type: ignore[union-attr]
                    control_trial_id=control_rollbacks[component][0].trial.trial_id,  # type: ignore[union-attr]
                    baseline_error=baseline_rollbacks[component][1],
                    control_error=control_rollbacks[component][1],
                    baseline_recovery=baseline_rollbacks[component][2],
                    control_recovery=control_rollbacks[component][2],
                    baseline_rank=baseline_ranks[component],
                    control_rank=control_ranks[component],
                    rank_delta=control_ranks[component] - baseline_ranks[component],
                )
                for component in baseline_order
                if component in control_rollbacks
            ]
            rank_correlation = None
            if len(rank_changes) > 1:
                squared_rank_delta = sum(change.rank_delta**2 for change in rank_changes)
                count = len(rank_changes)
                rank_correlation = 1.0 - 6.0 * squared_rank_delta / (count * (count**2 - 1))
            comparisons.append(
                CalibrationAblation(
                    baseline_trial_id=broad.trial.trial_id,
                    control_trial_id=record.trial.trial_id,
                    metric="action_l2_mean",
                    baseline_error=baseline_error,
                    control_error=control_error,
                    relative_error_change=(control_error - baseline_error) / max(baseline_error, 1e-12),
                    rank_metric="action_l2_recovery" if rank_changes else None,
                    rank_correlation=rank_correlation,
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
        holdout: ManifestCalibrationProvider,
        golden: GoldenCaptureManifest,
    ) -> SensitivityStudyRecord:
        if not trials:
            raise ValueError("sensitivity study requires at least one trial")
        if len(trials) != len(records) or len(records) != len(references):
            raise ValueError("study trial, record, and reference counts must match")
        study_path = self.artifact_root / "study.json"
        if study_path.exists():
            raise FileExistsError(f"sensitivity study evidence already exists: {study_path}")
        adapter = self.adapter_factory()
        golden_path = self.artifact_root / "golden" / "manifest.json"
        study = SensitivityStudyRecord(
            study_id=study_id,
            status="measured",
            model=adapter.spec,
            action_schema=adapter.action_schema,
            module_inventory_sha256=fingerprint(list(adapter.module_inventory())),
            calibration_manifest_fingerprint=trials[0].calibration_manifest_fingerprint,
            calibration_manifest_fingerprints=sorted({trial.calibration_manifest_fingerprint for trial in trials}),
            holdout_manifest_fingerprint=holdout.fingerprint,
            golden_manifest=ArtifactRef(kind="fp-golden-manifest", path=str(golden_path), sha256=_sha256(golden_path)),
            trials=list(trials),
            candidates=list(references),
            ranking=self.rank_records(records),
            calibration_ablation=self.calibration_ablation(records),
            notes=["Agent-generated source/offline measured evidence; human acceptance remains pending"],
        )
        if study.golden_manifest.sha256 != _sha256(Path(golden_path)) or golden.holdout_manifest_fingerprint != holdout.fingerprint:
            raise ValueError("golden manifest identity differs while finalizing study")
        self.evidence_store.write(study, str(study_path))
        return study

    def run_study(
        self,
        *,
        study_id: str,
        trials: Sequence[SensitivityTrial],
        plans: Mapping[str, OptimizationPlan],
        calibrations: Mapping[str, ManifestCalibrationProvider],
        holdout: ManifestCalibrationProvider,
        seed: int,
    ) -> SensitivityStudyRecord:
        if not trials:
            raise ValueError("sensitivity study requires at least one trial")
        golden = self.freeze_golden(holdout, seed=seed)
        records: list[EvidenceRecord] = []
        references: list[CandidateEvidenceRef] = []
        for trial in trials:
            calibration = calibrations[trial.calibration_manifest_fingerprint]
            record, reference = self.run_trial(trial, plans[trial.trial_id], calibration, holdout, golden)
            records.append(record)
            references.append(reference)
        return self.finalize_study(
            study_id=study_id,
            trials=trials,
            records=records,
            references=references,
            holdout=holdout,
            golden=golden,
        )
