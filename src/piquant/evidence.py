"""Evidence fingerprinting and persistence with explicit promotion status."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from piquant.contracts import Contract, EvidenceRecord, GoldenCaptureManifest, SensitivityStudyRecord, TargetFingerprint, fingerprint


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def target_fingerprint(device: str = "cpu") -> TargetFingerprint:
    """Capture environment identity without importing optional accelerators."""

    return TargetFingerprint(
        platform=f"{platform.system()}-{platform.machine()}",
        python_version=platform.python_version(),
        device=device,
        torch_version=_optional_version("torch"),
        modelopt_version=_optional_version("nvidia-modelopt"),
        onnx_version=_optional_version("onnx"),
        onnxruntime_version=_optional_version("onnxruntime"),
    )


class JsonEvidenceStore:
    """Write validated evidence atomically and never decide acceptance."""

    def write(self, record: Contract, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def read(self, path: str) -> EvidenceRecord:
        return EvidenceRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def metadata_fingerprint(metadata: dict[str, Any]) -> str:
    """Expose a stable helper for calibration and artifact lineage."""

    return fingerprint(metadata)


def load_sensitivity_study(path: str | Path) -> SensitivityStudyRecord:
    return SensitivityStudyRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))


def summarize_sensitivity_study(study: SensitivityStudyRecord) -> dict[str, Any]:
    return {
        "study_id": study.study_id,
        "status": study.status,
        "model_id": study.model.model_id,
        "evidence_boundary": study.evidence_boundary,
        "trial_count": len(study.trials),
        "candidate_count": len(study.candidates),
        "calibration_fingerprints": study.calibration_manifest_fingerprints or [study.calibration_manifest_fingerprint],
        "holdout_fingerprint": study.holdout_manifest_fingerprint,
        "ranking": [rank.model_dump(mode="json") for rank in study.ranking],
        "calibration_ablation": [comparison.model_dump(mode="json") for comparison in study.calibration_ablation],
    }


def validate_sensitivity_study(path: str | Path) -> dict[str, Any]:
    """Validate external artifact lineage without importing any optional runtime."""

    study = load_sensitivity_study(path)
    if len(study.trials) != len(study.candidates):
        raise ValueError("study must contain one candidate evidence reference per trial")
    if _sha256(study.golden_manifest.path) != study.golden_manifest.sha256:
        raise ValueError("golden manifest SHA256 differs from study reference")
    golden = GoldenCaptureManifest.model_validate_json(Path(study.golden_manifest.path).read_text(encoding="utf-8"))
    if golden.model != study.model or golden.action_schema != study.action_schema:
        raise ValueError("golden model/action contract differs from study")
    if golden.holdout_manifest_fingerprint != study.holdout_manifest_fingerprint:
        raise ValueError("golden holdout fingerprint differs from study")
    for chunk in golden.chunks:
        if _sha256(chunk.artifact.path) != chunk.artifact.sha256:
            raise ValueError(f"golden capture chunk SHA256 differs for {chunk.chunk_id!r}")

    trials = {trial.trial_id: trial for trial in study.trials}
    for candidate in study.candidates:
        if _sha256(candidate.path) != candidate.sha256:
            raise ValueError(f"candidate evidence SHA256 differs for trial {candidate.trial_id!r}")
        record = EvidenceRecord.model_validate_json(Path(candidate.path).read_text(encoding="utf-8"))
        trial = trials[candidate.trial_id]
        if record.record_id != candidate.record_id or record.status != candidate.status or record.trial != trial:
            raise ValueError(f"candidate evidence identity differs for trial {candidate.trial_id!r}")
        if record.model != study.model or record.calibration_fingerprint != trial.calibration_manifest_fingerprint:
            raise ValueError(f"candidate model/calibration identity differs for trial {candidate.trial_id!r}")
        if fingerprint(record.plan) != trial.resolved_plan_hash:
            raise ValueError(f"candidate resolved plan hash differs for trial {candidate.trial_id!r}")
        if trial.kind != "fp_control" and record.module_coverage.matched_count == 0:
            raise ValueError(f"quantized trial {candidate.trial_id!r} has zero module coverage")
        if record.quantization is None or record.quantization.module_coverage != record.module_coverage:
            raise ValueError(f"candidate quantization evidence is missing or inconsistent for trial {candidate.trial_id!r}")
    return summarize_sensitivity_study(study)


def package_import_report() -> dict[str, bool]:
    """Report discoverability only; this function intentionally imports nothing optional."""

    modules = ("torch", "modelopt", "onnx", "onnxruntime")
    return {module: module in sys.modules for module in modules}
