"""Deployment evidence helpers that do not own runtime acceptance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from piquant.contracts import DeploymentCandidateManifest, LatencyDistribution, load_deployment_manifest


def latency_distribution(values_ms: Sequence[float]) -> LatencyDistribution:
    if not values_ms:
        raise ValueError("latency distribution requires at least one value")
    values = np.asarray(values_ms, dtype=np.float64)
    if not bool(np.isfinite(values).all()) or bool((values < 0.0).any()):
        raise ValueError("latency values must be finite and non-negative")
    return LatencyDistribution(
        count=int(values.size),
        mean_ms=float(np.mean(values)),
        min_ms=float(np.min(values)),
        max_ms=float(np.max(values)),
        std_ms=float(np.std(values)),
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_deployment_manifest(path: str | Path, *, check_artifacts: bool = False) -> dict[str, object]:
    manifest = load_deployment_manifest(path)
    if check_artifacts:
        refs = []
        if manifest.source_model is not None:
            refs.append(manifest.source_model)
        refs.extend(artifact.artifact for record in manifest.compiler_records for artifact in record.artifacts)
        refs.extend(artifact for report in manifest.timing_reports for artifact in report.artifacts)
        for ref in refs:
            if _sha256(ref.path) != ref.sha256:
                raise ValueError(f"deployment artifact SHA256 differs: {ref.kind}:{ref.path}")
    return summarize_deployment_manifest(manifest)


def summarize_deployment_manifest(manifest: DeploymentCandidateManifest) -> dict[str, object]:
    return {
        "manifest_id": manifest.manifest_id,
        "status": manifest.status,
        "model_id": manifest.model.model_id,
        "target_device": manifest.target.device,
        "evidence_boundary": manifest.evidence_boundary,
        "compiler_record_count": len(manifest.compiler_records),
        "timing_report_count": len(manifest.timing_reports),
        "pi_cpp_integration_status": manifest.pi_cpp_integration_status,
        "human_acceptance": manifest.human_acceptance,
        "precisions": sorted(set(manifest.precision_map.values())),
    }


def write_deployment_manifest(manifest: DeploymentCandidateManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
