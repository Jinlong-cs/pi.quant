#!/usr/bin/env python3
"""Run the portable v0.1 synthetic VLA evidence workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from piquant.adapters import (
    SyntheticCalibrationProvider,
    SyntheticFlowVLAAdapter,
    TorchSyntheticFlowVLAAdapter,
)
from piquant.analysis import NumpyNumericalAnalyzer
from piquant.backends import ModelOptBackend, ReferenceQDQBackend
from piquant.contracts import ArtifactRef, EvidenceRecord, fingerprint, load_plan
from piquant.evidence import JsonEvidenceStore, target_fingerprint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _captures(adapter: object, batches: list[dict[str, object]], points: list[str]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {point: [] for point in points}
    for batch in batches:
        output = adapter.forward(batch, points)  # type: ignore[attr-defined]
        for point in points:
            grouped[point].append(output[point])
    return {point: np.concatenate(values, axis=0) for point, values in grouped.items()}


def run(recipe: Path, output_dir: Path) -> EvidenceRecord:
    plan = load_plan(recipe)
    calibration = SyntheticCalibrationProvider(plan.calibration.sample_count, plan.calibration.seed)
    golden_adapter: object = SyntheticFlowVLAAdapter(plan.seed)
    candidate_adapter: object
    if plan.backend == "modelopt":
        golden_torch_adapter = TorchSyntheticFlowVLAAdapter(plan.seed)
        modelopt_adapter = TorchSyntheticFlowVLAAdapter(plan.seed)
        quantized_model, quantization = ModelOptBackend().quantize(modelopt_adapter, plan, calibration)
        golden_adapter = golden_torch_adapter
        candidate_adapter = modelopt_adapter.with_backend_model(quantized_model)
    elif plan.backend == "reference_qdq":
        reference_adapter = SyntheticFlowVLAAdapter(plan.seed)
        candidate_adapter, quantization = ReferenceQDQBackend().quantize(reference_adapter, plan, calibration)
    else:
        raise ValueError(f"unsupported synthetic backend {plan.backend!r}")
    batches = calibration.batches()
    golden = _captures(golden_adapter, batches, plan.capture_points)
    candidate = _captures(candidate_adapter, batches, plan.capture_points)
    comparison = NumpyNumericalAnalyzer(gripper_index=5).compare(golden, candidate, "action")
    output_dir.mkdir(parents=True, exist_ok=True)
    golden_path = output_dir / "golden.npz"
    candidate_path = output_dir / "candidate.npz"
    np.savez(golden_path, **golden)
    np.savez(candidate_path, **candidate)
    notes = (
        [
            "ModelOpt 0.45.0 fake quantization executed through mtq.quantize",
            "This is a CPU synthetic integration result, not deployment evidence",
        ]
        if quantization.backend == "modelopt"
        else [
            "reference_qdq is a portable numerical harness, not a ModelOpt or deployment result",
            "The recipe explicitly selected reference_qdq",
        ]
    )
    record = EvidenceRecord(
        record_id=f"synthetic-{fingerprint({'plan': plan.plan_id, 'calibration': calibration.fingerprint})[:16]}",
        status="measured",
        model=golden_adapter.spec,  # type: ignore[attr-defined]
        target=target_fingerprint(),
        plan=plan,
        backend=quantization.backend,
        representation=quantization.representation,
        calibration_fingerprint=calibration.fingerprint,
        module_coverage=quantization.module_coverage,
        quantization=quantization,
        comparison=comparison,
        artifacts=[
            ArtifactRef(kind="golden-capture", path=str(golden_path), sha256=_sha256(golden_path)),
            ArtifactRef(kind="candidate-capture", path=str(candidate_path), sha256=_sha256(candidate_path)),
        ],
        commands=[" ".join(sys.argv)],
        timing_boundary=plan.timing_boundary,
        notes=notes,
    )
    evidence_path = output_dir / "evidence.json"
    JsonEvidenceStore().write(record, str(evidence_path))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.recipe, args.output_dir)
    print(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
