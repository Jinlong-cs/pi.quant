#!/usr/bin/env python3
"""Run the real Pi0.5/LIBERO source-level sensitivity workflow through explicit integrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from piquant.adapters.pi05 import Pi05TorchAdapter
from piquant.analysis import StreamingDiagnosticAnalyzer
from piquant.backends.modelopt import ModelOptBackend
from piquant.calibration import BatchedManifestCalibrationProvider, load_manifest, manifest_fingerprint, save_manifest
from piquant.contracts import CandidateEvidenceRef, GoldenCaptureManifest, SensitivityTrial, fingerprint, load_plan
from piquant.evidence import JsonEvidenceStore
from piquant.integrations.openpi import Pi05LiberoData, Pi05OpenPIConfig, build_pi05_libero_manifests
from piquant.sensitivity import SensitivityRunner


def _config(path: Path) -> Pi05OpenPIConfig:
    values: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for name in ("checkpoint_dir", "norm_stats_dir", "openpi_data_home"):
        values[name] = Path(values[name]).expanduser().resolve()
    config = Pi05OpenPIConfig(**values)
    config.validate_identity()
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runner(config: Pi05OpenPIConfig, data: Pi05LiberoData, artifact_root: Path, source_device: str) -> SensitivityRunner:
    return SensitivityRunner(
        adapter_factory=lambda: Pi05TorchAdapter(config, data),
        backend=ModelOptBackend(),
        analyzer_factory=lambda: StreamingDiagnosticAnalyzer(seed=config.seed),
        evidence_store=JsonEvidenceStore(),
        artifact_root=artifact_root,
        action_name="action.output",
        flow_name="flow.selected",
        command=shlex.join(sys.argv),
        source_device=source_device,
    )


def _manifests(args: argparse.Namespace) -> None:
    config = _config(args.identity)
    manifests = build_pi05_libero_manifests(
        args.dataset_root,
        normalization_revision=config.norm_stats_sha256,
        openpi_revision=config.openpi_revision,
        seed=config.seed,
        horizon=config.action_horizon,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {}
    for split, manifest in manifests.items():
        path = args.output_dir / f"{split}.json"
        save_manifest(manifest, path)
        summary[split] = {"path": str(path), "samples": len(manifest.samples), "fingerprint": manifest_fingerprint(manifest)}
    print(json.dumps(summary, indent=2, sort_keys=True))


def _golden(args: argparse.Namespace) -> None:
    config = _config(args.identity)
    data = Pi05LiberoData(config, args.dataset_root)
    holdout = load_manifest(args.holdout_manifest)
    provider = BatchedManifestCalibrationProvider(holdout, data.load_batch, args.batch_size)
    manifest = _runner(config, data, args.artifact_root, args.source_device).freeze_golden(provider, seed=config.seed)
    print(manifest.model_dump_json(indent=2))


def _trial(args: argparse.Namespace) -> None:
    config = _config(args.identity)
    data = Pi05LiberoData(config, args.dataset_root)
    calibration_manifest = load_manifest(args.calibration_manifest)
    holdout_manifest = load_manifest(args.holdout_manifest)
    calibration = BatchedManifestCalibrationProvider(calibration_manifest, data.load_batch, args.batch_size)
    holdout = BatchedManifestCalibrationProvider(holdout_manifest, data.load_batch, args.batch_size)
    golden = GoldenCaptureManifest.model_validate_json(args.golden_manifest.read_text(encoding="utf-8"))
    plan = load_plan(args.plan)
    is_control = args.kind == "fp_control"
    if is_control and (plan.backend != "none" or plan.representation != "fp_control" or plan.quant_format != "none"):
        raise ValueError("fp_control trials require backend=none, representation=fp_control, and quant_format=none")
    if not is_control and (plan.backend != "modelopt" or plan.representation != "fake_quant" or plan.quant_format != "int8"):
        raise ValueError("quantized trials require backend=modelopt, representation=fake_quant, and quant_format=int8")
    trial = SensitivityTrial(
        trial_id=args.trial_id,
        kind=args.kind,
        quantized_components=args.quantized_component,
        rollback_components=args.rollback_component,
        parent_trial_id=args.parent_trial_id,
        calibration_manifest_fingerprint=manifest_fingerprint(calibration_manifest),
        resolved_plan_hash=fingerprint(plan),
        seed=config.seed,
        notes=args.note,
    )
    record, reference = _runner(config, data, args.artifact_root, args.source_device).run_trial(trial, plan, calibration, holdout, golden)
    print(json.dumps({"record": record.model_dump(mode="json"), "reference": reference.model_dump(mode="json")}, indent=2))


def _study(args: argparse.Namespace) -> None:
    config = _config(args.identity)
    data = Pi05LiberoData(config, args.dataset_root)
    holdout_manifest = load_manifest(args.holdout_manifest)
    holdout = BatchedManifestCalibrationProvider(holdout_manifest, data.load_batch, args.batch_size)
    golden = GoldenCaptureManifest.model_validate_json(args.golden_manifest.read_text(encoding="utf-8"))
    trial_ids = json.loads(args.trial_index.read_text(encoding="utf-8"))
    records = []
    references = []
    trials = []
    store = JsonEvidenceStore()
    for trial_id in trial_ids:
        path = args.artifact_root / "candidates" / trial_id / "evidence.json"
        record = store.read(str(path))
        if record.trial is None or record.trial.trial_id != trial_id or record.status not in {"measured", "pending", "rejected"}:
            raise ValueError(f"candidate evidence differs from trial index for {trial_id!r}")
        records.append(record)
        trials.append(record.trial)
        references.append(
            CandidateEvidenceRef(
                trial_id=trial_id,
                record_id=record.record_id,
                path=str(path),
                sha256=_sha256(path),
                status=record.status,
            )
        )
    study = _runner(config, data, args.artifact_root, args.source_device).finalize_study(
        study_id=args.study_id,
        trials=trials,
        records=records,
        references=references,
        holdout=holdout,
        golden=golden,
    )
    print(study.model_dump_json(indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifests = subparsers.add_parser("manifests", help="build episode-disjoint real LIBERO manifests")
    manifests.add_argument("--identity", type=Path, required=True)
    manifests.add_argument("--dataset-root", type=Path, required=True)
    manifests.add_argument("--output-dir", type=Path, required=True)
    manifests.set_defaults(handler=_manifests)

    golden = subparsers.add_parser("golden", help="freeze chunked FP captures for one diagnostic manifest")
    golden.add_argument("--identity", type=Path, required=True)
    golden.add_argument("--dataset-root", type=Path, required=True)
    golden.add_argument("--holdout-manifest", type=Path, required=True)
    golden.add_argument("--artifact-root", type=Path, required=True)
    golden.add_argument("--batch-size", type=int, default=4)
    golden.add_argument("--source-device", required=True)
    golden.set_defaults(handler=_golden)

    trial = subparsers.add_parser("trial", help="run one immutable FP or ModelOpt sensitivity trial")
    trial.add_argument("--identity", type=Path, required=True)
    trial.add_argument("--dataset-root", type=Path, required=True)
    trial.add_argument("--plan", type=Path, required=True)
    trial.add_argument("--calibration-manifest", type=Path, required=True)
    trial.add_argument("--holdout-manifest", type=Path, required=True)
    trial.add_argument("--golden-manifest", type=Path, required=True)
    trial.add_argument("--artifact-root", type=Path, required=True)
    trial.add_argument("--trial-id", required=True)
    trial.add_argument(
        "--kind",
        choices=(
            "fp_control",
            "broad",
            "component_only",
            "rollback_component",
            "rollback_block_group",
            "rollback_block",
            "rollback_layer",
            "calibration_control",
            "calibration_rollback",
        ),
        required=True,
    )
    trial.add_argument("--quantized-component", action="append", default=[])
    trial.add_argument("--rollback-component", action="append", default=[])
    trial.add_argument("--parent-trial-id")
    trial.add_argument("--note", action="append", default=[])
    trial.add_argument("--batch-size", type=int, default=4)
    trial.add_argument("--source-device", required=True)
    trial.set_defaults(handler=_trial)

    study = subparsers.add_parser("study", help="finalize indexed candidate records into one sensitivity study")
    study.add_argument("--identity", type=Path, required=True)
    study.add_argument("--dataset-root", type=Path, required=True)
    study.add_argument("--holdout-manifest", type=Path, required=True)
    study.add_argument("--golden-manifest", type=Path, required=True)
    study.add_argument("--trial-index", type=Path, required=True)
    study.add_argument("--artifact-root", type=Path, required=True)
    study.add_argument("--study-id", required=True)
    study.add_argument("--batch-size", type=int, default=4)
    study.add_argument("--source-device", required=True)
    study.set_defaults(handler=_study)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
