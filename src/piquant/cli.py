"""Small CLI for inspecting plans and comparing captured tensors."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from piquant import __version__, load_compilation_plan, load_plan
from piquant.analysis import NumpyNumericalAnalyzer
from piquant.contracts import ModelSpec
from piquant.deployment import validate_deployment_manifest
from piquant.evidence import (
    load_study,
    package_import_report,
    summarize_study,
    target_fingerprint,
    validate_study,
)
from piquant.inspection import inspect_onnx_model
from piquant.integrations import TensorRTCliCompiler, build_trtexec_command, summarize_tensorrt_layers


def _doctor(_args: argparse.Namespace) -> int:
    print(
        json.dumps(
            {
                "piquant_version": __version__,
                "target": target_fingerprint().model_dump(mode="json"),
                "loaded_optional_modules": package_import_report(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_plan(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    print(plan.model_dump_json(indent=2))
    return 0


def _validate_compilation_plan(args: argparse.Namespace) -> int:
    plan = load_compilation_plan(args.plan)
    print(plan.model_dump_json(indent=2))
    return 0


def _inspect_onnx(args: argparse.Namespace) -> int:
    print(inspect_onnx_model(args.model).model_dump_json(indent=2))
    return 0


def _trtexec_command(args: argparse.Namespace) -> int:
    plan = load_compilation_plan(args.plan)
    command = build_trtexec_command(plan, engine_path=args.engine, layer_info_path=args.layer_info, skip_inference=args.skip_inference)
    print(json.dumps({"argv": command, "command": " ".join(command)}, indent=2, sort_keys=True))
    return 0


def _compile_tensorrt(args: argparse.Namespace) -> int:
    model = ModelSpec(
        model_id=args.model_id,
        family=args.family,
        framework=args.framework,
        revision=args.revision,
        task=args.task,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )
    record = TensorRTCliCompiler(output_dir=args.output_dir, trtexec=args.trtexec).compile(load_compilation_plan(args.plan), model)
    print(record.model_dump_json(indent=2))
    return 0


def _inspect_trt_layers(args: argparse.Namespace) -> int:
    print(summarize_tensorrt_layers(args.layer_info).model_dump_json(indent=2))
    return 0


def _load_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _compare(args: argparse.Namespace) -> int:
    reference = _load_npz(args.reference)
    candidate = _load_npz(args.candidate)
    report = NumpyNumericalAnalyzer().compare(reference, candidate, args.action_name)
    print(report.model_dump_json(indent=2))
    return 0


def _summarize_study(args: argparse.Namespace) -> int:
    summary = summarize_study(load_study(args.study))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _validate_study(args: argparse.Namespace) -> int:
    print(json.dumps(validate_study(args.study), indent=2, sort_keys=True))
    return 0


def _validate_deployment(args: argparse.Namespace) -> int:
    print(json.dumps(validate_deployment_manifest(args.manifest, check_artifacts=args.check_artifacts), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piquant", description="VLA quantization evidence tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="print local package and optional-dependency status")
    doctor.set_defaults(handler=_doctor)

    validate_plan = subparsers.add_parser("validate-plan", help="parse and validate a YAML/JSON plan")
    validate_plan.add_argument("plan", type=Path)
    validate_plan.set_defaults(handler=_validate_plan)

    validate_compilation_plan = subparsers.add_parser("validate-compilation-plan", help="parse and validate a target compiler plan")
    validate_compilation_plan.add_argument("plan", type=Path)
    validate_compilation_plan.set_defaults(handler=_validate_compilation_plan)

    inspect_onnx = subparsers.add_parser("inspect-onnx", help="inspect ONNX graph operators without compiling")
    inspect_onnx.add_argument("model", type=Path)
    inspect_onnx.set_defaults(handler=_inspect_onnx)

    trtexec_command = subparsers.add_parser("trtexec-command", help="render a TensorRT trtexec argv without running it")
    trtexec_command.add_argument("plan", type=Path)
    trtexec_command.add_argument("--engine", required=True)
    trtexec_command.add_argument("--layer-info", required=True)
    trtexec_command.add_argument("--skip-inference", action=argparse.BooleanOptionalAction, default=True)
    trtexec_command.set_defaults(handler=_trtexec_command)

    compile_tensorrt = subparsers.add_parser(
        "compile-tensorrt",
        help="run trtexec for one explicit compilation plan and emit evidence JSON",
    )
    compile_tensorrt.add_argument("plan", type=Path)
    compile_tensorrt.add_argument("--output-dir", required=True, type=Path)
    compile_tensorrt.add_argument("--trtexec", default="trtexec")
    compile_tensorrt.add_argument("--model-id", required=True)
    compile_tensorrt.add_argument("--family", required=True)
    compile_tensorrt.add_argument("--framework", required=True)
    compile_tensorrt.add_argument("--revision", default="local")
    compile_tensorrt.add_argument("--task", choices=["vla", "wam", "flow_action"], default="vla")
    compile_tensorrt.add_argument("--action-dim", required=True, type=int)
    compile_tensorrt.add_argument("--action-horizon", required=True, type=int)
    compile_tensorrt.set_defaults(handler=_compile_tensorrt)

    inspect_trt_layers = subparsers.add_parser("inspect-trt-layers", help="summarize a TensorRT --exportLayerInfo JSON file")
    inspect_trt_layers.add_argument("layer_info", type=Path)
    inspect_trt_layers.set_defaults(handler=_inspect_trt_layers)

    compare = subparsers.add_parser("compare", help="compare named tensors in two NumPy archives")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--action-name", default="action")
    compare.set_defaults(handler=_compare)

    summarize_study = subparsers.add_parser("summarize-study", help="summarize a sensitivity study without loading artifacts")
    summarize_study.add_argument("study", type=Path)
    summarize_study.set_defaults(handler=_summarize_study)

    validate_study = subparsers.add_parser("validate-study", help="validate sensitivity study identity, hashes, and evidence lineage")
    validate_study.add_argument("study", type=Path)
    validate_study.set_defaults(handler=_validate_study)

    validate_deployment = subparsers.add_parser("validate-deployment", help="validate a deployment handoff manifest")
    validate_deployment.add_argument("manifest", type=Path)
    validate_deployment.add_argument("--check-artifacts", action="store_true")
    validate_deployment.set_defaults(handler=_validate_deployment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
