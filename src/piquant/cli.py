"""Small CLI for inspecting plans and comparing captured tensors."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from piquant import __version__, load_plan
from piquant.analysis import NumpyNumericalAnalyzer
from piquant.evidence import (
    load_study,
    package_import_report,
    summarize_study,
    target_fingerprint,
    validate_study,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piquant", description="VLA quantization evidence tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="print local package and optional-dependency status")
    doctor.set_defaults(handler=_doctor)

    validate_plan = subparsers.add_parser("validate-plan", help="parse and validate a YAML/JSON plan")
    validate_plan.add_argument("plan", type=Path)
    validate_plan.set_defaults(handler=_validate_plan)

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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
