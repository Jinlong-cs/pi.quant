"""Evidence fingerprinting and persistence with explicit promotion status."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

from piquant.contracts import EvidenceRecord, TargetFingerprint, fingerprint


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

    def write(self, record: EvidenceRecord, path: str) -> None:
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


def package_import_report() -> dict[str, bool]:
    """Report discoverability only; this function intentionally imports nothing optional."""

    modules = ("torch", "modelopt", "onnx", "onnxruntime")
    return {module: module in sys.modules for module in modules}
