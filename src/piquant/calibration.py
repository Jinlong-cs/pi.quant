"""Manifest-backed calibration providers and split lineage validation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from piquant.contracts import CalibrationManifest, SampleRef, fingerprint

SAMPLE_REFS_KEY = "__piquant_sample_refs__"
BatchLoader = Callable[[Sequence[SampleRef]], Mapping[str, Any]]


def manifest_fingerprint(manifest: CalibrationManifest) -> str:
    return fingerprint(manifest)


def episode_keys(manifest: CalibrationManifest) -> set[tuple[str, int]]:
    return {(sample.suite, sample.episode_index) for sample in manifest.samples}


def require_episode_disjoint(*manifests: CalibrationManifest) -> None:
    for index, left in enumerate(manifests):
        left_keys = episode_keys(left)
        for right in manifests[index + 1 :]:
            overlap = sorted(left_keys & episode_keys(right))
            if overlap:
                raise ValueError(f"episode overlap between {left.manifest_id!r} and {right.manifest_id!r}: {overlap[:20]!r}")


def require_task_coverage(manifest: CalibrationManifest, expected_tasks: int) -> None:
    tasks = {sample.task_index for sample in manifest.samples}
    if len(tasks) < expected_tasks:
        raise ValueError(f"manifest {manifest.manifest_id!r} covers {len(tasks)} tasks, expected at least {expected_tasks}")


def save_manifest(manifest: CalibrationManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_manifest(path: str | Path) -> CalibrationManifest:
    return CalibrationManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


class BatchedManifestCalibrationProvider:
    """Load explicit sample references in deterministic batches through one injected model-specific loader."""

    def __init__(self, manifest: CalibrationManifest, loader: BatchLoader, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._manifest = manifest
        self._loader = loader
        self._batch_size = batch_size

    @property
    def manifest(self) -> CalibrationManifest:
        return self._manifest

    @property
    def fingerprint(self) -> str:
        return manifest_fingerprint(self._manifest)

    def batches(self) -> Iterable[Mapping[str, Any]]:
        for start in range(0, len(self._manifest.samples), self._batch_size):
            sample_refs = self._manifest.samples[start : start + self._batch_size]
            batch = dict(self._loader(sample_refs))
            if SAMPLE_REFS_KEY in batch:
                raise ValueError(f"batch loader must not populate reserved key {SAMPLE_REFS_KEY!r}")
            batch[SAMPLE_REFS_KEY] = sample_refs
            yield batch

    def forward_loop(self, model: Any) -> None:
        for batch in self.batches():
            model(batch)
