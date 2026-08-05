"""Optional ONNX Runtime intermediate-output capture on a temporary graph copy."""

from __future__ import annotations

import copy
import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


class OrtDebugCapture:
    """Expose selected ONNX values without mutating the source model."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)

    def _imports(self) -> tuple[Any, Any]:
        try:
            import onnx
            import onnxruntime as ort
        except ModuleNotFoundError as error:
            raise RuntimeError("ORT debug requires the optional 'onnx' extra") from error
        return onnx, ort

    def _source_hash(self) -> str:
        return hashlib.sha256(self.model_path.read_bytes()).hexdigest()

    def capture(self, feeds: dict[str, np.ndarray], tensor_names: list[str]) -> dict[str, np.ndarray]:
        onnx, ort = self._imports()
        if not tensor_names:
            raise ValueError("tensor_names must not be empty")
        source_hash = self._source_hash()
        model = onnx.load(str(self.model_path))
        model = onnx.shape_inference.infer_shapes(model)
        known = {value.name: value for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
        existing = {value.name for value in model.graph.output}
        for name in tensor_names:
            if name not in known:
                raise ValueError(f"tensor is not present in ONNX value info: {name}")
            if name not in existing:
                model.graph.output.append(copy.deepcopy(known[name]))
        with tempfile.TemporaryDirectory(prefix="piquant-ort-") as temporary_dir:
            augmented_path = Path(temporary_dir) / "debug.onnx"
            onnx.save(model, str(augmented_path))
            session = ort.InferenceSession(str(augmented_path), providers=["CPUExecutionProvider"])
            output_names = [output.name for output in session.get_outputs()]
            values = session.run(output_names, feeds)
        if self._source_hash() != source_hash:
            raise RuntimeError("source ONNX changed during debug capture")
        return {name: np.asarray(value) for name, value in zip(output_names, values, strict=True) if name in tensor_names}
