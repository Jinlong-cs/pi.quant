from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from piquant.integrations import OrtDebugCapture


@pytest.mark.ort
def test_ort_capture_uses_temporary_graph_copy(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    tensor = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
    output = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Relu", ["x"], ["hidden"], name="relu"),
            onnx.helper.make_node("Identity", ["hidden"], ["y"], name="identity"),
        ],
        "debug-graph",
        [tensor],
        [output],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model_path = tmp_path / "source.onnx"
    onnx.save(model, str(model_path))
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    values = OrtDebugCapture(model_path).capture({"x": np.asarray([[-1.0, 2.0]], dtype=np.float32)}, ["hidden", "y"])
    after = hashlib.sha256(model_path.read_bytes()).hexdigest()
    np.testing.assert_allclose(values["hidden"], [[0.0, 2.0]])
    np.testing.assert_allclose(values["y"], [[0.0, 2.0]])
    assert before == after
