"""Operator graph inspection helpers with optional ONNX loading."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from piquant.contracts import ArtifactRef, OperatorGraphReport, OperatorNodeInfo

_CONSTANT_WEIGHT_OPS = {"Conv", "Gemm", "MatMul"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_operator_graph(
    *,
    graph_id: str,
    nodes: Sequence[OperatorNodeInfo | Mapping[str, Any]],
    inputs: Sequence[str],
    outputs: Sequence[str],
    initializer_names: Sequence[str],
    dtype_counts: Mapping[str, int] | None = None,
    source: ArtifactRef | None = None,
    producer: str = "unknown",
    opset: Mapping[str, int] | None = None,
    external_data: bool = False,
) -> OperatorGraphReport:
    resolved_nodes = [node if isinstance(node, OperatorNodeInfo) else OperatorNodeInfo.model_validate(node) for node in nodes]
    op_counts: dict[str, int] = {}
    initializers = set(initializer_names)
    constant_weight_candidates: list[str] = []
    for index, node in enumerate(resolved_nodes):
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type in _CONSTANT_WEIGHT_OPS and any(name in initializers for name in node.inputs[1:]):
            constant_weight_candidates.append(node.name or f"{node.op_type}:{index}")
    return OperatorGraphReport(
        graph_id=graph_id,
        source=source,
        producer=producer,
        opset=dict(opset or {}),
        inputs=list(inputs),
        outputs=list(outputs),
        nodes=resolved_nodes,
        op_counts=dict(sorted(op_counts.items())),
        initializer_count=len(initializer_names),
        initializer_names=sorted(initializer_names),
        dtype_counts=dict(sorted((dtype_counts or {}).items())),
        external_data=external_data,
        constant_weight_candidates=constant_weight_candidates,
    )


def inspect_onnx_model(path: str | Path) -> OperatorGraphReport:
    """Inspect ONNX graph structure without making quantization or latency claims."""

    try:
        import onnx
    except ModuleNotFoundError as error:
        raise RuntimeError("ONNX inspection requires the optional 'onnx' extra") from error

    model_path = Path(path)
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    initializers = list(graph.initializer)
    dtype_counts: dict[str, int] = {}
    external_data = False
    for initializer in initializers:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        external_data = external_data or bool(getattr(initializer, "external_data", []))
    nodes = [
        OperatorNodeInfo(
            name=node.name,
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
        )
        for node in graph.node
    ]
    opset = {entry.domain or "ai.onnx": int(entry.version) for entry in model.opset_import}
    return summarize_operator_graph(
        graph_id=graph.name or model_path.stem,
        source=ArtifactRef(kind="onnx", path=str(model_path), sha256=sha256_file(model_path)),
        producer=model.producer_name or "unknown",
        opset=opset,
        inputs=[value.name for value in graph.input],
        outputs=[value.name for value in graph.output],
        nodes=nodes,
        initializer_names=[initializer.name for initializer in initializers],
        dtype_counts=dtype_counts,
        external_data=external_data,
    )
