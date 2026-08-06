import json
import sys
from pathlib import Path

import pytest

from piquant.contracts import (
    ArtifactRef,
    BenchmarkProtocol,
    CompilationPlan,
    DeploymentCandidateManifest,
    ModelSpec,
    ShapeProfile,
    StageTimingReport,
    TargetCapability,
    TargetFingerprint,
    load_compilation_plan,
)
from piquant.deployment import latency_distribution, summarize_deployment_manifest
from piquant.inspection import inspect_onnx_model
from piquant.integrations import TensorRTCliCompiler, build_trtexec_command, summarize_tensorrt_layers


def _target() -> TargetFingerprint:
    return TargetFingerprint(
        platform="Linux-aarch64",
        python_version="3.12.0",
        device="cuda:0",
        gpu_name="Jetson AGX Orin",
        compute_capability="8.7",
        driver_version="580.0",
        cuda_version="12.6",
        tensorrt_version="10.3.0",
    )


def _model() -> ModelSpec:
    return ModelSpec(model_id="fastwam", family="wam", framework="onnx", revision="test", task="wam", action_dim=7, action_horizon=32)


def _source() -> ArtifactRef:
    return ArtifactRef(kind="candidate-onnx", path="/external/candidate.onnx", sha256="0" * 64)


def test_v04_compilation_plan_recipe_loads_through_schema() -> None:
    plan = load_compilation_plan(Path("recipes/deployment/agx-orin-tensorrt-int8.yaml"))
    assert plan.compiler == "tensorrt"
    assert plan.precision == "int8"
    assert plan.target.compute_capability == "8.7"
    assert plan.shape_profiles[0].minimum == [1, 33, 256]


def test_shape_and_capability_contracts_fail_fast() -> None:
    with pytest.raises(ValueError, match="min <= opt <= max"):
        ShapeProfile(input_name="tokens", minimum=[1, 64], optimum=[1, 32], maximum=[1, 128])
    with pytest.raises(ValueError, match="require reason_code"):
        TargetCapability(capability_id="fp8", target=_target(), feature="fp8-parser", precision="fp8", status="unsupported")


def test_latency_and_deployment_manifest_preserve_acceptance_boundary() -> None:
    latency = latency_distribution([4.0, 2.0, 8.0, 6.0])
    assert latency.min_ms <= latency.p50_ms <= latency.p95_ms <= latency.p99_ms <= latency.max_ms
    protocol = BenchmarkProtocol(protocol_id="standalone", timing_boundary="standalone", warmup=5, repeat=4, synchronization="cuda-event")
    timing = StageTimingReport(
        report_id="fastwam-action-loop",
        status="measured",
        stage_name="action_loop",
        target=_target(),
        protocol=protocol,
        latency=latency,
    )
    manifest = DeploymentCandidateManifest(
        manifest_id="candidate",
        status="pending",
        model=_model(),
        target=_target(),
        precision_map={"action_loop": "int8", "head": "fp16"},
        timing_reports=[timing],
        notes=["machine evidence is not human acceptance"],
    )
    summary = summarize_deployment_manifest(manifest)
    assert summary["human_acceptance"] == "pending"
    assert summary["precisions"] == ["fp16", "int8"]
    with pytest.raises(ValueError, match="human_acceptance=accepted"):
        DeploymentCandidateManifest(manifest_id="bad", status="accepted", model=_model(), target=_target())


def test_tensorrt_command_and_layer_summary_are_offline(tmp_path: Path) -> None:
    plan = CompilationPlan(
        plan_id="fastwam-int8",
        compiler="tensorrt",
        source_artifact=_source(),
        target=_target(),
        precision="int8",
        strongly_typed=True,
        workspace_mib=1024,
        builder_optimization_level=4,
        shape_profiles=[ShapeProfile(input_name="tokens", minimum=[1, 33, 256], optimum=[1, 33, 256], maximum=[1, 33, 256])],
        timing_boundary="engine_stage",
    )
    command = build_trtexec_command(plan, engine_path=tmp_path / "candidate.engine", layer_info_path=tmp_path / "layers.json")
    assert "--int8" in command
    assert "--stronglyTyped" in command
    assert "--skipInference" in command
    assert "--minShapes=tokens:1x33x256" in command

    layer_info = tmp_path / "layers.json"
    layer_info.write_text(
        json.dumps(
            {
                "layers": [
                    {"Name": "QuantizeLinear_1", "LayerType": "Quantize", "Precision": "INT8", "TacticName": "tc"},
                    {"Name": "gemm + relu", "LayerType": "FullyConnected", "Precision": "FP16", "TacticName": "tc"},
                    {"Name": "Reformatting CopyNode", "LayerType": "Reformat", "Precision": "FP16"},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = summarize_tensorrt_layers(layer_info)
    assert report.layer_count == 3
    assert report.qdq_layer_count == 1
    assert report.reformat_layer_count == 1
    assert report.copy_layer_count == 1
    assert report.fused_layer_count == 1


@pytest.mark.onnx
def test_onnx_inspection_reports_constant_weight_matmul(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    tensor = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])
    output = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])
    weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["x", "weight"], ["y"], name="linear")],
        "inspect-graph",
        [tensor],
        [output],
        [weight],
    )
    model = onnx.helper.make_model(graph, producer_name="piquant-test", opset_imports=[onnx.helper.make_opsetid("", 17)])
    model_path = tmp_path / "inspect.onnx"
    onnx.save(model, str(model_path))

    report = inspect_onnx_model(model_path)
    assert report.producer == "piquant-test"
    assert report.op_counts == {"MatMul": 1}
    assert report.initializer_names == ["weight"]
    assert report.constant_weight_candidates == ["linear"]
    assert report.source is not None


def test_tensorrt_cli_compiler_reports_unsupported_without_importing_tensorrt(tmp_path: Path) -> None:
    plan = CompilationPlan(
        plan_id="no-trtexec",
        compiler="tensorrt",
        source_artifact=_source(),
        target=_target(),
        precision="fp16",
        timing_boundary="engine_stage",
    )
    record = TensorRTCliCompiler(output_dir=tmp_path, trtexec="definitely-not-a-real-trtexec").compile(plan, _model())
    assert record.status == "unsupported"
    assert record.reason_code == "trtexec-unavailable"
    assert "tensorrt" not in sys.modules
