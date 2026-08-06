"""Optional TensorRT CLI integration for target compiler evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from piquant.contracts import (
    ArtifactRef,
    CompilationPlan,
    CompiledArtifactRef,
    CompilerEvidenceRecord,
    ModelSpec,
    PrecisionMode,
    TensorRTLayerReport,
)
from piquant.inspection import inspect_onnx_model, sha256_file


def _shape_arg(values: list[int]) -> str:
    return "x".join(str(value) for value in values)


def _precision_flag(precision: PrecisionMode) -> str | None:
    flags = {"fp16": "--fp16", "bf16": "--bf16", "int8": "--int8", "fp8": "--fp8"}
    if precision == "nvfp4":
        raise ValueError("TensorRT CLI NVFP4 support must be proven by a target-specific integration before use")
    return flags.get(precision)


def build_trtexec_command(
    plan: CompilationPlan,
    *,
    engine_path: str | Path,
    layer_info_path: str | Path,
    skip_inference: bool = True,
) -> list[str]:
    if plan.compiler != "tensorrt":
        raise ValueError("TensorRT command builder requires compiler=tensorrt")
    command = ["trtexec", f"--onnx={plan.source_artifact.path}", f"--saveEngine={engine_path}", f"--exportLayerInfo={layer_info_path}"]
    if flag := _precision_flag(plan.precision):
        command.append(flag)
    if plan.strongly_typed:
        command.append("--stronglyTyped")
    if plan.workspace_mib is not None:
        command.append(f"--memPoolSize=workspace:{plan.workspace_mib}MiB")
    if plan.builder_optimization_level is not None:
        command.append(f"--builderOptimizationLevel={plan.builder_optimization_level}")
    if plan.timing_cache is not None:
        command.append(f"--timingCacheFile={plan.timing_cache.path}")
    for profile in plan.shape_profiles:
        command.extend(
            [
                f"--minShapes={profile.input_name}:{_shape_arg(profile.minimum)}",
                f"--optShapes={profile.input_name}:{_shape_arg(profile.optimum)}",
                f"--maxShapes={profile.input_name}:{_shape_arg(profile.maximum)}",
            ]
        )
    if skip_inference:
        command.append("--skipInference")
    command.extend(plan.flags)
    return command


def summarize_tensorrt_layers(path: str | Path) -> TensorRTLayerReport:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_layers: Any
    raw_layers = data.get("layers", data.get("Layers", [])) if isinstance(data, dict) else data
    if not isinstance(raw_layers, list):
        raise ValueError("TensorRT layer info JSON must contain a layer list")
    dtype_counts: dict[str, int] = {}
    tactic_names: set[str] = set()
    qdq_count = 0
    reformat_count = 0
    copy_count = 0
    fused_count = 0
    unsupported: list[str] = []
    for index, layer in enumerate(raw_layers):
        if not isinstance(layer, dict):
            raise ValueError(f"TensorRT layer entry {index} is not a mapping")
        name = str(layer.get("Name", layer.get("name", f"layer-{index}")))
        layer_type = str(layer.get("LayerType", layer.get("type", "")))
        dtype = str(layer.get("Precision", layer.get("precision", layer.get("OutputType", "unknown"))))
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        tactic = layer.get("TacticName", layer.get("tacticName"))
        if tactic:
            tactic_names.add(str(tactic))
        lowered = f"{name} {layer_type}".lower()
        qdq_count += int("quantize" in lowered or "dequantize" in lowered)
        reformat_count += int("reformat" in lowered)
        copy_count += int("copy" in lowered)
        fused_count += int(" + " in name or "||" in name)
        if "unsupported" in lowered:
            unsupported.append(name)
    return TensorRTLayerReport(
        layer_count=len(raw_layers),
        dtype_counts=dict(sorted(dtype_counts.items())),
        tactic_count=len(tactic_names),
        qdq_layer_count=qdq_count,
        reformat_layer_count=reformat_count,
        copy_layer_count=copy_count,
        fused_layer_count=fused_count,
        unsupported_layers=unsupported,
        source=ArtifactRef(kind="tensorrt-layer-info", path=str(path), sha256=sha256_file(path)),
    )


class TensorRTCliCompiler:
    """Compile one ONNX candidate with system trtexec when the target environment provides it."""

    name = "tensorrt"

    def __init__(self, *, output_dir: str | Path, trtexec: str = "trtexec") -> None:
        self.output_dir = Path(output_dir)
        self.trtexec = trtexec

    def compile(self, plan: CompilationPlan, model: ModelSpec) -> CompilerEvidenceRecord:
        if plan.compiler != self.name:
            raise ValueError("TensorRTCliCompiler requires compiler=tensorrt")
        executable = shutil.which(self.trtexec)
        if executable is None:
            return CompilerEvidenceRecord(
                record_id=f"{plan.plan_id}-unsupported",
                status="unsupported",
                model=model,
                compilation=plan,
                reason_code="trtexec-unavailable",
                notes=["No TensorRT build was attempted because trtexec is not on PATH"],
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        engine_path = self.output_dir / f"{plan.plan_id}.engine"
        layer_info_path = self.output_dir / f"{plan.plan_id}.layers.json"
        log_path = self.output_dir / f"{plan.plan_id}.trtexec.log"
        command = build_trtexec_command(plan, engine_path=engine_path, layer_info_path=layer_info_path, skip_inference=True)
        started = time.monotonic()
        result = subprocess.run(command, cwd=self.output_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        elapsed = time.monotonic() - started
        log_path.write_text(result.stdout, encoding="utf-8")
        graph = inspect_onnx_model(plan.source_artifact.path)
        if result.returncode != 0 or not engine_path.exists():
            return CompilerEvidenceRecord(
                record_id=f"{plan.plan_id}-rejected",
                status="rejected",
                model=model,
                compilation=plan,
                graph=graph,
                build_time_seconds=elapsed,
                commands=[" ".join(command)],
                notes=[f"trtexec log: {log_path}"],
                reason_code=f"trtexec-exit-{result.returncode}",
            )
        engine_ref = CompiledArtifactRef(
            artifact_id=f"{plan.plan_id}-engine",
            artifact=ArtifactRef(kind="tensorrt-engine", path=str(engine_path), sha256=sha256_file(engine_path)),
            compiler=self.name,
            precision=plan.precision,
            target=plan.target,
            source_sha256=plan.source_artifact.sha256,
            status="measured",
        )
        log_ref = ArtifactRef(kind="tensorrt-build-log", path=str(log_path), sha256=sha256_file(log_path))
        return CompilerEvidenceRecord(
            record_id=f"{plan.plan_id}-measured",
            status="measured",
            model=model,
            compilation=plan,
            artifacts=[engine_ref],
            graph=graph,
            layer_report=summarize_tensorrt_layers(layer_info_path),
            build_time_seconds=elapsed,
            commands=[" ".join(command)],
            notes=[f"TensorRT layer report: {layer_info_path}", f"TensorRT build log: {log_ref.path}"],
        )
