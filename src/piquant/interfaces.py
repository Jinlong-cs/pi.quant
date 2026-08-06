"""Small dependency-injection boundaries for model and backend integrations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from piquant.contracts import (
    ActionSchema,
    BenchmarkProtocol,
    CalibrationManifest,
    CaptureSpec,
    ComparisonReport,
    CompilationPlan,
    CompiledArtifactRef,
    CompilerEvidenceRecord,
    DeploymentCandidateManifest,
    ModelSpec,
    ModuleCoverage,
    ModuleDescriptor,
    OptimizationPlan,
    QuantizationResult,
    SensitivityDiagnostics,
    StageTimingReport,
    TemporalCalibrationManifest,
    TemporalCaptureSpec,
    TemporalMetricReport,
    TemporalMode,
)

TensorMap = Mapping[str, Any]


class ModelAdapter(Protocol):
    """Expose model identity, named modules, captures, and action output."""

    @property
    def spec(self) -> ModelSpec: ...

    def named_modules(self) -> Mapping[str, Any]: ...

    def forward(self, batch: Mapping[str, Any], capture_points: Sequence[str]) -> TensorMap: ...


@runtime_checkable
class SemanticModelAdapter(ModelAdapter, Protocol):
    """Real-model boundary with stable logical module and capture identities."""

    @property
    def action_schema(self) -> ActionSchema: ...

    def module_inventory(self) -> Sequence[ModuleDescriptor]: ...

    def capture_specs(self) -> Sequence[CaptureSpec]: ...

    def fresh(self) -> SemanticModelAdapter: ...


@runtime_checkable
class TemporalModelAdapter(SemanticModelAdapter, Protocol):
    """Narrow WAM capability with explicit teacher-forced and iterative paths."""

    def temporal_capture_specs(self) -> Sequence[TemporalCaptureSpec]: ...

    def forward_temporal(
        self,
        batch: Mapping[str, Any],
        capture_points: Sequence[str],
        *,
        mode: TemporalMode,
    ) -> TensorMap: ...


@runtime_checkable
class TorchQuantizableAdapter(SemanticModelAdapter, Protocol):
    """Narrow capability required by the Torch/ModelOpt backend."""

    def backend_model(self) -> Any: ...

    def forward_backend(self, model: Any, batch: Mapping[str, Any]) -> None: ...

    def with_backend_model(self, model: Any) -> TorchQuantizableAdapter: ...


@runtime_checkable
class TemporalTorchQuantizableAdapter(TemporalModelAdapter, TorchQuantizableAdapter, Protocol):
    """Narrow intersection required for temporal ModelOpt trials."""

    def with_backend_model(self, model: Any) -> TemporalTorchQuantizableAdapter: ...


class CalibrationProvider(Protocol):
    """Provide deterministic, stage-aware calibration batches."""

    @property
    def fingerprint(self) -> str: ...

    def batches(self) -> Iterable[Mapping[str, Any]]: ...

    def forward_loop(self, model: Any) -> None: ...


@runtime_checkable
class ManifestCalibrationProvider(CalibrationProvider, Protocol):
    @property
    def manifest(self) -> CalibrationManifest: ...


@runtime_checkable
class TemporalManifestCalibrationProvider(CalibrationProvider, Protocol):
    @property
    def manifest(self) -> TemporalCalibrationManifest: ...


class TaskLossProvider(Protocol):
    """Compute a task-shaped loss for offline candidate comparison."""

    def compute(self, outputs: TensorMap, batch: Mapping[str, Any]) -> float: ...


class QuantizationBackend(Protocol):
    """Apply one explicit optimization backend; no implicit backend registry."""

    name: str

    def inspect(self, adapter: ModelAdapter, plan: OptimizationPlan) -> ModuleCoverage: ...

    def quantize(
        self,
        adapter: ModelAdapter,
        plan: OptimizationPlan,
        calibration: CalibrationProvider,
    ) -> tuple[Any, QuantizationResult]: ...


class NumericalAnalyzer(Protocol):
    """Compare named intermediate tensors and action outputs."""

    def compare(self, reference: TensorMap, candidate: TensorMap, action_name: str) -> ComparisonReport: ...


class StreamingNumericalAnalyzer(NumericalAnalyzer, Protocol):
    def add(
        self,
        reference: TensorMap,
        candidate: TensorMap,
        sample_metadata: Sequence[Mapping[str, Any]],
        *,
        action_name: str,
        flow_name: str | None = None,
        action_schema: ActionSchema,
    ) -> None: ...

    def finalize(self) -> SensitivityDiagnostics: ...


class StreamingTemporalAnalyzer(Protocol):
    def add(
        self,
        reference: TensorMap,
        candidate: TensorMap,
        sequence_metadata: Sequence[Mapping[str, Any]],
        capture_specs: Sequence[TemporalCaptureSpec],
        *,
        mode: TemporalMode,
        action_schema: ActionSchema,
    ) -> None: ...

    def finalize(self) -> list[TemporalMetricReport]: ...


class TaskEvaluator(Protocol):
    """Evaluate a candidate in the currently supported offline task boundary."""

    def evaluate(self, adapter: ModelAdapter, calibration: CalibrationProvider) -> dict[str, float]: ...


class EvidenceStore(Protocol):
    """Persist machine-readable evidence without deciding human promotion."""

    def write(self, record: Any, path: str) -> None: ...


class CompilerBackend(Protocol):
    """Compile one frozen candidate artifact for one explicit target."""

    name: str

    def compile(self, plan: CompilationPlan, model: ModelSpec) -> CompilerEvidenceRecord: ...


class DeploymentEvaluator(Protocol):
    """Measure a compiled artifact without deciding acceptance."""

    def benchmark(self, artifact: CompiledArtifactRef, protocol: BenchmarkProtocol) -> StageTimingReport: ...


class DeploymentManifestBuilder(Protocol):
    """Build a pi.cpp handoff manifest from measured evidence."""

    def build(self, manifest: DeploymentCandidateManifest) -> DeploymentCandidateManifest: ...


AdapterFactory = Callable[[], SemanticModelAdapter]
TemporalAdapterFactory = Callable[[], TemporalModelAdapter]
