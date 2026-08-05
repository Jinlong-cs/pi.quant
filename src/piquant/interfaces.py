"""Small dependency-injection boundaries for model and backend integrations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from piquant.contracts import (
    ComparisonReport,
    ModelSpec,
    ModuleCoverage,
    OptimizationPlan,
    QuantizationResult,
)

TensorMap = Mapping[str, Any]


class ModelAdapter(Protocol):
    """Expose model identity, named modules, captures, and action output."""

    @property
    def spec(self) -> ModelSpec: ...

    def named_modules(self) -> Mapping[str, Any]: ...

    def forward(self, batch: Mapping[str, Any], capture_points: Sequence[str]) -> TensorMap: ...


class CalibrationProvider(Protocol):
    """Provide deterministic, stage-aware calibration batches."""

    @property
    def fingerprint(self) -> str: ...

    def batches(self) -> Iterable[Mapping[str, Any]]: ...

    def forward_loop(self, model: Any) -> None: ...


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


class TaskEvaluator(Protocol):
    """Evaluate a candidate in the currently supported offline task boundary."""

    def evaluate(self, adapter: ModelAdapter, calibration: CalibrationProvider) -> dict[str, float]: ...


class EvidenceStore(Protocol):
    """Persist machine-readable evidence without deciding human promotion."""

    def write(self, record: Any, path: str) -> None: ...
