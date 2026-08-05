"""Portable QDQ reference used to exercise the SDK without accelerator packages."""

from __future__ import annotations

from piquant.adapters.synthetic_flow_vla import SyntheticFlowVLAAdapter
from piquant.contracts import ModuleCoverage, OptimizationPlan, QuantizationResult
from piquant.interfaces import CalibrationProvider, ModelAdapter
from piquant.matching import require_matches, select_modules


class ReferenceQDQBackend:
    """A deterministic test/reference path, never a claim about deployment kernels."""

    name = "reference_qdq"

    def inspect(self, adapter: ModelAdapter, plan: OptimizationPlan) -> ModuleCoverage:
        coverage = select_modules(adapter.named_modules(), plan.module_include, plan.module_exclude)
        return require_matches(coverage, plan.module_include)

    def quantize(
        self,
        adapter: ModelAdapter,
        plan: OptimizationPlan,
        calibration: CalibrationProvider,
    ) -> tuple[ModelAdapter, QuantizationResult]:
        del calibration
        coverage = self.inspect(adapter, plan)
        if not isinstance(adapter, SyntheticFlowVLAAdapter):
            raise TypeError("ReferenceQDQBackend only supports SyntheticFlowVLAAdapter in v0.1")
        candidate = adapter.clone_with_reference_qdq(set(coverage.matched_names))
        return candidate, QuantizationResult(
            backend=self.name,
            representation="reference_qdq",
            status="measured",
            quant_format=plan.quant_format,
            module_coverage=coverage,
            metadata={"warning": "portable reference only; no deployment claim"},
        )
