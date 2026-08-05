"""Lazy ModelOpt adapter for explicit INT8 PTQ integration."""

from __future__ import annotations

import copy
from typing import Any, cast

from piquant.contracts import ModuleCoverage, OptimizationPlan, QuantizationResult
from piquant.interfaces import CalibrationProvider, ModelAdapter
from piquant.matching import require_matches, select_modules


class OptionalDependencyError(RuntimeError):
    """Raised when an explicitly selected optional backend is unavailable."""


class ModelOptBackend:
    """Use ModelOpt's public ``mtq.quantize`` API without importing it at module load."""

    name = "modelopt"

    def inspect(self, adapter: ModelAdapter, plan: OptimizationPlan) -> ModuleCoverage:
        coverage = select_modules(adapter.named_modules(), plan.module_include, plan.module_exclude)
        return require_matches(coverage, plan.module_include)

    @staticmethod
    def _imports() -> tuple[Any, Any]:
        try:
            import modelopt.torch.quantization as mtq
            import torch
        except ModuleNotFoundError as error:
            raise OptionalDependencyError(
                "ModelOpt backend requires the optional 'modelopt' extra; install nvidia-modelopt==0.45.0 in a compatible environment"
            ) from error
        return mtq, torch

    @staticmethod
    def _config(mtq: Any, module_names: list[str]) -> dict[str, Any]:
        """Build ordered config entries following ModelOpt 0.45 precedence rules."""

        config = cast(dict[str, Any], copy.deepcopy(mtq.INT8_DEFAULT_CFG))
        quant_cfg = [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": 8, "axis": 0}},
            {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": 8, "axis": None}},
        ]
        for module_name in module_names:
            quant_cfg.append({"quantizer_name": f"*{module_name}*weight_quantizer", "enable": True})
            quant_cfg.append({"quantizer_name": f"*{module_name}*input_quantizer", "enable": True})
        config["quant_cfg"] = quant_cfg
        config["algorithm"] = "max"
        return config

    def quantize(
        self,
        adapter: ModelAdapter,
        plan: OptimizationPlan,
        calibration: CalibrationProvider,
    ) -> tuple[Any, QuantizationResult]:
        mtq, _torch = self._imports()
        coverage = self.inspect(adapter, plan)
        if not hasattr(adapter, "backend_model") or not hasattr(adapter, "forward_backend"):
            raise TypeError("ModelOptBackend requires an adapter with backend_model and forward_backend")
        model = adapter.backend_model()
        config = self._config(mtq, coverage.matched_names)

        def forward_loop(runtime_model: Any) -> None:
            for batch in calibration.batches():
                adapter.forward_backend(runtime_model, batch)

        quantized_model = mtq.quantize(model, config, forward_loop=forward_loop)
        return quantized_model, QuantizationResult(
            backend=self.name,
            representation=plan.representation,
            status="measured",
            quant_format=plan.quant_format,
            module_coverage=coverage,
            metadata={"modelopt_api": "mtq.quantize", "modelopt_version": "0.45.0"},
        )
