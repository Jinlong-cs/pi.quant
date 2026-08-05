"""Lazy ModelOpt adapter for explicit INT8 PTQ integration."""

from __future__ import annotations

import copy
from fnmatch import fnmatchcase
from importlib.metadata import version
from typing import Any, cast

from piquant.contracts import ModuleCoverage, OptimizationPlan, QuantizationResult
from piquant.interfaces import CalibrationProvider, ModelAdapter, SemanticModelAdapter, TorchQuantizableAdapter
from piquant.matching import require_matches, select_modules


class OptionalDependencyError(RuntimeError):
    """Raised when an explicitly selected optional backend is unavailable."""


class ModelOptBackend:
    """Use ModelOpt's public ``mtq.quantize`` API without importing it at module load."""

    name = "modelopt"

    def inspect(self, adapter: ModelAdapter, plan: OptimizationPlan) -> ModuleCoverage:
        if isinstance(adapter, SemanticModelAdapter):
            return self._inspect_semantic(adapter, plan)
        coverage = select_modules(adapter.named_modules(), plan.module_include, plan.module_exclude)
        return require_matches(coverage, plan.module_include)

    @staticmethod
    def _inspect_semantic(adapter: SemanticModelAdapter, plan: OptimizationPlan) -> ModuleCoverage:
        inventory = [module for module in adapter.module_inventory() if module.quantizable]
        logical_ids = [module.logical_id for module in inventory]
        backend_paths = [module.backend_path for module in inventory]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("semantic module inventory contains duplicate logical ids")
        if len(backend_paths) != len(set(backend_paths)):
            raise ValueError("semantic module inventory contains duplicate backend paths")
        named_modules = adapter.named_modules()
        missing = sorted(set(backend_paths) - set(named_modules))
        if missing:
            raise ValueError(f"semantic module inventory resolves to missing backend paths: {missing!r}")
        matched = [
            module
            for module in inventory
            if any(fnmatchcase(module.logical_id, pattern) for pattern in plan.module_include)
            and not any(fnmatchcase(module.logical_id, pattern) for pattern in plan.module_exclude)
        ]
        excluded = [
            module
            for module in inventory
            if any(fnmatchcase(module.logical_id, pattern) for pattern in plan.module_include)
            and any(fnmatchcase(module.logical_id, pattern) for pattern in plan.module_exclude)
        ]
        coverage = ModuleCoverage(
            candidate_count=len(inventory),
            matched_count=len(matched),
            excluded_count=len(excluded),
            candidate_names=logical_ids,
            matched_names=[module.logical_id for module in matched],
            excluded_names=[module.logical_id for module in excluded],
            matched_logical_ids=[module.logical_id for module in matched],
            resolved_backend_names=[module.backend_path for module in matched],
            candidate_parameter_count=sum(module.parameter_count for module in inventory),
            matched_parameter_count=sum(module.parameter_count for module in matched),
        )
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
    def _tensor_summary(torch: Any, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        tensor = value.detach().to(dtype=torch.float32, device="cpu")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("ModelOpt quantizer metadata contains non-finite values")
        return {
            "shape": list(tensor.shape),
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
            "mean": float(tensor.mean()),
        }

    @classmethod
    def _quantizer_inventory(cls, mtq: Any, torch: Any, model: Any) -> tuple[list[str], list[dict[str, Any]]]:
        names: list[str] = []
        enabled: list[dict[str, Any]] = []
        for name, module in model.named_modules():
            if not isinstance(module, mtq.nn.TensorQuantizer):
                continue
            names.append(name)
            if not module.is_enabled:
                continue
            amax = module.amax
            num_bits = list(module.num_bits) if isinstance(module.num_bits, tuple) else module.num_bits
            axis = list(module.axis) if isinstance(module.axis, tuple) else module.axis
            enabled.append(
                {
                    "name": name,
                    "num_bits": num_bits,
                    "axis": axis,
                    "amax": cls._tensor_summary(torch, amax),
                    "scale": cls._tensor_summary(torch, None if amax is None else amax / float(module.maxbound)),
                    "pre_quant_scale": cls._tensor_summary(torch, module.pre_quant_scale),
                }
            )
        return names, enabled

    @staticmethod
    def _config(mtq: Any, module_names: list[str]) -> dict[str, Any]:
        """Build ordered config entries following ModelOpt 0.45 precedence rules."""

        config = cast(dict[str, Any], copy.deepcopy(mtq.INT8_DEFAULT_CFG))
        quant_cfg = [{"quantizer_name": "*", "enable": False}]
        for module_name in module_names:
            quant_cfg.append(
                {
                    "quantizer_name": f"{module_name}.weight_quantizer",
                    "cfg": {"num_bits": 8, "axis": 0},
                    "enable": True,
                }
            )
            quant_cfg.append(
                {
                    "quantizer_name": f"{module_name}.input_quantizer",
                    "cfg": {"num_bits": 8, "axis": None},
                    "enable": True,
                }
            )
        config["quant_cfg"] = quant_cfg
        config["algorithm"] = "max"
        return config

    def quantize(
        self,
        adapter: ModelAdapter,
        plan: OptimizationPlan,
        calibration: CalibrationProvider,
    ) -> tuple[Any, QuantizationResult]:
        if plan.backend != self.name or plan.quant_format != "int8" or plan.representation != "fake_quant":
            raise ValueError("ModelOptBackend requires backend=modelopt, quant_format=int8, and representation=fake_quant")
        mtq, torch = self._imports()
        modelopt_version = version("nvidia-modelopt")
        if modelopt_version != "0.45.0":
            raise RuntimeError(f"pi.quant requires nvidia-modelopt==0.45.0, found {modelopt_version}")
        coverage = self.inspect(adapter, plan)
        if not isinstance(adapter, TorchQuantizableAdapter):
            raise TypeError("ModelOptBackend requires a TorchQuantizableAdapter")
        model = adapter.backend_model()
        module_names = coverage.resolved_backend_names or coverage.matched_names
        config = self._config(mtq, module_names)

        def forward_loop(runtime_model: Any) -> None:
            for batch in calibration.batches():
                adapter.forward_backend(runtime_model, batch)

        quantized_model = mtq.quantize(model, config, forward_loop=forward_loop)
        quantizer_names, enabled_quantizers = self._quantizer_inventory(mtq, torch, quantized_model)
        enabled_names = {entry["name"] for entry in enabled_quantizers}
        expected_names = {
            quantizer_name
            for module_name in module_names
            for quantizer_name in (f"{module_name}.weight_quantizer", f"{module_name}.input_quantizer")
        }
        missing_quantizers = sorted(expected_names - enabled_names)
        unexpected_quantizers = sorted(enabled_names - expected_names)
        if missing_quantizers or unexpected_quantizers:
            raise ValueError(
                f"ModelOpt enabled quantizers differ from resolved module selection: "
                f"missing={missing_quantizers!r}, unexpected={unexpected_quantizers!r}"
            )
        return quantized_model, QuantizationResult(
            backend=self.name,
            representation=plan.representation,
            status="measured",
            quant_format=plan.quant_format,
            module_coverage=coverage,
            metadata={
                "modelopt_api": "mtq.quantize",
                "modelopt_version": modelopt_version,
                "inserted_quantizer_count": len(quantizer_names),
                "enabled_quantizer_count": len(enabled_quantizers),
                "inserted_quantizer_names": quantizer_names,
                "enabled_quantizers": enabled_quantizers,
            },
        )
