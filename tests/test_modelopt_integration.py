from __future__ import annotations

import pytest

from piquant.adapters import SyntheticCalibrationProvider, TorchSyntheticFlowVLAAdapter
from piquant.backends import ModelOptBackend
from piquant.contracts import load_plan


@pytest.mark.modelopt
def test_modelopt_int8_fake_quant_inserts_quantizers() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("modelopt.torch.quantization")
    plan = load_plan("recipes/synthetic/flow-vla-int8.yaml")
    calibration = SyntheticCalibrationProvider(plan.calibration.sample_count, plan.calibration.seed)
    adapter = TorchSyntheticFlowVLAAdapter(plan.seed)
    quantized_model, result = ModelOptBackend().quantize(adapter, plan, calibration)
    assert type(quantized_model).__name__ == "FlowModel"
    assert result.backend == "modelopt"
    assert result.representation == "fake_quant"
    assert result.module_coverage.matched_count == 6
    assert any("quantizer" in name for name, _module in quantized_model.named_modules())
