import sys

from piquant.adapters import FastWAMSourceAdapter, Pi05TorchAdapter
from piquant.backends import ModelOptBackend
from piquant.integrations import FastWAMInferenceContract, OrtDebugCapture, Pi05OpenPIConfig


def test_optional_integrations_are_lazy() -> None:
    assert "modelopt" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert "openpi" not in sys.modules
    assert "fastwam" not in sys.modules
    assert "pyarrow" not in sys.modules
    assert "torch" not in sys.modules
    assert ModelOptBackend.name == "modelopt"
    assert OrtDebugCapture is not None
    assert Pi05TorchAdapter is not None
    assert Pi05OpenPIConfig is not None
    assert FastWAMSourceAdapter is not None
    assert FastWAMInferenceContract is not None
