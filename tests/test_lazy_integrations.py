import sys

from piquant.backends import ModelOptBackend
from piquant.integrations import OrtDebugCapture


def test_optional_integrations_are_lazy() -> None:
    assert "modelopt" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert ModelOptBackend.name == "modelopt"
    assert OrtDebugCapture is not None
