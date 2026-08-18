import importlib.metadata
import importlib.resources
import sys

import pytest

from piquant import __version__
from piquant.adapters import FastWAMSourceAdapter, Pi05TorchAdapter
from piquant.backends import ModelOptBackend
from piquant.cli import build_parser
from piquant.integrations import FastWAMInferenceContract, OrtDebugCapture, Pi05OpenPIConfig


def test_optional_integrations_are_lazy() -> None:
    assert "modelopt" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert "openpi" not in sys.modules
    assert "fastwam" not in sys.modules
    assert "pyarrow" not in sys.modules
    assert "torch" not in sys.modules
    assert "tensorrt" not in sys.modules
    assert ModelOptBackend.name == "modelopt"
    assert OrtDebugCapture is not None
    assert Pi05TorchAdapter is not None
    assert Pi05OpenPIConfig is not None
    assert FastWAMSourceAdapter is not None
    assert FastWAMInferenceContract is not None


def test_v10_version_surface_matches_package_metadata(capsys: pytest.CaptureFixture[str]) -> None:
    assert __version__ == "1.0.0"
    assert importlib.metadata.version("pi-quant") == __version__
    assert importlib.resources.files("piquant").joinpath("py.typed").is_file()
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == "piquant 1.0.0"
    assert "torch" not in sys.modules
    assert "modelopt" not in sys.modules
    assert "onnxruntime" not in sys.modules
    assert "tensorrt" not in sys.modules
