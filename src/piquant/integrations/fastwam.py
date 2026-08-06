"""Lazy FastWAM source execution helpers.

The integration deliberately accepts model-specific teacher-forcing callbacks. The
audited FastWAM inference API exposes iterative action inference, while its
training path does not expose a comparable final-action capture contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from piquant.contracts import TemporalCaptureSpec, TemporalMode

FastWAMForward = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
FastWAMCaptureProvider = Callable[
    [Any, Mapping[str, Any], Sequence[TemporalCaptureSpec], TemporalMode],
    Mapping[str, Any],
]


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("FastWAM integration requires the external PyTorch/FastWAM environment") from error
    return torch


def _cpu_array(torch: Any, value: Any) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(
            "FastWAM capture output must be one tensor; nested tuple/mapping outputs require an injected "
            f"capture_provider, got {type(value).__name__}"
        )
    tensor = value
    return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()


def _capture_array(torch: Any, value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return cast(np.ndarray, _cpu_array(torch, value))
    return np.asarray(value, dtype=np.float32)


def _batch_first(torch: Any, value: Any, contract: FastWAMInferenceContract) -> Any:
    if not torch.is_tensor(value):
        raise TypeError(
            "FastWAM action output must be one tensor; nested tuple/mapping outputs require an injected "
            f"iterative_runner, got {type(value).__name__}"
        )
    tensor = value
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    expected = (1, contract.action_horizon, contract.action_dim)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"FastWAM action output must match the single-sample source ABI {expected}, got {tuple(tensor.shape)}")
    return tensor


@dataclass(frozen=True)
class FastWAMInferenceContract:
    """Fixed source inference defaults for the audited FastWAM LIBERO ABI."""

    action_horizon: int = 32
    action_dim: int = 7
    num_video_frames: int = 33
    num_inference_steps: int = 10


class FastWAMCaptureRunner:
    """Run one FastWAM sample while collecting explicit module outputs.

    ``teacher_forced_runner`` is required for teacher-forced studies. It must
    execute the source model with the same hooks active and return a mapping
    containing ``action`` when an action capture is requested.
    """

    def __init__(
        self,
        *,
        contract: FastWAMInferenceContract | None = None,
        teacher_forced_runner: FastWAMForward | None = None,
        iterative_runner: FastWAMForward | None = None,
        capture_provider: FastWAMCaptureProvider | None = None,
    ) -> None:
        self.contract = contract or FastWAMInferenceContract()
        self.teacher_forced_runner = teacher_forced_runner
        self.iterative_runner = iterative_runner
        self.capture_provider = capture_provider

    def _iterative(self, model: Any, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.iterative_runner is not None:
            return self.iterative_runner(model, batch)
        infer_action = getattr(model, "infer_action", None)
        if not callable(infer_action):
            raise TypeError("FastWAM model must expose infer_action or an injected iterative_runner")
        required = ("input_image",)
        missing = [name for name in required if name not in batch]
        if missing:
            raise KeyError(f"FastWAM iterative batch is missing required fields: {missing!r}")
        if "prompt" not in batch and not {"context", "context_mask"}.issubset(batch):
            raise KeyError("FastWAM iterative batch requires prompt or both context and context_mask")
        kwargs: dict[str, Any] = {
            "prompt": batch.get("prompt"),
            "input_image": batch["input_image"],
            "action_horizon": int(batch.get("action_horizon", self.contract.action_horizon)),
            "num_video_frames": int(batch.get("num_video_frames", self.contract.num_video_frames)),
            "num_inference_steps": int(batch.get("num_inference_steps", self.contract.num_inference_steps)),
            "proprio": batch.get("proprio"),
            "context": batch.get("context"),
            "context_mask": batch.get("context_mask"),
            "seed": batch.get("seed"),
            "rand_device": str(batch.get("rand_device", "cpu")),
            "sigma_shift": batch.get("sigma_shift"),
            "tiled": bool(batch.get("tiled", False)),
        }
        return cast(Mapping[str, Any], infer_action(**kwargs))

    def _forward(self, model: Any, batch: Mapping[str, Any], mode: TemporalMode) -> Mapping[str, Any]:
        if mode == "teacher_forced":
            if self.teacher_forced_runner is None:
                raise RuntimeError("teacher-forced FastWAM diagnostics require an injected teacher_forced_runner")
            return self.teacher_forced_runner(model, batch)
        return self._iterative(model, batch)

    def run(
        self,
        model: Any,
        batch: Mapping[str, Any],
        capture_specs: Sequence[TemporalCaptureSpec],
        *,
        mode: TemporalMode,
    ) -> dict[str, Any]:
        torch = _torch()
        if self.capture_provider is not None:
            provided = self.capture_provider(model, batch, capture_specs, mode)
            provided_result: dict[str, Any] = {}
            for spec in capture_specs:
                value = provided.get(spec.logical_id)
                if value is None and spec.backend_path == "$infer_action":
                    value = provided.get("action")
                if value is None:
                    raise ValueError(f"FastWAM capture provider did not return {spec.logical_id!r}")
                array = _capture_array(torch, value)
                if spec.backend_path == "$infer_action":
                    if array.ndim == 2:
                        array = array[None, ...]
                    expected = (1, self.contract.action_horizon, self.contract.action_dim)
                    if tuple(array.shape) != expected:
                        raise ValueError(f"FastWAM capture provider action must match {expected}, got {tuple(array.shape)}")
                provided_result[spec.logical_id] = array
            return provided_result
        modules = dict(model.named_modules())
        values: dict[str, list[Any]] = {spec.logical_id: [] for spec in capture_specs if not spec.backend_path.startswith("$")}
        handles = []

        def make_hook(logical_id: str) -> Callable[[Any, Any, Any], None]:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                values[logical_id].append(_cpu_array(torch, output))

            return hook

        for spec in capture_specs:
            if spec.backend_path.startswith("$"):
                continue
            module = modules.get(spec.backend_path)
            if module is None:
                raise ValueError(f"FastWAM capture path does not resolve: {spec.backend_path!r}")
            handles.append(module.register_forward_hook(make_hook(spec.logical_id)))
        try:
            output = self._forward(model, batch, mode)
        finally:
            for handle in handles:
                handle.remove()

        result: dict[str, Any] = {}
        for spec in capture_specs:
            if spec.backend_path == "$infer_action":
                if not isinstance(output, Mapping) or "action" not in output:
                    raise ValueError("FastWAM forward output must contain `action` for the final action capture")
                result[spec.logical_id] = _cpu_array(torch, _batch_first(torch, output["action"], self.contract))
                continue
            captured = values.get(spec.logical_id, [])
            if not captured:
                raise ValueError(f"FastWAM capture produced no output for {spec.logical_id!r}")
            result[spec.logical_id] = captured[0] if len(captured) == 1 else np.stack(captured, axis=1)
        return result
