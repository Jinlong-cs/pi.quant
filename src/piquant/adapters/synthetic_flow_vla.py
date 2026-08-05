"""A deterministic NumPy flow-action adapter used for offline SDK validation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from piquant.contracts import ModelSpec, fingerprint
from piquant.interfaces import CalibrationProvider, ModelAdapter, TaskLossProvider

CAPTURE_POINTS = ("vision", "language", "projector", "history", "action_hidden", "action")


def _relu(value: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(value, 0.0))


def _qdq(value: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(value))) / 127.0
    if scale == 0.0:
        return value.copy()
    return np.clip(np.rint(value / scale), -127, 127) * scale


@dataclass
class _Linear:
    weight: np.ndarray
    bias: np.ndarray


class SyntheticFlowVLAAdapter:
    """Small VLA-shaped graph with explicit vision, language, and action nodes."""

    def __init__(self, seed: int = 7, quantized_modules: set[str] | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._quantized_modules = set(quantized_modules or ())
        self._modules = {
            "vision.patch_embedding": self._linear(4, 8),
            "language.embedding": self._linear(4, 8),
            "projector": self._linear(16, 8),
            "history_encoder": self._linear(4, 8),
            "action_encoder": self._linear(21, 16),
            "action_head": self._linear(16, 6),
        }
        self._spec = ModelSpec(
            model_id="synthetic-flow-vla",
            family="synthetic_flow_action",
            framework="numpy-reference",
            action_dim=6,
            action_horizon=1,
        )

    def _linear(self, input_dim: int, output_dim: int) -> _Linear:
        return _Linear(
            weight=(self._rng.standard_normal((output_dim, input_dim)) * 0.15).astype(np.float64),
            bias=np.zeros(output_dim, dtype=np.float64),
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def named_modules(self) -> dict[str, _Linear]:
        return dict(self._modules)

    def _apply_linear(self, name: str, value: np.ndarray) -> np.ndarray:
        module = self._modules[name]
        weight = module.weight
        input_value = value
        if name in self._quantized_modules:
            weight = _qdq(weight)
            input_value = _qdq(input_value)
        output = input_value @ weight.T + module.bias
        return _qdq(output) if name in self._quantized_modules else output

    def forward(self, batch: dict[str, Any], capture_points: tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
        observation = np.asarray(batch["observation"], dtype=np.float64)
        language = np.asarray(batch["language"], dtype=np.float64)
        history = np.asarray(batch["history"], dtype=np.float64)
        noise = np.asarray(batch["noise"], dtype=np.float64)
        timestep = np.asarray(batch["timestep"], dtype=np.float64)
        vision = _relu(self._apply_linear("vision.patch_embedding", observation))
        language_hidden = _relu(self._apply_linear("language.embedding", language))
        projector = _relu(self._apply_linear("projector", np.concatenate([vision, language_hidden], axis=-1)))
        history_hidden = _relu(self._apply_linear("history_encoder", history))
        action_input = np.concatenate([projector, history_hidden, noise, timestep], axis=-1)
        action_hidden = _relu(self._apply_linear("action_encoder", action_input))
        action = self._apply_linear("action_head", action_hidden)
        all_outputs = {
            "vision": vision,
            "language": language_hidden,
            "projector": projector,
            "history": history_hidden,
            "action_hidden": action_hidden,
            "action": action,
        }
        return {name: all_outputs[name] for name in capture_points}

    def clone_with_reference_qdq(self, module_names: set[str]) -> SyntheticFlowVLAAdapter:
        """Create a portable reference candidate; it is never labeled ModelOpt."""

        clone = deepcopy(self)
        clone._quantized_modules = set(module_names)
        return clone


class SyntheticCalibrationProvider:
    """Fixed observation/language/history/action-stage samples for calibration."""

    def __init__(self, sample_count: int = 16, seed: int = 11) -> None:
        self.sample_count = sample_count
        self.seed = seed
        rng = np.random.default_rng(seed)
        stages = ("approach", "grasp", "lift", "place")
        self._batches = []
        for index in range(sample_count):
            observation = rng.normal(size=(4,))
            language = rng.normal(size=(4,))
            history = rng.normal(size=(4,))
            noise = rng.normal(size=(4,))
            timestep = np.asarray([(index % 8) / 7.0], dtype=np.float64)
            target_action = np.asarray(
                [
                    observation[0] + history[0],
                    observation[1] - history[1],
                    language[0],
                    language[1],
                    noise[0] * 0.1,
                    1.0 if stages[index % len(stages)] in {"grasp", "lift"} else 0.0,
                ],
                dtype=np.float64,
            )
            self._batches.append(
                {
                    "observation": observation[None, :],
                    "language": language[None, :],
                    "history": history[None, :],
                    "noise": noise[None, :],
                    "timestep": timestep[None, :],
                    "target_action": target_action[None, :],
                    "stage": stages[index % len(stages)],
                }
            )

    @property
    def fingerprint(self) -> str:
        return fingerprint({"dataset_id": "synthetic-flow-vla", "sample_count": self.sample_count, "seed": self.seed})

    def batches(self) -> list[dict[str, Any]]:
        return [deepcopy(batch) for batch in self._batches]

    def forward_loop(self, model: Any) -> None:
        """Run deterministic batches through a torch-like callable for calibration."""

        for batch in self._batches:
            model(batch)


class SyntheticFlowLoss:
    """Flow-action surrogate used only for offline candidate diagnostics."""

    def compute(self, outputs: Mapping[str, Any], batch: Mapping[str, Any]) -> float:
        delta = np.asarray(outputs["action"]) - np.asarray(batch["target_action"])
        return float(np.mean(delta * delta))


class OfflineActionEvaluator:
    """Evaluate action loss over the fixed calibration samples."""

    def __init__(self, loss: TaskLossProvider | None = None) -> None:
        self.loss = loss or SyntheticFlowLoss()

    def evaluate(self, adapter: ModelAdapter, calibration: CalibrationProvider) -> dict[str, float]:
        losses = [self.loss.compute(adapter.forward(batch, ("action",)), batch) for batch in calibration.batches()]
        return {"action_mse_mean": float(np.mean(losses)), "sample_count": float(len(losses))}
