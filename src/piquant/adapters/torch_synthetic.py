"""Optional Torch adapter for the same synthetic graph used by ModelOpt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from piquant.contracts import ActionSchema, CaptureSpec, ModelSpec, ModuleDescriptor


class TorchSyntheticFlowVLAAdapter:
    """A torch-only adapter imported explicitly by the ModelOpt integration."""

    def __init__(self, seed: int = 7, model: Any | None = None) -> None:
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as error:
            raise RuntimeError("TorchSyntheticFlowVLAAdapter requires torch") from error

        self.seed = seed
        torch.manual_seed(seed)
        if model is None:

            class FlowModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.vision: Any = nn.Module()
                    self.vision.patch_embedding = nn.Linear(4, 8)
                    self.language: Any = nn.Module()
                    self.language.embedding = nn.Linear(4, 8)
                    self.projector = nn.Linear(16, 8)
                    self.history_encoder = nn.Linear(4, 8)
                    self.action_encoder = nn.Linear(21, 16)
                    self.action_head = nn.Linear(16, 6)
                    self.activation = nn.ReLU()

                def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
                    observation = batch["observation"]
                    language = batch["language"]
                    history = batch["history"]
                    noise = batch["noise"]
                    timestep = batch["timestep"]
                    vision = self.activation(self.vision.patch_embedding(observation))
                    language_hidden = self.activation(self.language.embedding(language))
                    projector = self.activation(self.projector(torch.cat([vision, language_hidden], dim=-1)))
                    history_hidden = self.activation(self.history_encoder(history))
                    action_input = torch.cat([projector, history_hidden, noise, timestep], dim=-1)
                    action_hidden = self.activation(self.action_encoder(action_input))
                    action = self.action_head(action_hidden)
                    return {
                        "vision": vision,
                        "language": language_hidden,
                        "projector": projector,
                        "history": history_hidden,
                        "action_hidden": action_hidden,
                        "action": action,
                    }

            model = FlowModel()
        self.model = model
        self._spec = ModelSpec(
            model_id="synthetic-flow-vla",
            family="synthetic_flow_action",
            framework="torch",
            action_dim=6,
            action_horizon=1,
        )

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    def named_modules(self) -> dict[str, Any]:
        return {
            name: module for name, module in self.model.named_modules() if name and hasattr(module, "weight") and hasattr(module, "bias")
        }

    @property
    def action_schema(self) -> ActionSchema:
        return ActionSchema(
            model_action_dim=6,
            output_action_dim=6,
            horizon=1,
            denoise_steps=1,
            translation_indices=[0, 1, 2],
            rotation_indices=[3, 4],
            gripper_index=5,
            gripper_threshold=0.5,
            postprocess="synthetic_identity",
        )

    def module_inventory(self) -> list[ModuleDescriptor]:
        components = {
            "vision.patch_embedding": "vision",
            "language.embedding": "language",
            "projector": "projector",
            "history_encoder": "history",
            "action_encoder": "action_backbone",
            "action_head": "action_head",
        }
        return [
            ModuleDescriptor(
                logical_id=name,
                backend_path=name,
                component=components[name],
                op_family="linear",
                parameter_count=sum(parameter.numel() for parameter in module.parameters(recurse=False)),
                quantizable=True,
            )
            for name, module in self.named_modules().items()
        ]

    def capture_specs(self) -> list[CaptureSpec]:
        components = {
            "vision": "vision",
            "language": "language",
            "projector": "projector",
            "history": "history",
            "action_hidden": "action_backbone",
            "action": "action_head",
        }
        return [
            CaptureSpec(logical_id=name, backend_path=name, component=component, kind="action" if name == "action" else "activation")
            for name, component in components.items()
        ]

    def fresh(self) -> TorchSyntheticFlowVLAAdapter:
        return TorchSyntheticFlowVLAAdapter(seed=self.seed)

    def backend_model(self) -> Any:
        return self.model

    def with_backend_model(self, model: Any) -> TorchSyntheticFlowVLAAdapter:
        return TorchSyntheticFlowVLAAdapter(seed=self.seed, model=model)

    def with_model(self, model: Any) -> TorchSyntheticFlowVLAAdapter:
        return self.with_backend_model(model)

    @staticmethod
    def _tensor_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        return {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in batch.items()
            if name != "stage" and name != "target_action"
        }

    def forward_backend(self, model: Any, batch: Mapping[str, Any]) -> None:
        import torch

        with torch.no_grad():
            model(self._tensor_batch(batch))

    def forward(self, batch: Mapping[str, Any], capture_points: Sequence[str]) -> dict[str, np.ndarray]:
        import torch

        with torch.no_grad():
            values = self.model(self._tensor_batch(batch))
        return {name: values[name].detach().cpu().numpy() for name in capture_points}
