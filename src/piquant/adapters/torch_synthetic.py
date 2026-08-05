"""Optional Torch adapter for the same synthetic graph used by ModelOpt."""

from __future__ import annotations

from typing import Any

import numpy as np

from piquant.contracts import ModelSpec


class TorchSyntheticFlowVLAAdapter:
    """A torch-only adapter imported explicitly by the ModelOpt integration."""

    def __init__(self, seed: int = 7, model: Any | None = None) -> None:
        try:
            import torch
            from torch import nn
        except ModuleNotFoundError as error:
            raise RuntimeError("TorchSyntheticFlowVLAAdapter requires torch") from error

        torch.manual_seed(seed)
        if model is None:

            class FlowModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.vision = nn.Module()
                    self.vision.patch_embedding = nn.Linear(4, 8)
                    self.language = nn.Module()
                    self.language.embedding = nn.Linear(4, 8)
                    self.projector = nn.Linear(16, 8)
                    self.history_encoder = nn.Linear(4, 8)
                    self.action_encoder = nn.Linear(21, 16)
                    self.action_head = nn.Linear(16, 6)

                def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
                    observation = batch["observation"]
                    language = batch["language"]
                    history = batch["history"]
                    noise = batch["noise"]
                    timestep = batch["timestep"]
                    vision = torch.relu(self.vision.patch_embedding(observation))
                    language_hidden = torch.relu(self.language.embedding(language))
                    projector = torch.relu(self.projector(torch.cat([vision, language_hidden], dim=-1)))
                    history_hidden = torch.relu(self.history_encoder(history))
                    action_input = torch.cat([projector, history_hidden, noise, timestep], dim=-1)
                    action_hidden = torch.relu(self.action_encoder(action_input))
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

    def backend_model(self) -> Any:
        return self.model

    def with_model(self, model: Any) -> TorchSyntheticFlowVLAAdapter:
        return TorchSyntheticFlowVLAAdapter(model=model)

    @staticmethod
    def _tensor_batch(batch: dict[str, Any]) -> dict[str, Any]:
        import torch

        return {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in batch.items()
            if name != "stage" and name != "target_action"
        }

    def forward_backend(self, model: Any, batch: dict[str, Any]) -> None:
        import torch

        with torch.no_grad():
            model(self._tensor_batch(batch))

    def forward(self, batch: dict[str, Any], capture_points: list[str] | tuple[str, ...]) -> dict[str, np.ndarray]:
        import torch

        with torch.no_grad():
            values = self.model(self._tensor_batch(batch))
        return {name: values[name].detach().cpu().numpy() for name in capture_points}
