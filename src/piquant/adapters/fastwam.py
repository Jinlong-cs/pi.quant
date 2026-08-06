"""Explicit semantic adapter for the audited FastWAM source model."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from piquant.contracts import (
    ActionSchema,
    ModelSpec,
    ModuleDescriptor,
    TemporalCaptureSpec,
    TemporalMode,
)
from piquant.integrations.fastwam import FastWAMCaptureRunner, FastWAMInferenceContract

_BLOCK = re.compile(r"^(video_expert|action_expert)\.blocks\.(\d+)(?:\.(.*))?$")


@dataclass(frozen=True)
class FastWAMSourceConfig:
    """Identity and execution injection for one external FastWAM environment."""

    model_id: str = "fastwam"
    revision: str = "external"
    model_factory: Callable[[], Any] | None = field(repr=False, default=None)
    inference: FastWAMInferenceContract = field(default_factory=FastWAMInferenceContract)
    output_action_dim: int = 7
    gripper_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.model_factory is None or not callable(self.model_factory):
            raise ValueError("FastWAMSourceConfig requires an injected model_factory")


class FastWAMSourceAdapter:
    """Expose semantic modules and temporal captures without importing FastWAM at load time."""

    def __init__(
        self,
        config: FastWAMSourceConfig,
        *,
        model: Any | None = None,
        runner: FastWAMCaptureRunner | None = None,
    ) -> None:
        self.config = config
        factory = config.model_factory
        if factory is None:
            raise ValueError("FastWAMSourceConfig requires an injected model_factory")
        self.model = factory() if model is None else model
        self.runner = runner or FastWAMCaptureRunner(contract=config.inference)
        self._all_modules = self._resolve_modules()
        self._inventory = self._build_inventory()
        self._capture_specs = self._build_capture_specs()

    def _resolve_modules(self) -> dict[str, Any]:
        named_modules = getattr(self.model, "named_modules", None)
        if not callable(named_modules):
            raise TypeError("FastWAM model must expose named_modules()")
        return {name: module for name, module in named_modules() if name}

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            model_id=self.config.model_id,
            family="FastWAM",
            framework="torch",
            revision=self.config.revision,
            task="wam",
            action_dim=self.config.inference.action_dim,
            action_horizon=self.config.inference.action_horizon,
        )

    @property
    def action_schema(self) -> ActionSchema:
        return ActionSchema(
            model_action_dim=self.config.inference.action_dim,
            output_action_dim=self.config.output_action_dim,
            horizon=self.config.inference.action_horizon,
            denoise_steps=self.config.inference.num_inference_steps,
            translation_indices=[0, 1, 2],
            rotation_indices=[3, 4, 5],
            gripper_index=6,
            gripper_threshold=self.config.gripper_threshold,
            postprocess="external-fastwam-normalizer-and-action-postprocess",
        )

    def named_modules(self) -> Mapping[str, Any]:
        torch = self._torch()
        return {name: module for name, module in self._all_modules.items() if isinstance(module, torch.nn.Linear)}

    def module_inventory(self) -> Sequence[ModuleDescriptor]:
        return self._inventory

    def temporal_capture_specs(self) -> Sequence[TemporalCaptureSpec]:
        return self._capture_specs

    def capture_specs(self) -> Sequence[Any]:
        return list(self._capture_specs)

    def fresh(self) -> FastWAMSourceAdapter:
        return FastWAMSourceAdapter(self.config, runner=self.runner)

    def backend_model(self) -> Any:
        return self.model

    def with_backend_model(self, model: Any) -> FastWAMSourceAdapter:
        return FastWAMSourceAdapter(self.config, model=model, runner=self.runner)

    def forward(self, batch: Mapping[str, Any], capture_points: Sequence[str]) -> Mapping[str, Any]:
        return self.forward_temporal(batch, capture_points, mode="iterative")

    def forward_backend(self, model: Any, batch: Mapping[str, Any]) -> None:
        mode = cast(TemporalMode, batch.get("__piquant_temporal_mode__", "iterative"))
        self.runner.run(model, batch, (), mode=mode)

    def forward_temporal(
        self,
        batch: Mapping[str, Any],
        capture_points: Sequence[str],
        *,
        mode: TemporalMode,
    ) -> Mapping[str, Any]:
        specs = {spec.logical_id: spec for spec in self._capture_specs}
        unknown = sorted(set(capture_points) - set(specs))
        if unknown:
            raise ValueError(f"FastWAM capture points are not declared by the adapter: {unknown!r}")
        selected = [specs[name] for name in capture_points]
        return self.runner.run(self.model, batch, selected, mode=mode)

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError("FastWAM adapter requires the external PyTorch/FastWAM environment") from error
        return torch

    @staticmethod
    def _component(path: str) -> tuple[str, list[str], int | None]:
        match = _BLOCK.match(path)
        if match:
            expert, index_text, tail = match.groups()
            component = "video_backbone" if expert == "video_expert" else "action_backbone"
            return component, ["backbone", expert], int(index_text)
        if path.startswith("video_expert.time_") or path.startswith("video_expert.head"):
            return "video_boundary", ["time_or_output_boundary", "precision_hypothesis"], None
        if path.startswith("action_expert.time_") or path in {"action_expert.action_encoder", "action_expert.head"}:
            return "action_boundary", ["action_boundary", "precision_hypothesis"], None
        if path.startswith("video_expert.text_embedding") or path.startswith("action_expert.text_embedding"):
            return "context_bridge", ["context", "projector"], None
        if path == "proprio_encoder":
            return "proprio_bridge", ["context", "proprio"], None
        if path.startswith("text_encoder"):
            return "context_language", ["context"], None
        if path.startswith("vae"):
            return "video_vae", ["world", "vae"], None
        return "unclassified", ["unclassified"], None

    def _build_inventory(self) -> list[ModuleDescriptor]:
        descriptors: list[ModuleDescriptor] = []
        for path, module in self.named_modules().items():
            component, tags, block_index = self._component(path)
            parameter_count = sum(int(parameter.numel()) for parameter in module.parameters(recurse=False))
            descriptors.append(
                ModuleDescriptor(
                    logical_id=self._logical_id(path, component, block_index),
                    backend_path=path,
                    component=component,
                    block_index=block_index,
                    op_family="linear",
                    parameter_count=parameter_count,
                    quantizable=component != "unclassified",
                    tags=tags,
                )
            )
        logical_ids = [descriptor.logical_id for descriptor in descriptors]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("FastWAM semantic inventory contains duplicate logical IDs")
        return descriptors

    @staticmethod
    def _logical_id(path: str, component: str, block_index: int | None) -> str:
        match = _BLOCK.match(path)
        if match and block_index is not None:
            tail = match.group(3) or "output"
            return f"{component}.block.{block_index:02d}.{tail}"
        if path == "proprio_encoder":
            return "proprio_bridge.output"
        if path.startswith("text_encoder"):
            return f"context_language.{path.removeprefix('text_encoder.')}"
        if path.startswith("vae"):
            return f"video_vae.{path.removeprefix('vae.')}"
        if path.startswith("video_expert."):
            return f"{component}.{path.removeprefix('video_expert.')}"
        if path.startswith("action_expert."):
            return f"{component}.{path.removeprefix('action_expert.')}"
        return f"{component}.{path}"

    def _block_indices(self, expert: str) -> list[int]:
        indices = sorted({int(match.group(2)) for name in self._all_modules if (match := _BLOCK.match(name)) and match.group(1) == expert})
        if not indices:
            raise ValueError(f"FastWAM model exposes no {expert}.blocks modules")
        return [indices[0], indices[len(indices) // 2], indices[-1]]

    def _build_capture_specs(self) -> list[TemporalCaptureSpec]:
        specs: list[TemporalCaptureSpec] = []
        for expert, component, axes in (
            ("video_expert", "video_backbone", ["batch", "execution", "token", "hidden"]),
            ("action_expert", "action_backbone", ["batch", "denoise_step", "token", "hidden"]),
        ):
            for index in self._block_indices(expert):
                path = f"{expert}.blocks.{index}"
                if path in self._all_modules:
                    specs.append(
                        TemporalCaptureSpec(
                            logical_id=f"{component}.block.{index:02d}.output",
                            backend_path=path,
                            component=component,
                            kind="activation",
                            axes=axes,
                            block_index=index,
                        )
                    )
        for logical_id, path, component, kind, axes in (
            (
                "context.video_text_embedding",
                "video_expert.text_embedding",
                "context_bridge",
                "activation",
                ["batch", "token", "hidden"],
            ),
            (
                "action.boundary.input",
                "action_expert.action_encoder",
                "action_boundary",
                "activation",
                ["batch", "denoise_step", "token", "hidden"],
            ),
            (
                "action.boundary.flow",
                "action_expert.head",
                "action_boundary",
                "flow",
                ["batch", "denoise_step", "horizon", "action_dim"],
            ),
            ("action.output", "$infer_action", "action_boundary", "action", ["batch", "horizon", "action_dim"]),
        ):
            if path.startswith("$") or path in self._all_modules:
                specs.append(
                    TemporalCaptureSpec(
                        logical_id=logical_id,
                        backend_path=path,
                        component=component,
                        kind=cast(Any, kind),
                        axes=axes,
                    )
                )
        if "proprio_encoder" in self._all_modules:
            specs.append(
                TemporalCaptureSpec(
                    logical_id="context.proprio_bridge",
                    backend_path="proprio_encoder",
                    component="proprio_bridge",
                    kind="activation",
                    axes=["batch", "hidden"],
                )
            )
        logical_ids = [spec.logical_id for spec in specs]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("FastWAM temporal capture IDs are not unique")
        return specs
