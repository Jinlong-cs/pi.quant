"""Checkpoint-aligned OpenPI Pi0.5 semantic adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from piquant.contracts import ActionSchema, CaptureSpec, ModelSpec, ModuleDescriptor
from piquant.integrations.openpi import (
    PI05_CAMERA_KEYS,
    PI05_DENOISE_TIMESTEPS,
    Pi05LiberoData,
    Pi05OpenPIConfig,
    load_pi05_torch_model,
)

_VISION_BLOCK = re.compile(r"^paligemma_with_expert\.paligemma\.model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.(.+)$")
_VLM_BLOCK = re.compile(r"^paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\.(.+)$")
_ACTION_BLOCK = re.compile(r"^paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class _CaptureBinding:
    spec: CaptureSpec
    mode: Literal["single", "camera", "steps", "step", "selected", "derived"]
    step_index: int | None = None


def _linear_role(tail: str) -> tuple[str, list[str], bool]:
    roles = {
        "self_attn.q_proj": ("attention.q", ["attention", "q_projection"], True),
        "self_attn.k_proj": ("attention.k", ["attention", "k_projection", "cache_projection"], True),
        "self_attn.v_proj": ("attention.v", ["attention", "v_projection", "cache_projection"], True),
        "self_attn.o_proj": ("attention.o", ["attention", "output_projection"], True),
        "self_attn.out_proj": ("attention.out", ["attention", "output_projection"], True),
        "mlp.gate_proj": ("mlp.gate", ["mlp", "gate_projection"], True),
        "mlp.up_proj": ("mlp.up", ["mlp", "up_projection"], True),
        "mlp.down_proj": ("mlp.down", ["mlp", "down_projection"], True),
        "mlp.fc1": ("mlp.fc1", ["mlp", "up_projection"], True),
        "mlp.fc2": ("mlp.fc2", ["mlp", "down_projection"], True),
        "input_layernorm.dense": ("norm.input_condition", ["adaptive_norm", "precision_guard"], False),
        "post_attention_layernorm.dense": ("norm.post_attention_condition", ["adaptive_norm", "precision_guard"], False),
    }
    if tail not in roles:
        raise ValueError(f"unrecognized Pi0.5 linear role: {tail}")
    return roles[tail]


class Pi05TorchAdapter:
    """Expose stable semantic IDs while preserving exact OpenPI backend paths."""

    def __init__(self, config: Pi05OpenPIConfig, data: Pi05LiberoData, model: Any | None = None) -> None:
        self.config = config
        self.data = data
        self.model = load_pi05_torch_model(config) if model is None else model
        self._module_inventory = self._build_module_inventory()
        self._capture_bindings = self._build_capture_bindings()

    @property
    def spec(self) -> ModelSpec:
        return self.config.model_spec

    @property
    def action_schema(self) -> ActionSchema:
        return ActionSchema(
            model_action_dim=self.config.action_dim,
            output_action_dim=self.config.output_action_dim,
            horizon=self.config.action_horizon,
            denoise_steps=self.config.denoise_steps,
            translation_indices=[0, 1, 2],
            rotation_indices=[3, 4, 5],
            gripper_index=6,
            gripper_threshold=0.0,
            postprocess=f"openpi-quantile-unnormalize:{self.config.norm_stats_sha256};libero-first7",
        )

    def named_modules(self) -> dict[str, Any]:
        import torch

        return {name: module for name, module in self.model.named_modules() if name and isinstance(module, torch.nn.Linear)}

    def module_inventory(self) -> Sequence[ModuleDescriptor]:
        return self._module_inventory

    def capture_specs(self) -> Sequence[CaptureSpec]:
        return [binding.spec for binding in self._capture_bindings.values()]

    def fresh(self) -> Pi05TorchAdapter:
        return Pi05TorchAdapter(self.config, self.data)

    def backend_model(self) -> Any:
        return self.model

    def with_backend_model(self, model: Any) -> Pi05TorchAdapter:
        return Pi05TorchAdapter(self.config, self.data, model=model)

    def _descriptor(self, name: str, module: Any) -> ModuleDescriptor:
        parameter_count = sum(parameter.numel() for parameter in module.parameters(recurse=False))
        match = _VISION_BLOCK.match(name)
        if match:
            block_index = int(match.group(1))
            role, tags, quantizable = _linear_role(match.group(2))
            return ModuleDescriptor(
                logical_id=f"vision.block.{block_index:02d}.{role}",
                backend_path=name,
                component="vision",
                block_index=block_index,
                op_family="linear",
                parameter_count=parameter_count,
                quantizable=quantizable,
                tags=tags,
            )
        match = _VLM_BLOCK.match(name)
        if match:
            block_index = int(match.group(1))
            role, tags, quantizable = _linear_role(match.group(2))
            component = "vlm_attention" if "attention" in tags else "vlm_mlp"
            return ModuleDescriptor(
                logical_id=f"vlm.block.{block_index:02d}.{role}",
                backend_path=name,
                component=component,
                block_index=block_index,
                op_family="linear",
                parameter_count=parameter_count,
                quantizable=quantizable,
                tags=tags,
            )
        match = _ACTION_BLOCK.match(name)
        if match:
            block_index = int(match.group(1))
            role, tags, quantizable = _linear_role(match.group(2))
            return ModuleDescriptor(
                logical_id=f"action.block.{block_index:02d}.{role}",
                backend_path=name,
                component="action_norm" if "adaptive_norm" in tags else "action_backbone",
                block_index=block_index,
                op_family="linear",
                parameter_count=parameter_count,
                quantizable=quantizable,
                tags=tags,
            )
        fixed = {
            "paligemma_with_expert.paligemma.model.multi_modal_projector.linear": (
                "projector.linear",
                "projector",
                True,
                ["cross_modal_projector"],
            ),
            "action_in_proj": ("action.input", "action_input", True, ["action_boundary"]),
            "action_out_proj": ("action.head", "action_head", True, ["action_boundary", "precision_guard"]),
            "time_mlp_in": ("time_embedding.in", "time_embedding", True, ["time_embedding"]),
            "time_mlp_out": ("time_embedding.out", "time_embedding", True, ["time_embedding"]),
            "paligemma_with_expert.gemma_expert.model.norm.dense": (
                "action.norm.final_condition",
                "action_norm",
                False,
                ["adaptive_norm", "precision_guard"],
            ),
            "paligemma_with_expert.paligemma.lm_head": ("vlm.lm_head", "vlm_head", False, ["unused_policy_path"]),
            "paligemma_with_expert.gemma_expert.lm_head": (
                "action.lm_head",
                "action_head",
                False,
                ["unused_policy_path"],
            ),
        }
        if name not in fixed:
            raise ValueError(f"unclassified Pi0.5 Linear module: {name}")
        logical_id, component, quantizable, tags = fixed[name]
        return ModuleDescriptor(
            logical_id=logical_id,
            backend_path=name,
            component=component,
            op_family="linear",
            parameter_count=parameter_count,
            quantizable=quantizable,
            tags=tags,
        )

    def _build_module_inventory(self) -> list[ModuleDescriptor]:
        inventory = [self._descriptor(name, module) for name, module in self.named_modules().items()]
        logical_ids = [descriptor.logical_id for descriptor in inventory]
        if len(logical_ids) != len(set(logical_ids)):
            raise ValueError("Pi0.5 semantic inventory contains duplicate logical IDs")
        return inventory

    def _block_indices(self, prefix: str) -> list[int]:
        indices = sorted(
            {
                descriptor.block_index
                for descriptor in self._module_inventory
                if descriptor.logical_id.startswith(prefix) and descriptor.block_index is not None
            }
        )
        if not indices:
            raise ValueError(f"Pi0.5 inventory contains no blocks for {prefix}")
        return [indices[0], indices[len(indices) // 2], indices[-1]]

    @staticmethod
    def _capture_binding(
        logical_id: str,
        backend_path: str,
        component: str,
        kind: Literal["activation", "cache", "flow", "action", "gripper"],
        mode: Literal["single", "camera", "steps", "step", "selected", "derived"],
        *,
        block_index: int | None = None,
        step_index: int | None = None,
    ) -> _CaptureBinding:
        return _CaptureBinding(
            spec=CaptureSpec(
                logical_id=logical_id,
                backend_path=backend_path,
                component=component,
                kind=kind,
                block_index=block_index,
            ),
            mode=mode,
            step_index=step_index,
        )

    def _build_capture_bindings(self) -> dict[str, _CaptureBinding]:
        bindings = [
            self._capture_binding(
                "vision.patch_embedding",
                "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding",
                "vision",
                "activation",
                "camera",
            ),
            self._capture_binding(
                "projector.output",
                "paligemma_with_expert.paligemma.model.multi_modal_projector",
                "projector",
                "activation",
                "camera",
            ),
            self._capture_binding(
                "vlm.embedding",
                "paligemma_with_expert.paligemma.model.language_model.embed_tokens",
                "vlm",
                "activation",
                "single",
            ),
            self._capture_binding("action.input", "action_in_proj", "action_input", "activation", "steps"),
            self._capture_binding("time_embedding.output", "time_mlp_out", "time_embedding", "activation", "steps"),
            self._capture_binding("flow.selected", "action_out_proj", "action_head", "flow", "selected"),
            self._capture_binding("action.normalized", "$sample_actions", "action_head", "action", "derived"),
            self._capture_binding("action.output", "$postprocess", "action_head", "action", "derived"),
            self._capture_binding("gripper.normalized", "$sample_actions[...,6]", "action_head", "gripper", "derived"),
            self._capture_binding("gripper.output", "$postprocess[...,6]", "action_head", "gripper", "derived"),
        ]
        for index in self._block_indices("vision.block."):
            bindings.append(
                self._capture_binding(
                    f"vision.block.{index:02d}.output",
                    f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{index}",
                    "vision",
                    "activation",
                    "camera",
                    block_index=index,
                )
            )
        vlm_indices = self._block_indices("vlm.block.")
        for index in vlm_indices:
            block_path = f"paligemma_with_expert.paligemma.model.language_model.layers.{index}"
            bindings.append(
                self._capture_binding(f"vlm.block.{index:02d}.output", block_path, "vlm", "activation", "single", block_index=index)
            )
        for index in (vlm_indices[0], vlm_indices[-1]):
            attention_path = f"paligemma_with_expert.paligemma.model.language_model.layers.{index}.self_attn"
            bindings.extend(
                [
                    self._capture_binding(
                        f"prefix.block.{index:02d}.k_projection",
                        f"{attention_path}.k_proj",
                        "prefix_cache",
                        "cache",
                        "single",
                        block_index=index,
                    ),
                    self._capture_binding(
                        f"prefix.block.{index:02d}.v_projection",
                        f"{attention_path}.v_proj",
                        "prefix_cache",
                        "cache",
                        "single",
                        block_index=index,
                    ),
                ]
            )
        for index in self._block_indices("action.block."):
            bindings.append(
                self._capture_binding(
                    f"action.block.{index:02d}.output",
                    f"paligemma_with_expert.gemma_expert.model.layers.{index}",
                    "action_backbone",
                    "activation",
                    "steps",
                    block_index=index,
                )
            )
        for index in range(self.config.denoise_steps):
            bindings.append(
                self._capture_binding(
                    f"flow.step.{index:02d}",
                    "action_out_proj",
                    "action_head",
                    "flow",
                    "step",
                    block_index=index,
                    step_index=index,
                )
            )
        result = {binding.spec.logical_id: binding for binding in bindings}
        if len(result) != len(bindings):
            raise ValueError("Pi0.5 capture inventory contains duplicate logical IDs")
        model_paths = dict(self.model.named_modules())
        missing = sorted({binding.spec.backend_path for binding in bindings if binding.mode != "derived"} - set(model_paths))
        if missing:
            raise ValueError(f"Pi0.5 capture inventory resolves to missing backend paths: {missing!r}")
        return result

    @staticmethod
    def _tensor_output(torch: Any, output: Any) -> Any:
        if torch.is_tensor(output):
            return output
        if isinstance(output, tuple | list) and output and torch.is_tensor(output[0]):
            return output[0]
        if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
            return output.last_hidden_state
        raise TypeError(f"capture hook received unsupported output type {type(output).__name__}")

    def _observation(self, batch: Mapping[str, Any]) -> Any:
        import torch
        from openpi.models.model import Observation

        device = self.config.device
        return Observation(
            images={key: torch.as_tensor(batch["images"][key], dtype=torch.float32, device=device) for key in PI05_CAMERA_KEYS},
            image_masks={key: torch.as_tensor(batch["image_masks"][key], dtype=torch.bool, device=device) for key in PI05_CAMERA_KEYS},
            state=torch.as_tensor(batch["state"], dtype=torch.float32, device=device),
            tokenized_prompt=torch.as_tensor(batch["tokenized_prompt"], dtype=torch.long, device=device),
            tokenized_prompt_mask=torch.as_tensor(batch["tokenized_prompt_mask"], dtype=torch.bool, device=device),
        )

    def _sample(self, model: Any, batch: Mapping[str, Any]) -> Any:
        import torch

        observation = self._observation(batch)
        noise = torch.as_tensor(batch["noise"], dtype=torch.float32, device=self.config.device)
        with torch.inference_mode():
            return model.sample_actions(self.config.device, observation, noise=noise, num_steps=self.config.denoise_steps)

    def forward_backend(self, model: Any, batch: Mapping[str, Any]) -> None:
        self._sample(model, batch)

    @staticmethod
    def _stack_calls(values: list[np.ndarray], *, expected: int, logical_id: str) -> np.ndarray:
        if len(values) != expected:
            raise ValueError(f"capture {logical_id!r} observed {len(values)} calls, expected {expected}")
        return np.stack(values, axis=1)

    def forward(self, batch: Mapping[str, Any], capture_points: Sequence[str]) -> dict[str, np.ndarray]:
        import torch

        unknown = sorted(set(capture_points) - set(self._capture_bindings))
        if unknown:
            raise ValueError(f"unknown Pi0.5 capture logical IDs: {unknown!r}")
        requested = [self._capture_bindings[logical_id] for logical_id in capture_points]
        hook_paths = sorted({binding.spec.backend_path for binding in requested if binding.mode != "derived"})
        modules = dict(self.model.named_modules())
        captured: dict[str, list[np.ndarray]] = {path: [] for path in hook_paths}
        handles = []
        for path in hook_paths:

            def hook(_module: Any, _inputs: Any, output: Any, *, capture_path: str = path) -> None:
                tensor = self._tensor_output(torch, output)
                captured[capture_path].append(tensor.detach().to(dtype=torch.float32, device="cpu").numpy())

            handles.append(modules[path].register_forward_hook(hook))
        try:
            normalized_tensor = self._sample(self.model, batch)
        finally:
            for handle in handles:
                handle.remove()
        normalized = normalized_tensor.detach().to(dtype=torch.float32, device="cpu").numpy()
        output = self.data.postprocess_actions(normalized, np.asarray(batch["state"]))
        flow_values = captured.get("action_out_proj", [])
        flows = None if not flow_values else self._stack_calls(flow_values, expected=self.config.denoise_steps, logical_id="flow")
        results: dict[str, np.ndarray] = {}
        for binding in requested:
            logical_id = binding.spec.logical_id
            if binding.mode == "derived":
                derived = {
                    "action.normalized": normalized,
                    "action.output": output,
                    "gripper.normalized": normalized[..., 6:7],
                    "gripper.output": output[..., 6:7],
                }
                results[logical_id] = derived[logical_id]
            elif binding.mode == "single":
                values = captured[binding.spec.backend_path]
                if len(values) != 1:
                    raise ValueError(f"capture {logical_id!r} observed {len(values)} calls, expected one")
                results[logical_id] = values[0]
            elif binding.mode == "camera":
                results[logical_id] = self._stack_calls(captured[binding.spec.backend_path], expected=3, logical_id=logical_id)
            elif binding.mode == "steps":
                results[logical_id] = self._stack_calls(
                    captured[binding.spec.backend_path], expected=self.config.denoise_steps, logical_id=logical_id
                )
            elif binding.mode == "step":
                if flows is None or binding.step_index is None:
                    raise ValueError(f"flow capture {logical_id!r} was not produced")
                results[logical_id] = flows[:, binding.step_index]
            elif binding.mode == "selected":
                if flows is None:
                    raise ValueError("selected flow capture was not produced")
                timesteps = np.asarray(batch["timestep"], dtype=np.float32)
                schedule = np.asarray(PI05_DENOISE_TIMESTEPS, dtype=np.float32)
                step_indices = np.argmin(np.abs(timesteps[:, None] - schedule[None, :]), axis=1)
                results[logical_id] = np.stack([flows[index, step_index] for index, step_index in enumerate(step_indices)])
        return results
