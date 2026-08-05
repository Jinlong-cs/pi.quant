"""Lazy OpenPI Pi0.5 model and LIBERO data integration."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from piquant.calibration import require_episode_disjoint, require_task_coverage
from piquant.contracts import CalibrationManifest, ModelSpec, SampleRef

PI05_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
PI05_DENOISE_TIMESTEPS = tuple(round(1.0 - index / 10.0, 1) for index in range(10))
PI05_TIMESTEP_QUANTILES = tuple(PI05_DENOISE_TIMESTEPS[index] for index in (0, 3, 6, 9))
PI05_STAGE_RULE = (
    "gripper-transition-v1:early=10pct;pre=first-transition-index;post=first-transition+1;"
    "late=episode_length-action_horizon;fallback=45pct/55pct"
)
ManifestSplit = Literal["calibration", "diagnostic_holdout", "random_control", "promotion_reserved"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _derived_seed(seed: int, *parts: object) -> int:
    encoded = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


@dataclass(frozen=True)
class Pi05OpenPIConfig:
    openpi_revision: str
    checkpoint_dir: Path
    checkpoint_sha256: str
    norm_stats_dir: Path
    norm_stats_sha256: str
    openpi_data_home: Path
    tokenizer_sha256: str
    device: str = "cuda:0"
    seed: int = 20260805
    token_length: int = 200
    action_horizon: int = 50
    action_dim: int = 32
    output_action_dim: int = 7
    denoise_steps: int = 10
    discrete_state_input: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_dir / "model.safetensors"

    @property
    def norm_stats_path(self) -> Path:
        return self.norm_stats_dir / "norm_stats.json"

    @property
    def tokenizer_path(self) -> Path:
        return self.openpi_data_home / "big_vision" / "paligemma_tokenizer.model"

    @property
    def model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="pi05-libero",
            family="pi0.5",
            framework="openpi-pytorch",
            revision=f"openpi:{self.openpi_revision};checkpoint-sha256:{self.checkpoint_sha256}",
            task="flow_action",
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )

    def validate_identity(self, *, hash_assets: bool = True) -> None:
        if self.token_length != 200 or self.action_horizon != 50 or self.action_dim != 32 or self.output_action_dim != 7:
            raise ValueError("Pi0.5 LIBERO integration requires token200, horizon50, model action dim32 and output action dim7")
        if self.denoise_steps != 10 or self.discrete_state_input:
            raise ValueError("audited Pi0.5 LIBERO integration requires D10 and discrete_state_input=False")
        for path in (self.checkpoint_path, self.norm_stats_path, self.tokenizer_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        configured_home = Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
        if configured_home != self.openpi_data_home.expanduser().resolve():
            raise ValueError(f"OPENPI_DATA_HOME resolves to {configured_home}, expected {self.openpi_data_home.resolve()}")
        if hash_assets and sha256_file(self.checkpoint_path) != self.checkpoint_sha256:
            raise ValueError("Pi0.5 checkpoint SHA256 differs from the frozen identity")
        if hash_assets and sha256_file(self.norm_stats_path) != self.norm_stats_sha256:
            raise ValueError("Pi0.5 normalization SHA256 differs from the frozen identity")
        if hash_assets and sha256_file(self.tokenizer_path) != self.tokenizer_sha256:
            raise ValueError("PaliGemma tokenizer SHA256 differs from the frozen identity")


def load_pi05_torch_model(config: Pi05OpenPIConfig) -> Any:
    """Load one fresh checkpoint-aligned PyTorch model through public OpenPI APIs."""

    config.validate_identity(hash_assets=False)
    import safetensors.torch
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    torch.manual_seed(config.seed)
    model_config = Pi0Config(
        pi05=True,
        action_horizon=config.action_horizon,
        max_token_len=config.token_length,
        discrete_state_input=config.discrete_state_input,
        dtype="bfloat16",
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(model_config).to(config.device)
    missing, unexpected = safetensors.torch.load_model(model, config.checkpoint_path, strict=True, device=config.device)
    if missing or unexpected:
        raise ValueError(f"checkpoint load mismatch: missing={missing!r}, unexpected={unexpected!r}")
    model.eval()
    return model


def _episode_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / f"data/chunk-{episode_index // 1000:03d}/episode_{episode_index:06d}.parquet"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dataset_revision(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl"):
        path = dataset_root / "meta" / name
        digest.update(name.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _action_window(actions: np.ndarray, frame_index: int, horizon: int) -> np.ndarray:
    window = np.asarray(actions[frame_index : frame_index + horizon], dtype=np.float32)
    if window.shape[0] == 0:
        raise ValueError(f"empty action window at frame {frame_index}")
    if window.shape[0] < horizon:
        window = np.concatenate([window, np.repeat(window[-1:], horizon - window.shape[0], axis=0)])
    return window


def _stage_frames(actions: np.ndarray, horizon: int) -> dict[str, int]:
    length = actions.shape[0]
    if length == 0:
        raise ValueError("cannot derive stages from an empty episode")
    transitions = np.flatnonzero(np.abs(np.diff(actions[:, 6])) > 0.1)
    if transitions.size:
        pre_transition = int(transitions[0])
        post_transition = min(pre_transition + 1, length - 1)
    else:
        pre_transition = int(round((length - 1) * 0.45))
        post_transition = int(round((length - 1) * 0.55))
    return {
        "early": int(round((length - 1) * 0.10)),
        "pre_gripper_transition": pre_transition,
        "post_gripper_transition": post_transition,
        "late": max(0, length - horizon),
    }


def _sample_ref(
    *,
    split: ManifestSplit,
    suite: str,
    task: str,
    task_index: int,
    episode_index: int,
    frame_index: int,
    episode_length: int,
    stage: str,
    timestep: float,
    actions: np.ndarray,
    source_path: Path,
    source_sha256: str,
    normalization_revision: str,
    seed: int,
    horizon: int,
) -> SampleRef:
    action_target = _action_window(actions, frame_index, horizon)
    sample_seed = _derived_seed(seed, split, task_index, episode_index, frame_index, stage)
    return SampleRef(
        sample_id=f"{split}-task{task_index:02d}-ep{episode_index:06d}-frame{frame_index:04d}-{stage}",
        split=split,
        suite=suite,
        task=task,
        task_index=task_index,
        episode_index=episode_index,
        frame_index=frame_index,
        window_start=frame_index,
        window_end=min(episode_length, frame_index + horizon),
        camera_keys=list(PI05_CAMERA_KEYS),
        instruction_sha256=hashlib.sha256(task.encode()).hexdigest(),
        history_window=[],
        action_target_sha256=sha256_array(action_target),
        normalization_revision=normalization_revision,
        stage=stage,
        source_path=str(source_path),
        source_sha256=source_sha256,
        seed=sample_seed,
        flow_noise_seed=_derived_seed(sample_seed, "flow-noise"),
        timestep=timestep,
    )


def build_pi05_libero_manifests(
    dataset_root: str | Path,
    *,
    normalization_revision: str,
    openpi_revision: str,
    seed: int = 20260805,
    expected_tasks: int = 40,
    horizon: int = 50,
) -> dict[ManifestSplit, CalibrationManifest]:
    """Build stage-balanced calibration/holdout and a same-size random-frame control."""

    import pyarrow.parquet as pq

    root = Path(dataset_root).resolve()
    tasks = {str(item["task"]): int(item["task_index"]) for item in _load_jsonl(root / "meta/tasks.jsonl")}
    episodes_by_task: dict[str, list[int]] = defaultdict(list)
    for item in _load_jsonl(root / "meta/episodes.jsonl"):
        episode_tasks = item["tasks"]
        if len(episode_tasks) != 1:
            raise ValueError(f"expected one task per LIBERO episode, found {episode_tasks!r}")
        episodes_by_task[str(episode_tasks[0])].append(int(item["episode_index"]))
    if len(tasks) != expected_tasks:
        raise ValueError(f"LIBERO task metadata contains {len(tasks)} tasks, expected {expected_tasks}")

    normalization_id = f"sha256:{normalization_revision}"
    preprocess_revision = f"openpi:{openpi_revision}:libero-pi05-3slot-token200-h50-d10-v1"
    dataset_revision = _dataset_revision(root)
    samples: dict[ManifestSplit, list[SampleRef]] = {"calibration": [], "diagnostic_holdout": [], "random_control": []}
    source_hashes: dict[Path, str] = {}
    for task, task_index in sorted(tasks.items(), key=lambda item: item[1]):
        episode_indices = sorted(episodes_by_task[task])
        if len(episode_indices) < 2:
            raise ValueError(f"task {task_index} has fewer than two episodes")
        balanced_splits: tuple[tuple[ManifestSplit, int], ...] = (
            ("calibration", episode_indices[0]),
            ("diagnostic_holdout", episode_indices[1]),
        )
        for split, episode_index in balanced_splits:
            path = _episode_path(root, episode_index)
            table = pq.read_table(path, columns=["actions"])
            actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
            if path not in source_hashes:
                source_hashes[path] = sha256_file(path)
            source_hash = source_hashes[path]
            for (stage, frame_index), timestep in zip(_stage_frames(actions, horizon).items(), PI05_TIMESTEP_QUANTILES, strict=True):
                samples[split].append(
                    _sample_ref(
                        split=split,
                        suite="libero-40",
                        task=task,
                        task_index=task_index,
                        episode_index=episode_index,
                        frame_index=frame_index,
                        episode_length=len(actions),
                        stage=stage,
                        timestep=timestep,
                        actions=actions,
                        source_path=path,
                        source_sha256=source_hash,
                        normalization_revision=normalization_id,
                        seed=seed,
                        horizon=horizon,
                    )
                )

        control_episode = episode_indices[0]
        control_path = _episode_path(root, control_episode)
        control_table = pq.read_table(control_path, columns=["actions"])
        control_actions = np.asarray(control_table["actions"].to_pylist(), dtype=np.float32)
        if control_path not in source_hashes:
            source_hashes[control_path] = sha256_file(control_path)
        control_hash = source_hashes[control_path]
        rng = np.random.default_rng(_derived_seed(seed, "random-control", task_index, control_episode))
        random_frames = sorted(int(index) for index in rng.choice(len(control_actions), size=4, replace=False))
        for index, (frame_index, timestep) in enumerate(zip(random_frames, PI05_TIMESTEP_QUANTILES, strict=True)):
            samples["random_control"].append(
                _sample_ref(
                    split="random_control",
                    suite="libero-40",
                    task=task,
                    task_index=task_index,
                    episode_index=control_episode,
                    frame_index=frame_index,
                    episode_length=len(control_actions),
                    stage=f"random_{index}",
                    timestep=timestep,
                    actions=control_actions,
                    source_path=control_path,
                    source_sha256=control_hash,
                    normalization_revision=normalization_id,
                    seed=seed,
                    horizon=horizon,
                )
            )

    manifests = {
        split: CalibrationManifest(
            manifest_id=f"pi05-libero-{split}-seed{seed}",
            dataset_id="physical-intelligence/libero",
            dataset_revision=dataset_revision,
            split=split,
            preprocess_revision=preprocess_revision,
            normalization_revision=normalization_id,
            stage_rule=PI05_STAGE_RULE if split != "random_control" else "uniform-random-frame-v1",
            samples=split_samples,
        )
        for split, split_samples in samples.items()
    }
    require_task_coverage(manifests["calibration"], expected_tasks)
    require_task_coverage(manifests["diagnostic_holdout"], expected_tasks)
    require_task_coverage(manifests["random_control"], expected_tasks)
    require_episode_disjoint(manifests["calibration"], manifests["diagnostic_holdout"])
    require_episode_disjoint(manifests["random_control"], manifests["diagnostic_holdout"])
    return manifests


class Pi05LiberoData:
    """Load exact manifest samples through OpenPI's inference transforms."""

    def __init__(self, config: Pi05OpenPIConfig, dataset_root: str | Path) -> None:
        self.config = config
        self.dataset_root = Path(dataset_root).resolve()
        self._input_transform: Any | None = None
        self._norm_stats: Any | None = None
        self._verified_sources: set[Path] = set()

    def _transforms(self) -> tuple[Any, Any]:
        if self._input_transform is None:
            import openpi.shared.normalize as normalize
            from openpi import transforms
            from openpi.models import model as openpi_model
            from openpi.models.tokenizer import PaligemmaTokenizer
            from openpi.policies.libero_policy import LiberoInputs

            self._norm_stats = normalize.load(self.config.norm_stats_dir)
            self._input_transform = transforms.compose(
                [
                    LiberoInputs(model_type=openpi_model.ModelType.PI05),
                    transforms.Normalize(self._norm_stats, use_quantiles=True),
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(PaligemmaTokenizer(self.config.token_length), discrete_state_input=False),
                    transforms.PadStatesAndActions(self.config.action_dim),
                ]
            )
        return self._input_transform, self._norm_stats

    @staticmethod
    def _decode_image(value: dict[str, Any]) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(value["bytes"])).convert("RGB"))

    def load_batch(self, refs: Sequence[SampleRef]) -> dict[str, Any]:
        import pyarrow.parquet as pq

        if not refs:
            raise ValueError("Pi0.5 LIBERO batch cannot be empty")
        transform, _norm_stats = self._transforms()
        tables: dict[Path, Any] = {}
        actions_by_source: dict[Path, np.ndarray] = {}
        transformed_samples = []
        target_actions = []
        noises = []
        timesteps = []
        for ref in refs:
            source = Path(ref.source_path).resolve()
            if not source.is_relative_to(self.dataset_root):
                raise ValueError(f"sample source is outside dataset root: {source}")
            if source not in self._verified_sources:
                if sha256_file(source) != ref.source_sha256:
                    raise ValueError(f"source hash differs for sample {ref.sample_id}")
                self._verified_sources.add(source)
            if source not in tables:
                tables[source] = pq.read_table(source)
                actions_by_source[source] = np.asarray(tables[source]["actions"].to_pylist(), dtype=np.float32)
            table = tables[source]
            if ref.frame_index >= table.num_rows:
                raise ValueError(f"frame {ref.frame_index} outside episode {ref.episode_index}")
            row = table.slice(ref.frame_index, 1).to_pylist()[0]
            action_window = _action_window(actions_by_source[source], ref.frame_index, self.config.action_horizon)
            if sha256_array(action_window) != ref.action_target_sha256:
                raise ValueError(f"action target hash differs for sample {ref.sample_id}")
            if hashlib.sha256(ref.task.encode()).hexdigest() != ref.instruction_sha256:
                raise ValueError(f"instruction hash differs for sample {ref.sample_id}")
            transformed = transform(
                {
                    "observation/image": self._decode_image(row["image"]),
                    "observation/wrist_image": self._decode_image(row["wrist_image"]),
                    "observation/state": np.asarray(row["state"], dtype=np.float32),
                    "actions": action_window,
                    "prompt": ref.task,
                }
            )
            transformed_samples.append(transformed)
            target_actions.append(action_window)
            noise_shape = (self.config.action_horizon, self.config.action_dim)
            noises.append(np.random.default_rng(ref.flow_noise_seed).standard_normal(noise_shape, dtype=np.float32))
            timesteps.append(ref.timestep)

        images = {
            key: np.stack(
                [
                    np.transpose(np.asarray(sample["image"][key], dtype=np.float32) / 255.0 * 2.0 - 1.0, (2, 0, 1))
                    for sample in transformed_samples
                ]
            )
            for key in PI05_CAMERA_KEYS
        }
        return {
            "images": images,
            "image_masks": {
                key: np.asarray([sample["image_mask"][key] for sample in transformed_samples], dtype=bool) for key in PI05_CAMERA_KEYS
            },
            "state": np.stack([np.asarray(sample["state"], dtype=np.float32) for sample in transformed_samples]),
            "tokenized_prompt": np.stack([np.asarray(sample["tokenized_prompt"], dtype=np.int64) for sample in transformed_samples]),
            "tokenized_prompt_mask": np.stack([np.asarray(sample["tokenized_prompt_mask"], dtype=bool) for sample in transformed_samples]),
            "noise": np.stack(noises),
            "timestep": np.asarray(timesteps, dtype=np.float32),
            "target_normalized_actions": np.stack([np.asarray(sample["actions"], dtype=np.float32) for sample in transformed_samples]),
            "target_actions": np.stack(target_actions),
        }

    def postprocess_actions(self, normalized_actions: np.ndarray, normalized_state: np.ndarray) -> np.ndarray:
        from openpi import transforms

        _input_transform, norm_stats = self._transforms()
        outputs = {"state": np.asarray(normalized_state), "actions": np.asarray(normalized_actions)}
        unnormalized = transforms.Unnormalize(norm_stats, use_quantiles=True)(outputs)["actions"]
        return np.asarray(unnormalized[..., : self.config.output_action_dim], dtype=np.float32)
