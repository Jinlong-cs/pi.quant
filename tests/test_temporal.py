from __future__ import annotations

import numpy as np
import pytest

from piquant.contracts import ActionSchema, SequenceRef, TemporalCalibrationManifest, TemporalCaptureSpec
from piquant.integrations.fastwam import FastWAMCaptureRunner
from piquant.temporal import (
    RolloutDivergenceAccumulator,
    TemporalSensitivityRunner,
    TemporalStreamingAnalyzer,
    require_temporal_episode_disjoint,
)
from piquant.temporal_study import TemporalSensitivityRunner as TemporalStudyRunner


def _schema() -> ActionSchema:
    return ActionSchema(
        model_action_dim=7,
        output_action_dim=7,
        horizon=2,
        denoise_steps=2,
        translation_indices=[0, 1, 2],
        rotation_indices=[3, 4, 5],
        gripper_index=6,
        gripper_threshold=0.0,
        postprocess="identity",
    )


def _sequence(sequence_id: str, split: str, episode_index: int) -> SequenceRef:
    return SequenceRef(
        sequence_id=sequence_id,
        split=split,  # type: ignore[arg-type]
        suite="suite",
        task="task",
        task_index=0,
        episode_index=episode_index,
        frame_start=0,
        frame_end=4,
        observation_indices=[0, 1, 2],
        camera_keys=["wrist", "front"],
        instruction_sha256="0" * 64,
        action_target_sha256="1" * 64,
        normalization_revision="norm-v1",
        stage="approach",
        source_path="external://episode",
        source_sha256="2" * 64,
        seed=7,
        flow_noise_seed=11,
        timestep_schedule=[1.0, 0.0],
        denoise_steps=2,
        action_horizon=2,
        action_dim=7,
        sampling_mode="temporal_balanced",
    )


def _manifest(manifest_id: str, split: str, episode_index: int) -> TemporalCalibrationManifest:
    return TemporalCalibrationManifest(
        manifest_id=manifest_id,
        dataset_id="dataset",
        dataset_revision="rev",
        split=split,  # type: ignore[arg-type]
        preprocess_revision="preprocess-v1",
        normalization_revision="norm-v1",
        stage_rule="unit-test",
        sampling_mode="temporal_balanced",
        denoise_steps=2,
        action_horizon=2,
        sequences=[_sequence(f"{manifest_id}-seq", split, episode_index)],
    )


def test_temporal_manifests_reject_episode_overlap() -> None:
    calibration = _manifest("calibration", "calibration", 3)
    holdout = _manifest("holdout", "diagnostic_holdout", 3)
    with pytest.raises(ValueError, match="episode overlap"):
        require_temporal_episode_disjoint(calibration, holdout)


def test_temporal_runner_import_path_remains_compatible() -> None:
    assert TemporalSensitivityRunner is TemporalStudyRunner


def test_temporal_analyzer_rejects_axis_rank_mismatch() -> None:
    capture = TemporalCaptureSpec(
        logical_id="action.flow",
        backend_path="action.head",
        component="action_boundary",
        kind="flow",
        axes=["batch", "denoise_step", "horizon", "action_dim"],
    )
    analyzer = TemporalStreamingAnalyzer(seed=1)
    with pytest.raises(ValueError, match="rank does not match"):
        analyzer.add(
            {"action.flow": np.ones((1, 2, 7), dtype=np.float32)},
            {"action.flow": np.ones((1, 2, 7), dtype=np.float32)},
            [{"stage": "approach", "timestep": 1.0}],
            [capture],
            mode="iterative",
            action_schema=_schema(),
        )


def test_temporal_analyzer_keeps_teacher_forced_and_iterative_separate() -> None:
    capture = TemporalCaptureSpec(
        logical_id="hidden",
        backend_path="block.0",
        component="video_backbone",
        kind="activation",
        axes=["batch", "hidden"],
    )
    analyzer = TemporalStreamingAnalyzer(seed=1)
    reference = {"hidden": np.ones((1, 3), dtype=np.float32)}
    candidate = {"hidden": np.full((1, 3), 0.9, dtype=np.float32)}
    analyzer.add(reference, candidate, [{"stage": "approach", "timestep": 1.0}], [capture], mode="teacher_forced", action_schema=_schema())
    with pytest.raises(ValueError, match="separate analyzers"):
        analyzer.add(reference, candidate, [{"stage": "approach", "timestep": 1.0}], [capture], mode="iterative", action_schema=_schema())


def test_temporal_action_and_rollout_metrics_track_steps_and_horizons() -> None:
    schema = _schema()
    flow = TemporalCaptureSpec(
        logical_id="action.flow",
        backend_path="action.head",
        component="action_boundary",
        kind="flow",
        axes=["batch", "denoise_step", "horizon", "action_dim"],
    )
    reference = {"action.flow": np.ones((1, 2, 2, 7), dtype=np.float32)}
    candidate = {"action.flow": reference["action.flow"].copy()}
    candidate["action.flow"][0, 1, :, 0] += 0.25

    analyzer = TemporalStreamingAnalyzer(seed=4)
    analyzer.add(reference, candidate, [{"stage": "lift", "timestep": 0.5}], [flow], mode="iterative", action_schema=schema)
    report = analyzer.finalize()[0]
    assert report.action is not None
    assert sorted(report.by_denoise_step) == ["0", "1"]
    assert sorted(report.by_stage) == ["lift"]
    assert sorted(report.by_timestep) == ["0.500"]
    assert len(report.action.horizon_l2) == 2

    action = TemporalCaptureSpec(
        logical_id="action.rollout",
        backend_path="$rollout_action",
        component="action_boundary",
        kind="rollout",
        axes=["batch", "rollout_horizon", "horizon", "action_dim"],
    )
    latent = TemporalCaptureSpec(
        logical_id="world.latent",
        backend_path="world.latent",
        component="world_model",
        kind="latent",
        axes=["batch", "rollout_horizon", "hidden"],
    )
    rollout = RolloutDivergenceAccumulator(action, [latent], mode="iterative", exceedance_threshold=0.1, seed=9)
    rollout.add(
        {
            "action.rollout": np.ones((1, 3, 2, 7), dtype=np.float32),
            "world.latent": np.ones((1, 3, 4), dtype=np.float32),
        },
        {
            "action.rollout": np.ones((1, 3, 2, 7), dtype=np.float32) + 0.2,
            "world.latent": np.ones((1, 3, 4), dtype=np.float32) + 0.1,
        },
        schema,
    )
    divergence = rollout.finalize()
    assert divergence.rollout_horizon_steps == [1, 2, 3]
    assert divergence.latent_horizon_steps == [1, 2, 3]
    assert divergence.exceedance_rate == 1.0
    assert sorted(divergence.action_l2_by_horizon) == ["1", "2", "3"]
    assert sorted(divergence.latent_relative_l2_by_horizon) == ["1", "2", "3"]


def test_fastwam_capture_provider_preserves_action_abi() -> None:
    torch = pytest.importorskip("torch")
    action = TemporalCaptureSpec(
        logical_id="action.output",
        backend_path="$infer_action",
        component="action_boundary",
        kind="action",
        axes=["batch", "horizon", "action_dim"],
    )

    def provider(_model: object, _batch: dict[str, object], _specs: list[TemporalCaptureSpec], _mode: str) -> dict[str, object]:
        return {"action": torch.zeros((32, 7))}

    result = FastWAMCaptureRunner(capture_provider=provider).run(object(), {}, [action], mode="iterative")
    assert result["action.output"].shape == (1, 32, 7)

    def invalid_provider(_model: object, _batch: dict[str, object], _specs: list[TemporalCaptureSpec], _mode: str) -> dict[str, object]:
        return {"action": torch.zeros((1, 31, 7))}

    with pytest.raises(ValueError, match="must match"):
        FastWAMCaptureRunner(capture_provider=invalid_provider).run(object(), {}, [action], mode="iterative")

    class NestedOutputModel:
        def named_modules(self) -> list[tuple[str, object]]:
            return []

        def infer_action(self, **_kwargs: object) -> dict[str, object]:
            return {"action": {"nested": torch.zeros((32, 7))}}

    with pytest.raises(TypeError, match="one tensor"):
        FastWAMCaptureRunner().run(
            NestedOutputModel(),
            {"input_image": object(), "prompt": "test"},
            [action],
            mode="iterative",
        )
