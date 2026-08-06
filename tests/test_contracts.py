from pathlib import Path

import pytest

from piquant.contracts import ActionSchema, CaptureSpec, GoldenCaptureManifest, ModelSpec, OptimizationPlan, fingerprint, load_plan


def test_recipe_loads_through_one_schema() -> None:
    plan = load_plan(Path("recipes/synthetic/flow-vla-int8.yaml"))
    assert isinstance(plan, OptimizationPlan)
    assert plan.backend == "modelopt"
    assert plan.calibration.stages == ["approach", "grasp", "lift", "place"]
    assert fingerprint(plan) == fingerprint(plan.model_copy(deep=True))


def test_pi05_recipes_share_the_audited_capture_and_calibration_contract() -> None:
    control = load_plan(Path("recipes/pi05/libero-fp-control.yaml"))
    broad = load_plan(Path("recipes/pi05/libero-int8-broad.yaml"))
    assert control.capture_points == broad.capture_points
    assert len(control.capture_points) == 33
    assert control.calibration == broad.calibration
    assert control.representation == "fp_control"
    assert broad.representation == "fake_quant"


def test_unknown_recipe_fields_fail_fast() -> None:
    with pytest.raises(ValueError):
        OptimizationPlan.model_validate(
            {
                "plan_id": "bad",
                "backend": "modelopt",
                "calibration": {
                    "dataset_id": "synthetic",
                    "sample_count": 1,
                    "seed": 1,
                    "stages": ["approach"],
                    "input_fields": ["observation"],
                    "unexpected": True,
                },
                "seed": 1,
                "capture_points": ["action"],
                "timing_boundary": "offline",
            }
        )


def test_golden_manifest_rejects_duplicate_sample_lineage() -> None:
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        GoldenCaptureManifest(
            manifest_id="duplicate",
            model=ModelSpec(model_id="model", family="test", framework="test", action_dim=7, action_horizon=1),
            action_schema=ActionSchema(
                model_action_dim=7,
                output_action_dim=7,
                horizon=1,
                denoise_steps=1,
                translation_indices=[0, 1, 2],
                rotation_indices=[3, 4, 5],
                gripper_index=6,
                postprocess="identity",
            ),
            holdout_manifest_fingerprint="0" * 64,
            capture_specs=[CaptureSpec(logical_id="action", backend_path="$action", component="head", kind="action")],
            chunks=[
                {"chunk_id": "a", "sample_ids": ["sample"], "artifact": {"kind": "capture", "path": "/a", "sha256": "1" * 64}},
                {"chunk_id": "b", "sample_ids": ["sample"], "artifact": {"kind": "capture", "path": "/b", "sha256": "2" * 64}},
            ],
            seed=1,
            status="measured",
        )
