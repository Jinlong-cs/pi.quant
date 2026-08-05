from pathlib import Path

import pytest

from piquant.contracts import OptimizationPlan, fingerprint, load_plan


def test_recipe_loads_through_one_schema() -> None:
    plan = load_plan(Path("recipes/synthetic/flow-vla-int8.yaml"))
    assert isinstance(plan, OptimizationPlan)
    assert plan.backend == "modelopt"
    assert plan.calibration.stages == ["approach", "grasp", "lift", "place"]
    assert fingerprint(plan) == fingerprint(plan.model_copy(deep=True))


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
