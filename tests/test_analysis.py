import numpy as np

from piquant.analysis import NumpyNumericalAnalyzer, StreamingDiagnosticAnalyzer, compare_action, compare_tensor
from piquant.contracts import ActionSchema


def test_tensor_metrics_include_sqnr_and_cosine() -> None:
    metric = compare_tensor(np.array([[1.0, 2.0]]), np.array([[1.0, 2.5]]))
    assert metric.shape_match is True
    assert metric.finite is True
    assert metric.max_abs == 0.5
    assert metric.cosine is not None and metric.cosine < 1.0
    assert metric.sqnr_db is not None


def test_action_metrics_include_direction_and_gripper_mismatch() -> None:
    reference = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 1.0]])
    candidate = np.array([[0.2, 0.3, 0.4, 0.5, 0.6, 0.0]])
    metric = compare_action(reference, candidate, gripper_index=5)
    assert metric.l2_mean == 1.0
    assert metric.direction_cosine_mean is not None
    assert metric.gripper_mismatch_rate == 1.0


def test_named_analyzer_rejects_missing_capture() -> None:
    analyzer = NumpyNumericalAnalyzer(gripper_index=1)
    try:
        analyzer.compare({"hidden": np.ones(1), "action": np.ones(2)}, {"action": np.ones(2)}, "action")
    except ValueError as error:
        assert "missing tensors" in str(error)
    else:
        raise AssertionError("missing capture did not fail")


def test_streaming_diagnostics_preserve_stage_timestep_and_horizon_axes() -> None:
    schema = ActionSchema(
        model_action_dim=7,
        output_action_dim=7,
        horizon=1,
        denoise_steps=1,
        translation_indices=[0, 1, 2],
        rotation_indices=[3, 4, 5],
        gripper_index=6,
        postprocess="identity",
    )
    reference = {"hidden": np.ones((2, 3)), "action": np.ones((2, 1, 7))}
    candidate = {"hidden": np.full((2, 3), 0.9), "action": np.full((2, 1, 7), 0.9)}
    metadata = [{"stage": "early", "timestep": 1.0}, {"stage": "late", "timestep": 0.1}]
    analyzer = StreamingDiagnosticAnalyzer(seed=3)
    analyzer.add(reference, candidate, metadata, action_name="action", action_schema=schema)
    diagnostics = analyzer.finalize()
    assert diagnostics.overall.sample_count == 2
    assert len(diagnostics.overall.action.horizon_l2) == 1
    assert sorted(diagnostics.by_stage) == ["early", "late"]
    assert sorted(diagnostics.by_timestep) == ["0.100", "1.000"]
