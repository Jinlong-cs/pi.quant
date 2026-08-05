import numpy as np

from piquant.analysis import NumpyNumericalAnalyzer, compare_action, compare_tensor


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
