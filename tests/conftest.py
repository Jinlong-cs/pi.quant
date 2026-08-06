import numpy as np
import pytest

from piquant.adapters import SyntheticCalibrationProvider, SyntheticFlowVLAAdapter
from piquant.analysis import NumpyNumericalAnalyzer
from piquant.backends import ReferenceQDQBackend
from piquant.contracts import EvidenceRecord, OptimizationPlan, fingerprint
from piquant.evidence import target_fingerprint


@pytest.fixture
def synthetic_record() -> EvidenceRecord:
    plan = OptimizationPlan(
        plan_id="fixture",
        backend="reference_qdq",
        calibration={
            "dataset_id": "synthetic",
            "sample_count": 2,
            "seed": 3,
            "stages": ["approach"],
            "input_fields": ["observation"],
        },
        seed=7,
        capture_points=["action"],
        timing_boundary="offline",
    )
    calibration = SyntheticCalibrationProvider(2, 3)
    adapter = SyntheticFlowVLAAdapter(7)
    candidate, result = ReferenceQDQBackend().quantize(adapter, plan, calibration)
    batches = calibration.batches()
    reference = {"action": np.concatenate([adapter.forward(batch, ["action"])["action"] for batch in batches])}
    values = {"action": np.concatenate([candidate.forward(batch, ["action"])["action"] for batch in batches])}
    return EvidenceRecord(
        record_id="fixture",
        status="measured",
        model=adapter.spec,
        target=target_fingerprint(),
        plan=plan,
        backend=result.backend,
        representation=result.representation,
        calibration_fingerprint=calibration.fingerprint,
        module_coverage=result.module_coverage,
        quantization=result,
        comparison=NumpyNumericalAnalyzer(gripper_index=5).compare(reference, values, "action"),
        timing_boundary="offline",
        notes=[fingerprint(plan)],
    )
