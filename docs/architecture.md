# Architecture

`pi.quant` is the control plane around a quantization backend. The model and
calibration boundaries carry VLA semantics; the backend performs numerical
optimization; the analyzer and evidence store preserve reproducibility.

## Boundaries

- `ModelAdapter` owns model-specific forward inputs, named modules, captures,
  and action outputs.
- `CalibrationProvider` owns deterministic observation/language/history/action
  samples and stage distribution.
- `TaskLossProvider` owns a task-shaped offline loss such as flow-action MSE.
- `QuantizationBackend` owns one explicit backend invocation and reports module
  coverage plus fake/real representation.
- `NumericalAnalyzer` consumes named arrays, so PyTorch hooks and ORT outputs
  share metrics without sharing runtime dependencies.
- `TaskEvaluator` owns the current offline evaluation boundary.
- `EvidenceStore` persists structured evidence but cannot promote it.

There is no automatic backend registry in v0.1. The caller chooses a concrete
backend explicitly. This keeps optional packages lazy and prevents a recipe
from silently changing compiler or runtime semantics.

## Data flow

```text
ModelSpec + OptimizationPlan + CalibrationSpec
                         |
             adapter / calibration provider
                         |
                selected quant backend
                         |
            FP and candidate named captures
                         |
        tensor metrics + action metrics + evidence
```

The v0.1 reference QDQ backend is a portable test harness. ModelOpt is the
first optional production-oriented backend; TensorRT and hardware-specific
compiler work are future feature PRs.
