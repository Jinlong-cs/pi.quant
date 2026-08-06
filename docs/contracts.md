# Contracts

All public records are Pydantic models with `extra="forbid"`; undocumented
recipe or evidence fields fail at the boundary. JSON and YAML enter the same
versioned schema.

## Candidate contracts

- `ModelSpec` fixes model identity, family, framework revision, task, action
  dimension, and horizon.
- `ActionSchema` separates model/output dimensions and fixes horizon, denoise
  steps, translation/rotation/gripper indices, threshold, and postprocess.
- `ModuleDescriptor` maps a stable logical selector to the backend path,
  component, block, operator family, parameters, and quantizable state.
- `CaptureSpec` maps one logical diagnostic tensor to its concrete hook/output.
- `CalibrationManifest` and `SampleRef` preserve dataset/preprocess/
  normalization identity, episode/frame/window, cameras, instruction/action
  hashes, stage, seed, flow noise, timestep, and source file hash.

`OptimizationPlan` is the only interpretation of a YAML recipe. It fixes the
backend, format, fake/real representation, module patterns, calibration
identity, seed, capture points, and timing boundary.

The plan calibration identity, sample count, stages, and seed must match the
manifest used by a trial. An FP control requires `backend=none`; a ModelOpt
candidate requires INT8 `fake_quant`. These labels cannot be changed after the
fact to imply packed quantization.

## Study contracts

- `GoldenCaptureManifest` indexes chunked FP captures, sample order, capture
  specs, model/action contract, holdout fingerprint, and hashes.
- `SensitivityTrial` distinguishes FP, broad, component-only, component
  rollback, block-group, block, layer, calibration-control, and matched
  calibration-rollback experiments.
- `EvidenceRecord` is one immutable candidate execution with plan, target,
  coverage, quantizer metadata, paired diagnostics, command, and boundary.
- `CandidateEvidenceRef` links a trial to one hashed record.
- `SensitivityStudyRecord` indexes candidates, recovery ranking, calibration
  drift and rank-change ablations, golden, inventory, and split fingerprints
  without embedding raw tensors.

`piquant validate-study` re-hashes the golden and candidate records and checks
model, action, plan, trial, calibration, coverage, and status lineage. An Agent
may write `measured`, `pending`, or `rejected` evidence. Only a human promotion
owner can mark deployment evidence `accepted`.
