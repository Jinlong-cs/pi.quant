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

## Temporal contracts

- `SequenceRef` binds one temporal window to an episode, camera keys,
  instruction/action hashes, stage, normalization, seed, flow noise, timestep
  schedule, denoise steps, and action ABI.
- `TemporalCalibrationManifest` defines calibration, diagnostic holdout,
  static-control, or promotion-reserved sequence sets. Callers must enforce
  episode disjointness before a study; `require_temporal_episode_disjoint`
  provides the shared guard.
- `TemporalCaptureSpec` requires explicit tensor axes and distinguishes
  activation, flow, action, latent, and rollout captures.
- `TemporalMetricReport` keeps teacher-forced and iterative reports separate
  and buckets stage, timestep, denoise step, and action horizon. Optional
  `RolloutDivergenceReport` records action and world-latent drift only when the
  source adapter actually exposes those tensors.
- Action horizon and rollout horizon are distinct axes. Action horizon indexes
  the actions inside one policy output; rollout horizon indexes successive
  policy/environment states and must come from a `kind=rollout` capture that
  explicitly declares a `rollout_horizon` axis.
- `TemporalGoldenCaptureManifest` and `TemporalStudyRecord` carry the same
  artifact hashes and candidate lineage as the VLA study contracts, with an
  explicit source/offline-temporal boundary.

`piquant validate-study` re-hashes the golden and candidate records and checks
model, action, plan, trial, calibration, coverage, and status lineage. An Agent
may write `measured`, `pending`, or `rejected` evidence. Only a human promotion
owner can mark deployment evidence `accepted`.

## Target compiler contracts

- `TargetFingerprint` records the measured target environment: platform, Python,
  device, GPU, compute capability, driver, CUDA, TensorRT, memory, power/clocks
  and container identity. Unknown values remain explicit strings rather than
  inferred support.
- `TargetCapability` records one capability probe such as INT8, FP8, NVFP4,
  QDQ parser support, DLA coverage or profiling availability. `unsupported`
  and `pending` capabilities require a reason code.
- `CompilationPlan` is a versioned YAML/JSON schema for one target compiler
  invocation. It binds an ONNX artifact hash, target, precision, shape profiles,
  TensorRT builder options, timing cache, flags and timing boundary.
- `OperatorGraphReport` is source/export graph evidence only. Node counts and
  constant-weight candidates are not FLOP, latency or success claims.
- `CompilerEvidenceRecord` is one build attempt. `unsupported`, `pending` and
  `rejected` records are valid evidence only with reason codes; `measured`
  records must point to hashed artifacts.
- `BenchmarkProtocol` and `StageTimingReport` separate build, engine-stage,
  standalone, server inference, client roundtrip and closed-loop timing. A
  measured timing report requires latency distribution.
- `DeploymentCandidateManifest` is the pi.cpp handoff document. It carries
  model/recipe/calibration lineage, ABI, compiler records, timing records,
  precision map and integration status. `status=accepted` requires explicit
  `human_acceptance=accepted`; agents cannot promote by writing machine
  evidence.
