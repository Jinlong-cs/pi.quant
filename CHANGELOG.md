# Changelog

All notable public changes are recorded here. Package versions follow semantic
versioning; schema versions are independent and remain embedded in each
contract.

## [1.0.0] - Unreleased

### Added

- Stable `ArtifactLineageNode` and `ArtifactLineageManifest` contracts for
  source-to-promotion artifact graphs.
- Deterministic lineage hashing, stage and parent validation, evidence-boundary
  checks, optional filesystem SHA256 verification, and `piquant
  validate-lineage`.
- Root `piquant --version`, declared-Python CI coverage, and clean-wheel install
  smoke.
- PEP 561 `py.typed` metadata for external type-checker discovery.

### Stability

- Existing schema-version-1 plans, manifests, evidence, CLI commands, and
  imports remain readable; v1.0 additions are additive.
- The default package remains independent of Torch, OpenPI, FastWAM, ModelOpt,
  ONNX, ONNX Runtime, TensorRT, CUDA, datasets, and simulators.
- Library release acceptance remains separate from deployment-candidate,
  server/client, Gate40, full400, closed-loop, tag, and publication decisions.

## [0.5.0]

- Added deterministic mixed-precision candidate generation, source and target
  Pareto ranking, immutable candidate lineage, budget enforcement, and
  pending-first promotion plans.

## [0.4.0]

- Added target compiler, ONNX/TensorRT inspection, benchmark, and pi.cpp
  handoff evidence boundaries.

## [0.3.0]

- Added temporal/WAM manifests, captures, metrics, and rollout-aware study
  orchestration.

## [0.2.0]

- Added real Pi0.5/LIBERO manifests, FP golden capture, semantic sensitivity,
  and calibration-ablation contracts.

## [0.1.0]

- Established the lightweight contracts, explicit dependency-injection,
  ModelOpt, ORT, numerical-analysis, evidence-store, and synthetic workflow
  foundations.
