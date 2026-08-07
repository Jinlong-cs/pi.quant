# Architecture

`pi.quant` is the control plane around quantization and numerical-debug tools.
The model/data adapters carry VLA semantics, the backend generates one explicit
candidate, and the analyzer/evidence layers preserve comparable results.

## Boundaries

- `ModelAdapter` owns model-specific forward inputs, named modules, captures,
  and action outputs. `SemanticModelAdapter` adds stable module/capture IDs,
  action schema, and fresh model construction.
- `TorchQuantizableAdapter` is the narrow capability required by the ModelOpt
  backend. JAX and ONNX adapters do not inherit Torch behavior.
- `CalibrationProvider` owns deterministic observation/language/history/action
  samples and stage distribution. `ManifestCalibrationProvider` exposes exact
  sample lineage and a reproducible fingerprint.
- `TaskLossProvider` owns a task-shaped offline loss such as flow-action MSE.
- `QuantizationBackend` owns one explicit backend invocation and reports module
  coverage plus fake/real representation.
- `NumericalAnalyzer` consumes named arrays, so PyTorch hooks and ORT outputs
  share metrics without sharing runtime dependencies. The streaming analyzer
  retains paired distributions without retaining a full activation corpus.
- `TaskEvaluator` owns the current offline evaluation boundary.
- `EvidenceStore` persists structured evidence but cannot promote it.
- `TemporalModelAdapter` is a narrow WAM extension. It adds explicit temporal
  capture axes and separate teacher-forced/iterative execution; it does not
  make world-latent or rollout capture mandatory for ordinary VLA adapters.
- `piquant.temporal_study.TemporalSensitivityRunner` reuses the same
  fresh-model, explicit-backend, chunked-golden pattern for WAM sequences.
  Missing teacher-forced or latent callbacks fail fast instead of manufacturing
  a comparable signal.
- `CompilerBackend` is the target compiler boundary. It receives a frozen
  source/export candidate and returns build, graph, layer, artifact, command,
  and reason-code evidence for one target. It does not quantize weights, choose
  precision, own pi.cpp packaging, or decide acceptance.
- `DeploymentEvaluator` measures one compiled artifact under one
  `BenchmarkProtocol`. Engine-stage, standalone, server/client, and closed-loop
  timing remain separate records.

There is no automatic backend/model registry. The caller explicitly injects
the adapter, providers, backend, analyzer, evaluator, and store. This prevents
a recipe from silently changing model, dependency, precision, or runtime.

## Data flow

```text
ModelSpec + ActionSchema + CalibrationManifest
                         |
        semantic inventory + logical capture mapping
                         |
              chunked source FP golden
                         |
 OptimizationPlan -> fresh model -> selected quant backend
                         |
        streaming activation/flow/action diagnostics
                         |
       EvidenceRecord[] -> SensitivityStudyRecord
```

For temporal studies, the data path is:

```text
episode-disjoint sequence manifests
             |
  temporal FP golden (mode + axes + seed)
             |
  broad/component/rollback candidate
             |
  streaming stage/timestep/denoise/horizon metrics
             |
  action + optional latent rollout divergence
```

Target compiler studies add a downstream path:

```text
source/offline candidate evidence
             |
       frozen ONNX artifact
             |
 CompilationPlan -> target-local TensorRT build
             |
 ONNX graph + TensorRT layer + artifact evidence
             |
 StageTimingReport[] -> DeploymentCandidateManifest
             |
 pi.cpp handoff pending human promotion
```

## Stable identities

Recipes select semantic logical IDs such as `vlm.block.09.attention.q`; the
adapter resolves them to current framework paths. Evidence records both forms,
matched module/parameter counts, and exact enabled quantizers. A selector that
matches zero modules fails before calibration.

Every sensitivity candidate reloads the same immutable checkpoint. ModelOpt
fake-quant mutation from one candidate cannot leak into another. The plan,
trial, manifest, golden, candidate evidence, and artifact hashes are checked
before a study can be validated.

## Ownership boundary

The reference QDQ backend remains a synthetic harness. ModelOpt is the first
real candidate backend. ORT is a temporary-graph capture integration, not a
quantizer. FastWAM temporal execution is an explicit optional source
integration; it does not imply a world model is available. TensorRT compilation
is now a target evidence layer, but pi.cpp runtime packaging, server/client
operation, full closed loop, and human acceptance remain outside the compiler
boundary.
