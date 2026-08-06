# Target Compiler Evidence

v0.4 introduces the deployment layer that sits after source/offline
quantization studies and before pi.cpp runtime promotion.

## Boundary

`CompilerBackend` consumes a frozen candidate artifact and a `CompilationPlan`.
It may build a target artifact and inspect graph/layer structure, but it does
not run ModelOpt, search precision, package pi.cpp assets, run server/client
traffic, execute LIBERO, or decide acceptance.

`DeploymentEvaluator` records timing for one compiled artifact under one
`BenchmarkProtocol`. The timing boundary must say whether it is build,
engine-stage, standalone, server inference, client roundtrip, or closed loop.

`DeploymentCandidateManifest` is the JSON handoff to pi.cpp. It is a lineage and
evidence document, not an accepted model package.

## TensorRT CLI Integration

The first compiler implementation is `TensorRTCliCompiler`. It shells out to
target-local `trtexec` and writes a `CompilerEvidenceRecord`. If `trtexec` is
not present, the result is `unsupported` with
`reason_code=trtexec-unavailable`. There is no CPU fallback and no TensorRT
Python import.

The command renderer is safe to run on a development machine:

```bash
uv run piquant trtexec-command recipes/deployment/agx-orin-tensorrt-int8.yaml \
  --engine /external/artifacts/candidate.engine \
  --layer-info /external/artifacts/candidate.layers.json
```

Target-local build requires explicit ownership of the GPU and output directory:

```bash
uv run piquant compile-tensorrt recipes/deployment/agx-orin-tensorrt-int8.yaml \
  --output-dir /external/artifacts/agx-build \
  --model-id fastwam --family wam --framework onnx --task wam \
  --action-dim 7 --action-horizon 32
```

## Evidence Levels

- capability: hardware/software support probe, including unsupported reason;
- graph: ONNX operators, initializers, dtypes, external data and candidates;
- build: engine artifact, build command, build time, log and layer report;
- stage: engine or model stage timing under a frozen protocol;
- runtime: pi.cpp standalone or server/client timing, owned outside pi.quant;
- closed-loop: LIBERO or robot evaluation, owned outside the compiler boundary;
- accepted: human promotion after reviewing all upstream evidence.

Unsupported and pending records are valid evidence. They must not be described
as performance results or success-preserving deployment.

## Current Hardware Gate

AGX Orin and RTX 5090 capability probes are kept in the external v0.4 Task
Contract. They are not committed to this repository because they include live
machine state. The public repo contains reusable contracts, templates, CLI and
tests only.
