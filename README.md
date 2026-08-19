# pi.quant

`pi.quant` is a quantization control plane for vision-language-action (VLA)
and world-action (WAM) models. It provides versioned contracts, explicit
integration boundaries, numerical diagnostics, target compiler evidence,
mixed-precision search, artifact lineage, and promotion planning.

It does not replace a model runtime, quantization algorithm, TensorRT runtime,
`pi.cpp`, a simulator, or a closed-loop evaluator. Those systems are injected
or operated by their owning environments.

## Features

- Versioned Pydantic contracts for model, action, calibration, capture,
  candidate, compiler, benchmark, search, and promotion records.
- Explicit model adapters for Pi0.5, FastWAM, and synthetic workflows.
- Lazy optional integrations for ModelOpt, OpenPI, ONNX, ONNX Runtime,
  TensorRT, CUDA, and simulator environments.
- Streaming activation, action, temporal, rollout, sensitivity, and coverage
  diagnostics with machine-readable evidence and rejection reasons.
- ONNX inspection and target-local TensorRT compilation evidence without
  claiming deployment success from a build or benchmark alone.
- Deterministic mixed-precision candidate generation and separate source and
  target Pareto ranking.
- Hash-stable artifact lineage from source identity through calibration,
  candidate, export, compiler, benchmark, server/client, closed loop, and
  promotion evidence.
- Pending-first promotion plans. Only an explicit human decision can accept a
  deployment candidate.

## Install

From a source checkout:

```bash
uv sync --extra dev
uv run piquant --version
uv run piquant doctor
```

The default runtime dependencies are NumPy, Pydantic, and PyYAML. Optional
model and compiler runtimes are not imported by `import piquant`, `doctor`,
plan validation, search recipe generation, ranking, promotion planning, or
lineage validation.

Install only the integration required by the workflow:

```bash
uv sync --extra modelopt
uv sync --extra onnx
uv sync --extra pi05
```

## Workflows

All models, datasets, captures, engines, logs, and evidence must be stored in
an external artifact root. Use a fresh candidate directory for every run.

### Synthetic local check

Run the portable deterministic reference workflow:

```bash
uv run python examples/synthetic_flow_vla/run.py \
  --recipe recipes/synthetic/flow-vla-int8.yaml \
  --output-dir /external/artifacts/synthetic
```

This is an SDK and evidence-format check. It is not Pi0.5, FastWAM, TensorRT,
hardware, or closed-loop evidence.

### Pi0.5 source diagnostics

Use an identity-matched OpenPI environment with the `pi05` and `modelopt`
extras. Keep the checkpoint, LIBERO data, identity manifest, captures, and
study output outside the repository.

```bash
uv run piquant validate-plan recipes/pi05/libero-fp-control.yaml
uv run piquant validate-plan recipes/pi05/libero-int8-broad.yaml
uv run python examples/pi05_libero/run.py --help
```

The complete manifest, golden-capture, trial, and study commands are in
[examples/pi05_libero/README.md](examples/pi05_libero/README.md).

### FastWAM temporal diagnostics

Wire `FastWAMSourceAdapter` to the exact source model and inject the temporal
calibration, capture, teacher-forced, or world-latent callbacks required by
the study. Missing source capabilities fail fast and remain pending.

```bash
uv run piquant validate-plan recipes/fastwam/temporal-fp-control.yaml
uv run piquant validate-plan recipes/fastwam/temporal-int8-broad.yaml
```

Adapter wiring and temporal callback requirements are documented in
[examples/fastwam_temporal/README.md](examples/fastwam_temporal/README.md).

### Target compiler evidence

Validate a target plan and render its TensorRT command without running a
hardware workload:

```bash
uv run piquant validate-compilation-plan \
  recipes/deployment/agx-orin-tensorrt-int8.yaml
uv run piquant trtexec-command \
  recipes/deployment/agx-orin-tensorrt-int8.yaml \
  --engine /external/artifacts/candidate.engine \
  --layer-info /external/artifacts/candidate.layers.json
```

In an authorized target environment, compilation and layer inspection use
explicit model identity and external output paths:

```bash
uv run piquant compile-tensorrt \
  recipes/deployment/agx-orin-tensorrt-int8.yaml \
  --output-dir /external/artifacts/agx-build \
  --model-id fastwam --family wam --framework onnx --task wam \
  --action-dim 7 --action-horizon 32
uv run piquant inspect-trt-layers \
  /external/artifacts/agx-build/agx-orin-tensorrt-int8-template.layers.json
```

Compiler records, layer inspection, parity, stage timing, standalone timing,
server/client timing, closed-loop results, and human acceptance are separate
evidence lanes.

### Search and promotion planning

Search and ranking operate only on explicit plans and candidate records:

```bash
uv run piquant validate-search-plan /external/artifacts/search-plan.json
uv run piquant search /external/artifacts/search-plan.json
uv run piquant rank /external/artifacts/candidates.json \
  --boundary target \
  --search-plan /external/artifacts/search-plan.json
uv run piquant promote /external/artifacts/candidate.json \
  --baseline-candidate /external/artifacts/fp-control.json \
  --target-front /external/artifacts/target-front.json \
  --search-plan /external/artifacts/search-plan.json
```

These commands do not discover a backend, run a compiler, launch a benchmark,
execute a promotion gate, or assign human acceptance.

### Artifact lineage

Validate a complete lineage independently of model and compiler runtimes:

```bash
uv run piquant validate-lineage /external/artifacts/artifact-lineage.json
uv run piquant validate-lineage \
  /external/artifacts/artifact-lineage.json \
  --check-artifacts
```

Lineage validation checks schema, stage order, parent identity, terminal
status, evidence boundary, canonical hash, and optionally local artifact
SHA256 values. It validates evidence identity; it does not execute or accept
the referenced deployment.

## Development

Run the repository checks before submitting a change:

```bash
uv sync --frozen --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
uv build
```

Tests are deterministic contract and integration regressions. Hardware
probes, experiment runners, model identity records, generated evidence, and
large assets belong in the external Task Contract artifact root, not in this
repository.

## Documentation

- [Architecture](docs/architecture.md)
- [Contracts](docs/contracts.md)
- [ModelOpt backend](docs/modelopt-backend.md)
- [Target compiler](docs/target-compiler.md)
- [Release and compatibility policy](docs/release.md)
