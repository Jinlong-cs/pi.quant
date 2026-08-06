# pi.quant

`pi.quant` is a quantization control plane for vision-language-action and
world-action models. It binds generic quantization backends to the model, data,
capture, action, temporal, target, and promotion semantics that determine
whether a robotics candidate is actually comparable.

The library does not reimplement NVIDIA ModelOpt or turn offline error into a
deployment claim. It makes every candidate traceable from an immutable model
and calibration contract through named intermediate tensors, action behavior,
module coverage, artifacts, and an explicit evidence boundary.

## v0.2

v0.2 adds a real Pi0.5/LIBERO source-level vertical workflow:

- checkpoint-aligned OpenPI PyTorch adapter for three image slots, token length
  200, horizon 50, ten denoise steps, model output `[50,32]`, and LIBERO output
  `[50,7]`;
- episode-disjoint, stage-aware calibration and diagnostic manifests with
  source, normalization, noise, timestep, and action-target fingerprints;
- stable semantic module and capture IDs mapped to exact backend paths;
- chunked FP golden capture and fresh-model broad, component-only, rollback,
  block/layer, and calibration-control trials;
- paired activation, flow, action, direction, translation, rotation, gripper,
  horizon, stage, timestep, bootstrap, coverage, and quantizer evidence;
- `SensitivityStudyRecord`, `piquant validate-study`, and
  `piquant summarize-study` for external artifact lineage.

The v0.2 release gate used one exact OpenPI revision and checkpoint with 160
stage-balanced calibration samples and 160 episode-disjoint diagnostic samples
across 40 LIBERO tasks. Matched JAX/PyTorch `[50,7]` parity reached `0.999996`
cosine and `0.003065` relative L2. Broad INT8 fake quant covered `418` modules
and `99.9988%` of eligible parameters; a 54-trial hierarchical rollback and
calibration-ablation study then identified `vlm.block.07.mlp.down` and
`action.block.14.mlp.down`. Keeping only
those two projections in source precision retained `416/418` quantized modules
and `98.61%` parameter coverage while recovering `24.01%` of broad action L2
error and `11.43%` of broad flow L2 error. Stage-balanced calibration reduced
action L2 from `0.296083` for the matched random-frame control to `0.216278`.
Across seven matched component rollbacks, the calibration-specific sensitivity
rankings had Spearman correlation `0.642857`: VLM MLP and action backbone stayed
first and second, while secondary components changed order.

This is real source/offline ModelOpt INT8 fake-quant evidence. v0.2 does not
claim ONNX/TensorRT parity, packed INT8, AGX or RTX 5090 latency, LIBERO
closed-loop success, or accepted deployment.

## Install

```bash
uv sync --extra dev
uv run piquant doctor
uv run piquant validate-plan recipes/synthetic/flow-vla-int8.yaml
```

The default package imports NumPy/Pydantic/YAML only. Torch, OpenPI, ModelOpt,
ONNX, ORT, TensorRT, CUDA, datasets, and simulators remain external or optional.
Install only the integration being used:

```bash
uv sync --extra modelopt
uv sync --extra onnx
uv sync --extra pi05
uv run pytest -q -m modelopt
uv run pytest -q -m ort
```

## Portable synthetic workflow

Keep generated captures and evidence outside Git:

```bash
uv run python examples/synthetic_flow_vla/run.py \
  --recipe recipes/synthetic/flow-vla-int8.yaml \
  --output-dir /tmp/piquant-v0.1-evidence
```

The reference QDQ path is an explicitly selected portable numerical harness;
it is not a fallback, ModelOpt result, target engine, or closed-loop result.

## Real Pi0.5 workflow

Run the real integration inside a checkpoint-aligned OpenPI environment. Model,
dataset, manifests, captures, and evidence stay under an external artifact
root and are never committed.

```bash
uv run piquant validate-plan recipes/pi05/libero-fp-control.yaml
uv run piquant validate-plan recipes/pi05/libero-int8-broad.yaml
uv run python examples/pi05_libero/run.py --help
```

The complete manifest -> golden -> trial commands and external identity
boundary are in [examples/pi05_libero/README.md](examples/pi05_libero/README.md).
A sensitivity study uses one immutable candidate directory per trial and fresh
checkpoint weights for every fake-quant mutation.

## v0.3 FastWAM temporal diagnostics

v0.3 extends the same contracts to WAM sequences without turning a model's
exporter names into universal rules:

- `FastWAMSourceAdapter` exposes semantic module inventory and explicit backend
  paths for the audited 33-frame, `[32,7]` action ABI;
- `FastWAMCaptureRunner` hooks iterative source inference lazily and requires
  injected callbacks for teacher-forced execution or world-latent capture;
- temporal manifests enforce episode-disjoint splits and preserve stage,
  timestep, denoise-step, action-horizon, seed, and flow-noise lineage;
- streaming diagnostics separate teacher-forced and iterative modes and report
  activation, flow, action, direction, gripper, stage, timestep, denoise-step,
  and horizon metrics;
- rollout divergence is reported as an explicit offline diagnostic and is not a
  closed-loop success or deployment acceptance claim.

The public FastWAM recipes are templates. Replace their external dataset
identity and sample count with real manifests before running a study:

```bash
uv run piquant validate-plan recipes/fastwam/temporal-fp-control.yaml
uv run piquant validate-plan recipes/fastwam/temporal-int8-broad.yaml
```

The complete adapter wiring is documented in
[examples/fastwam_temporal/README.md](examples/fastwam_temporal/README.md).
Real captures, model assets, experiment logs, and evidence remain under an
external Task Contract artifact root. v0.3 source/offline evidence does not
claim TensorRT, AGX, RTX 5090, server/client timing, full LIBERO promotion, or
the historical five-node precision guard.

## v0.4 target compiler evidence

v0.4 adds the deployment evidence layer between source-level quantization and
pi.cpp runtime promotion:

- `CompilationPlan` describes one frozen ONNX candidate, one target fingerprint,
  one precision mode, shape profiles, TensorRT builder flags, and timing
  boundary;
- `TensorRTCliCompiler` is an optional CLI integration around target-local
  `trtexec`; if `trtexec` is absent it returns structured `unsupported`
  evidence instead of falling back;
- ONNX inspection reports operators, initializers, dtype counts, external data,
  and constant-weight Conv/Gemm/MatMul candidates without making latency claims;
- TensorRT layer inspection records dtype, Q/DQ, reformat/copy, fusion and
  tactic counts from `--exportLayerInfo`;
- `BenchmarkProtocol`, `StageTimingReport`, and
  `DeploymentCandidateManifest` keep engine-stage, standalone, server/client,
  closed-loop, pi.cpp handoff, and human acceptance separate.

Validate public templates and render a target command without running hardware:

```bash
uv run piquant validate-compilation-plan recipes/deployment/agx-orin-tensorrt-int8.yaml
uv run piquant trtexec-command recipes/deployment/agx-orin-tensorrt-int8.yaml \
  --engine /external/artifacts/candidate.engine \
  --layer-info /external/artifacts/candidate.layers.json
```

Run compilation only inside an authorized target environment:

```bash
uv run piquant compile-tensorrt recipes/deployment/agx-orin-tensorrt-int8.yaml \
  --output-dir /external/artifacts/agx-build \
  --model-id fastwam --family wam --framework onnx --task wam \
  --action-dim 7 --action-horizon 32
uv run piquant inspect-trt-layers /external/artifacts/agx-build/agx-orin-tensorrt-int8-template.layers.json
```

The current AGX/RTX 5090 hardware build gates remain pending until target
storage, ownership, TensorRT environment, parity and benchmark protocols are
verified. Capability probes are not performance results.

## Architecture

```text
ModelAdapter + CalibrationProvider + semantic inventory
                         |
             versioned plan and manifest contracts
                         |
          ModelOpt candidate on fresh model weights
                         |
       named FP/candidate captures + streaming metrics
                         |
      EvidenceRecord -> StudyRecord -> human promotion
```

Read [docs/architecture.md](docs/architecture.md), [docs/contracts.md](docs/contracts.md),
[docs/modelopt-backend.md](docs/modelopt-backend.md), and
[docs/target-compiler.md](docs/target-compiler.md) for the public boundaries.
Agent workflows live under `.agents/`; the root `AGENTS.md` is the repository
coding policy copied from the project-level policy source.

## Roadmap

The version sequence is intentionally serial. Mixed-precision promotion remains
a separate future feature PR; see [docs/roadmap.md](docs/roadmap.md).
