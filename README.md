# pi.quant

`pi.quant` is a quantization control plane for vision-language-action models.
It keeps VLA-specific calibration, intermediate-layer diagnostics, action
metrics, experiment lineage, and promotion evidence together while delegating
quantization algorithms to established backends such as NVIDIA ModelOpt.

The important boundary is deliberate: `pi.quant` does not pretend that a
generic PTQ API is a complete VLA quantization system. A candidate must be
traceable from the model contract and calibration distribution through
intermediate tensors and action behavior before it can become a deployment
candidate.

## v0.1 status

v0.1 ships the control-plane contracts and a portable synthetic flow-VLA
vertical workflow. It does not claim Pi0.5, FastWAM, TensorRT, AGX, RTX 5090,
or closed-loop support. The optional ModelOpt and ONNX Runtime integrations
are isolated from the default installation and reported as separate evidence
gates.

## Install

```bash
uv sync --extra dev
uv run piquant doctor
uv run piquant validate-plan recipes/synthetic/flow-vla-int8.yaml
```

The default package imports NumPy/Pydantic/YAML only. Install a backend or
debug integration explicitly:

```bash
uv sync --extra modelopt
uv sync --extra onnx
uv run pytest -q -m modelopt
uv run pytest -q -m ort
```

## Reproduce the v0.1 workflow

Keep generated captures and evidence outside Git:

```bash
uv run python examples/synthetic_flow_vla/run.py \
  --recipe recipes/synthetic/flow-vla-int8.yaml \
  --output-dir /tmp/piquant-v0.1-evidence
```

The output is a versioned `EvidenceRecord` containing model and calibration
fingerprints, module coverage, FP/reference-QDQ tensor metrics, action L1/L2,
direction cosine, gripper mismatch, artifact hashes, and the timing boundary.
The reference QDQ path is a portable numerical harness; it is not a ModelOpt,
TensorRT, hardware, or closed-loop result.

## Architecture

```text
ModelAdapter + CalibrationProvider + TaskLossProvider
                         |
                 pi.quant contracts
                         |
       ModelOpt backend / ORT capture / NumericalAnalyzer
                         |
                 EvidenceRecord + human promotion
```

Read [docs/architecture.md](docs/architecture.md), [docs/contracts.md](docs/contracts.md),
and [docs/modelopt-backend.md](docs/modelopt-backend.md) for the public
boundaries. Agent workflows live under `.agents/`; the root `AGENTS.md` is the
repository coding policy copied from the project-level policy source.

## Roadmap

The version sequence is intentionally serial. See [docs/roadmap.md](docs/roadmap.md).
