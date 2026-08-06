---
name: piquant-target-compiler
description: Build target compiler evidence without mixing it with quantization, runtime, closed-loop or acceptance claims.
---

# piquant Target Compiler

## Use When

- Adding or reviewing ONNX/TensorRT compiler evidence for pi.quant.
- Creating AGX Orin or RTX 5090 build, layer inspection, benchmark or pi.cpp
  handoff records.
- Separating capability, graph, build, stage, runtime, closed-loop and accepted
  evidence lanes.

## Do Not Use When

- Running ModelOpt/PTQ/QAT source-level quantization.
- Writing pi.cpp runtime code or packaging accepted model directories.
- Launching high-cost closed-loop evaluation without a separate promotion gate.

## Preconditions

- Read `AGENTS.md`, `.agents/README.md`, `docs/target-compiler.md` and the
  relevant Task Contract.
- Re-check branch, origin, worktree cleanliness, GPU owner, disk, CUDA,
  TensorRT, active processes and accepted artifacts.
- Keep ONNX files, engines, logs, layer JSON, profiler traces and benchmark
  outputs in the external artifact root.

## Steps

1. Validate or write a `CompilationPlan` template through
   `piquant validate-compilation-plan`.
2. Probe target capability read-only and record unsupported or pending states
   with reason codes.
3. Render the target `trtexec` command before running it.
4. Build only in the authorized target environment and write
   `CompilerEvidenceRecord`.
5. Inspect ONNX and TensorRT layer evidence separately from latency.
6. Record `BenchmarkProtocol` and `StageTimingReport` for each timing boundary.
7. Emit `DeploymentCandidateManifest` for pi.cpp handoff; do not mark accepted.

## Validation Gates

- Default `import piquant` does not load TensorRT, ONNX, ORT, Torch, ModelOpt,
  CUDA or simulator modules.
- `unsupported` and `pending` records include explicit reason codes.
- Engine-stage, standalone, server/client and closed-loop timings are not
  merged.
- Hardware claims cite target-local artifact hashes and commands.

## Promotion Boundary

Agents may write `pending`, `measured`, `rejected` and `unsupported` machine
evidence. Only a human owner can promote a deployment manifest to `accepted`.
