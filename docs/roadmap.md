# Roadmap

Each version is a separate serial feature PR.

| Version | Scope | Acceptance focus |
| --- | --- | --- |
| v0.1 | Contracts, ModelOpt/ORT boundaries, synthetic vertical workflow | Offline tests and structured evidence |
| v0.2 | Real Pi0.5/LIBERO manifests, FP golden, semantic captures, hierarchical sensitivity | Source parity, deterministic replay, coverage, and selective recovery |
| v0.3 | FastWAM/WAM temporal calibration and rollout-aware metrics | Temporal divergence, calibration controls, and task-evaluator boundary |
| v0.4 | Target compiler contracts, ONNX/TensorRT inspection, deployment handoff, AGX/RTX 5090 capability gates | Target-local graph/build/timing evidence with unsupported states |
| v0.5 | Budgeted mixed-precision search, target-local Pareto ranking, staged LIBERO promotion planning | Split isolation, deterministic search, explicit rejection, and pending-first promotion boundaries |
| v1.0 | Stable public control-plane API, production artifact lineage, and release engineering | Backward compatibility, lineage validation, dependency isolation, package installation, and human release review |

No roadmap item is accepted by documentation alone. The target platform,
runtime, timing boundary, and closed-loop protocol must be measured in the
feature that introduces them.

Versions are serial feature PRs. v0.3 starts only after v0.2 merges; v0.4 starts
only after both real model studies exist; v0.5 consumes measured source
sensitivity and target cost rather than parameter/FLOP guesses. v0.4 can
publish target compiler plumbing and capability probes while AGX/RTX builds
remain `pending`; it cannot call those probes latency or deployment evidence.
QAT, pruning, distillation, step reduction, and custom kernels remain
independent future features.

v0.5 keeps one search problem per model, target, ABI, and benchmark protocol.
It starts with FP, broad-quant, and manual-selective controls, generates a
bounded sensitivity/cost-guided beam, filters by pre-registered source gates,
and compiles only controls plus source-Pareto survivors. The target-local front
retains uncertainty ties instead of selecting a winner through a hidden score.

Promotion remains a separate ladder. Mechanical, offline, target latency, and
server/client evidence do not imply Gate40 or full400 success. High-cost gates
require an external approval record, and no search or promotion API may assign
human acceptance.

The v0.5 code release is accepted independently from a deployment candidate.
A source or target failure remains preserved evidence and does not block the
reusable search/promotion control plane from merging. Claims of a complete
deployment-promotion vertical slice still require the real target,
server/client, Gate40, full400, and human candidate gates.

## v1.0 Stable Release

v1.0 stabilizes the existing control plane rather than adding another model or
hardware experiment:

- keep schema-version-1 v0.1-v0.5 readers and CLI commands compatible;
- freeze the public import, dependency-injection, evidence-status, and
  promotion-authority boundaries;
- add a deterministic artifact-lineage graph from source identity through
  promotion evidence;
- verify lineage graph structure, stage completeness, terminal status,
  canonical hash, and optional artifact hashes without optional runtimes;
- align package metadata, `--version`, declared Python support, CI, wheel/sdist
  build, and clean-install smoke;
- publish a changelog, support policy, release checklist, and explicit
  merge/tag/package-publication authority boundary.

Real-model, target-hardware, server/client, Gate40, full400, and deployment
candidate acceptance remain independent Task Contracts. They are not inferred
from a successful v1.0 library build.
