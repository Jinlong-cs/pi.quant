# Roadmap

Each version is a separate serial feature PR.

| Version | Scope | Acceptance focus |
| --- | --- | --- |
| v0.1 | Contracts, ModelOpt/ORT boundaries, synthetic vertical workflow | Offline tests and structured evidence |
| v0.2 | Real Pi0.5/LIBERO manifests, FP golden, semantic captures, hierarchical sensitivity | Source parity, deterministic replay, coverage, and selective recovery |
| v0.3 | FastWAM/WAM temporal calibration and rollout-aware metrics | Temporal divergence and task evaluation |
| v0.4 | AGX Ampere INT8 and RTX 5090 Blackwell FP8/NVFP4 compiler evidence | Target-local parity and timing gates |
| v0.5 | Mixed-precision search, candidate ranking, LIBERO promotion | Comparable deployment and closed-loop evidence |
| v1.0 | Stable plugin API and production artifact lineage | Cross-model and hardware promotion gates |

No roadmap item is accepted by documentation alone. The target platform,
runtime, timing boundary, and closed-loop protocol must be measured in the
feature that introduces them.

Versions are serial feature PRs. v0.3 starts only after v0.2 merges; v0.4 starts
only after both real model studies exist; v0.5 consumes measured source
sensitivity and target cost rather than parameter/FLOP guesses. QAT, pruning,
distillation, step reduction, and custom kernels remain independent future
features.
