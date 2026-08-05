# Roadmap

Each version is a separate serial feature PR.

| Version | Scope | Acceptance focus |
| --- | --- | --- |
| v0.1 | Contracts, ModelOpt/ORT boundaries, synthetic vertical workflow | Offline tests and structured evidence |
| v0.2 | Pi0.5 adapter, real calibration, intermediate/action sensitivity | Matched FP golden and offline candidate analysis |
| v0.3 | FastWAM/WAM temporal calibration and rollout-aware metrics | Temporal divergence and task evaluation |
| v0.4 | AGX Ampere INT8 and RTX 5090 Blackwell FP8/NVFP4 compiler evidence | Target-local parity and timing gates |
| v0.5 | Mixed-precision search, candidate ranking, LIBERO promotion | Comparable deployment and closed-loop evidence |
| v1.0 | Stable plugin API and production artifact lineage | Cross-model and hardware promotion gates |

No roadmap item is accepted by documentation alone. The target platform,
runtime, timing boundary, and closed-loop protocol must be measured in the
feature that introduces them.
