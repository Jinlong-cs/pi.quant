# Contracts

The four stable records are `ModelSpec`, `TargetFingerprint`,
`OptimizationPlan`, and `EvidenceRecord`. They are Pydantic models with
`extra="forbid"` so undocumented recipe fields fail at the boundary.

`OptimizationPlan` is the only interpretation of a YAML recipe. It fixes the
backend, format, fake/real representation, module patterns, calibration
identity, seed, capture points, and timing boundary.

`EvidenceRecord` links the plan to the model, target environment, calibration
fingerprint, module coverage, comparisons, artifact hashes, command, and
status. `measured`, `accepted`, and `pending` are distinct states. The Agent
can write measured evidence but cannot turn it into accepted evidence.
