---
name: piquant-ptq
description: Use for one explicit VLA post-training quantization plan and its evidence record.
---

# pi.quant PTQ

## Use when

Run one selected recipe against a known model adapter and calibration provider.

## Do not use when

Searching multiple recipes, debugging numerical drift, or promoting a result.
Use the corresponding piquant skill for those tasks.

## Prerequisites

Confirm `ModelSpec`, `OptimizationPlan`, calibration fingerprint, backend
availability, module match counts, and external artifact root before execution.

## Steps

1. Freeze the FP golden inputs, seed, history, action horizon, and capture points.
2. Inspect candidate modules and fail if the include patterns match zero.
3. Run the selected backend with an explicit fake/real representation.
4. Compare intermediate tensors and action outputs with the same inputs.
5. Write `EvidenceRecord`; keep target hardware and closed-loop claims separate.

## Gates and outputs

Required outputs are module coverage, tensor/action comparison, backend/version
metadata, artifact hashes, command, timing boundary, and measured/pending status.
The Agent may produce machine evidence but cannot self-approve promotion.
