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

Confirm exact model/checkpoint/normalization identity, `ActionSchema`, an
episode-disjoint calibration manifest, `OptimizationPlan`, backend version,
external artifact root, and a fresh-model factory. The plan dataset revision,
sample count, stages, seed, and capture IDs must match the supplied manifests
and golden.

## Steps

1. Freeze source FP captures with identical observation, language, state/history,
   seed, flow noise, timestep, horizon, denoise schedule, and postprocess.
2. Resolve semantic module IDs to backend paths. Record candidate/matched/
   excluded modules and parameters; fail on zero matches.
3. Reload immutable weights, run the explicitly selected backend, and verify the
   exact enabled quantizer set after calibration.
4. Replay the diagnostic manifest and stream activation, flow, action, horizon,
   stage, timestep, direction, and gripper metrics.
5. Write one immutable `EvidenceRecord` with command, versions, coverage,
   artifact hashes, and source/offline timing boundary.

## Gates and outputs

Required outputs are source repeatability, nonzero coverage, finite/shape gates,
paired diagnostics, backend/quantizer metadata, artifact hashes, command, and
measured/pending/rejected status. Fake quant is not packed quant. The Agent may
produce machine evidence but cannot self-approve promotion.
