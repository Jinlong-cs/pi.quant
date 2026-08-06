---
name: piquant-temporal
description: Use for WAM temporal calibration, FastWAM captures, denoise-step diagnostics, and rollout divergence.
---

# pi.quant Temporal Diagnostics

## Use when

Run source/offline WAM or temporal VLA diagnostics with episode-aware
calibration, a frozen FP golden, and an explicit candidate model.

## Do not use when

Running TensorRT builds, target latency, server/client timing, or closed-loop
promotion. Those belong to the target compiler and promotion workflows.

## Prerequisites

Confirm the exact source model revision and ABI, episode-disjoint calibration,
diagnostic holdout, and promotion-reserved manifests, sequence window,
camera/language/proprio/history contract, normalization, denoise schedule,
flow-noise seed, timestep schedule, and external artifact root.

## Steps

1. Construct an explicit temporal adapter with a fresh model factory. Expose
   semantic module IDs, backend paths, capture kinds, and tensor axes.
2. Freeze FP captures with the same sequence order, seed, noise, timestep, and
   execution mode used by every candidate.
3. Run iterative inference through the audited source API. Inject a
   teacher-forced callback only when the source provides a comparable action
   boundary; otherwise fail fast and keep that mode pending.
4. Capture world/video latents only through an explicit source callback. Do not
   infer a latent from decoded images or replace missing values with synthetic
   tensors.
5. Compare activation, flow, action direction, gripper, stage, timestep,
   denoise-step, action-horizon, and optional rollout-horizon metrics using the
   shared analyzer.
6. Run broad, component-only, and rollback trials from fresh weights. Compare
   temporal-balanced calibration with a same-size static control when the
   manifests support it.
7. Write `EvidenceRecord` candidates and one `TemporalStudyRecord` with hashes,
   rankings, coverage, and the source/offline boundary.

## Verification gates

- Sequence manifests are episode-disjoint and fingerprints are reproducible.
- Capture axes, batch order, shape, finite values, and action ABI are explicit.
- Teacher-forced and iterative reports are never aggregated in one analyzer.
- A rollback ranking has a matched broad parent and nonzero retained coverage.
- Denoise-step counts are not confused with video prefill executions.
- A temporal source result is not described as target latency, closed-loop
  success, or accepted deployment.

## Evidence output

Keep raw captures, manifests, model assets, logs, and study JSON under the
external Task Contract artifact root. Commit only reusable contracts, adapter
logic, recipes, documentation, and small deterministic contract tests.

## Promotion boundary

This workflow may produce `measured`, `pending`, or `rejected`
source/offline-temporal evidence. Target compiler, standalone, server/client,
closed-loop, and human acceptance gates remain separate.
