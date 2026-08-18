---
name: piquant-promotion
description: Use when deciding whether measured optimization evidence can become a deployment candidate.
---

# pi.quant Promotion

## Required evidence

Verify artifact identity, plan/calibration/target fingerprints, module coverage,
FP parity, numerical/action metrics, and the correct timing boundary. Add
standalone, server/client, and closed-loop gates only when those systems are in
scope.

For a source sensitivity study, validate the golden and every candidate hash,
episode-disjoint manifests, fresh-model execution, nonzero quantizer coverage,
recovery ranking, and calibration ablation. This can become `measured`
source/offline evidence only.

## Rules

- Measured is not accepted.
- Offline action error is not closed-loop success.
- A smoke test is not a full benchmark.
- A missing hardware gate is pending, never passed.
- Parameter/FLOP coverage is not latency.
- H200 source fake quant is not AGX/RTX TensorRT evidence.
- Human acceptance is required for final promotion.
- A `PromotionPlan` must use the same search-plan hash, model, target, and
  benchmark as the measured target candidate.
- The candidate must be non-dominated on the referenced target front, and its
  baseline must be the measured FP target control from the same plan. Record
  both recipe hashes; a baseline ID string alone is insufficient lineage.
- Complete gates in order. A rejected/unsupported gate is terminal; later
  evidence cannot resurrect or skip it.
- The owning search budget must reserve Gate40 and full400 before a promotion
  plan is created. Measured or failed execution of either high-cost gate
  requires an explicit approval artifact. `piquant promote` and
  `validate-promotion-plan` never execute a gate.

## Output

Update the evidence record and handoff with the exact next gate, residual risk,
artifact pointer, and promotion decision owner.

The normal ladder is source repeatability -> source quant diagnostics ->
source/export parity -> target build/parity -> standalone timing -> real
server/client smoke -> matched task gate -> full protocol -> human acceptance.

Keep calibration, sensitivity, search-validation, and promotion-reserved
episodes disjoint. A target Pareto candidate is a promotion input, not a
recommendation or accepted deployment.
