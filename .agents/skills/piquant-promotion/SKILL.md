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

## Rules

- Measured is not accepted.
- Offline action error is not closed-loop success.
- A smoke test is not a full benchmark.
- A missing hardware gate is pending, never passed.
- Human acceptance is required for final promotion.

## Output

Update the evidence record and handoff with the exact next gate, residual risk,
artifact pointer, and promotion decision owner.
