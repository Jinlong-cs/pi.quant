---
name: piquant-recipe-search
description: Use for multi-candidate quantization recipe exploration and sensitivity-guided next steps.
---

# pi.quant Recipe Search

## Use when

The objective is to compare explicit format, calibration, module, or precision
guard candidates against a matched FP baseline.

## Do not use when

Running one already-selected recipe; use `piquant-ptq` instead.

## Steps

1. Define the primary metric, accuracy boundary, calibration budget, and target.
2. Include FP16/BF16 and a near-lossless candidate before aggressive formats.
3. Change one major axis at a time and preserve the same evidence schema.
4. Use module coverage and intermediate/action metrics to choose the next candidate.
5. Promote only after comparable offline and deployment/closed-loop gates pass.

## Evidence boundary

Auto-generated candidates are candidates, not recommendations. Record recipe
hash, calibration fingerprint, selected/excluded modules, metrics, and rejection
reason. Hardware-specific tactics belong to a later platform feature PR.
