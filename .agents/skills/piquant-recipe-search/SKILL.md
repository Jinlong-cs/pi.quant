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

1. Define the primary metric, quality boundary, data split, search budget, and
   current evidence level. Source-only studies have no target latency cost.
2. Freeze FP, broad quant, component-only, and manual selective controls under
   one model/calibration/golden identity.
3. From broad, restore one component at a time and compute
   `(broad_error - rollback_error) / max(broad_error, epsilon)` while retaining
   module and parameter coverage.
4. Expand only the top one or two recovering components: contiguous block
   groups, blocks inside the best group, then projections inside the best block.
5. Compare stage-balanced/temporal calibration with a same-size random/static
   control. Re-run the same component rollbacks under both calibrations and
   record rank changes instead of inferring sensitivity from broad drift alone.
   Keep diagnostic and promotion episodes disjoint.
6. Preserve ties and uncertainty. Do not turn component-only error, parameter
   count, FLOPs, or a source proxy into a target winner.
7. Promote only after the separately required compiler, latency, server/client,
   and closed-loop gates pass.

## Evidence boundary

Auto-generated candidates are candidates, not recommendations. Record immutable
parent/mutation lineage, recipe hash, calibration fingerprint, resolved
selectors, exact quantizers, metrics, retained coverage, and rejection reason.
Hardware-specific tactics and measured costs belong to the target compiler
feature, not a source sensitivity study.
