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

1. Define one `SearchPlan` for one model, target, ABI, and benchmark protocol.
   Freeze supported precision, numeric hard constraints, source/compiler and
   promotion budgets, and the current evidence level.
2. Audit episode and seed isolation across calibration, sensitivity,
   search-validation, and promotion-reserved manifests. Freeze exactly one FP,
   broad-quant, and manual-selective control under the same identity.
3. From broad, restore one component at a time and compute
   `(broad_error - rollback_error) / max(broad_error, epsilon)` while retaining
   module and parameter coverage.
4. Expand only the top one or two recovering components: contiguous block
   groups, blocks inside the best group, then projections inside the best block.
5. Compare stage-balanced/temporal calibration with a same-size random/static
   control. Re-run the same component rollbacks under both calibrations and
   record rank changes instead of inferring sensitivity from broad drift alone.
   Keep diagnostic and promotion episodes disjoint.
6. Resolve the search-plan hash before execution. Every source/target candidate
   must carry that hash, immutable recipe lineage, split fingerprints, shape and
   finite gates, metrics, uncertainty, and a terminal reason when rejected.
7. Preserve all three controls for target measurement, then fill the remaining
   compiler budget with source-Pareto search candidates in the deterministic
   sensitivity-guided generation order. Rebuild the Pareto front from target-local
   build, parity, timing, memory, coverage, and evidence level.
   FP is an unconstrained comparator; broad/manual source rejection remains
   recorded but does not erase those explicit target controls. Generated search
   candidates must pass the source gate.
8. Preserve ties and uncertainty. Do not turn component-only error, parameter
   count, FLOPs, or a source proxy into a target winner.
9. Promote only after the separately required compiler, latency, server/client,
   and closed-loop gates pass.

## Evidence boundary

Auto-generated candidates are candidates, not recommendations. Record immutable
parent/mutation lineage, recipe hash, calibration fingerprint, resolved
selectors, exact quantizers, metrics, retained coverage, and rejection reason.
Hardware-specific tactics and measured costs belong to the target compiler
feature, not a source sensitivity study.

`piquant search` generates recipes only; evaluator and compiler callbacks are
explicit Python dependencies. Resume only a terminal candidate whose recipe,
search-plan hash, model, target, and all split fingerprints still match.
`piquant rank` must load that same plan; callers cannot replace its objectives
at ranking time.
