---
name: piquant-numerical-debug
description: Use for intermediate activation, action, ORT, and quantization-drift diagnosis.
---

# pi.quant Numerical Debugging

## Use when

An FP/candidate pair has shape, finite, cosine, SQNR, action, or task-loss drift.

## Steps

1. Re-run identical observation, language, history, seed, noise, and timestep.
2. Verify candidate coverage and exact enabled quantizers before interpreting
   model error; reject unintended quantizers as a configuration failure.
3. Compare the earliest divergent logical capture first. Bucket paired metrics
   by component, block, stage, timestep, and horizon.
4. Separate representation drift, decision drift, and temporal amplification.
5. Start with component-only and broad controls. Rank component rollback
   recovery relative to broad, then split only the top components into
   block-groups, blocks, and finally projections/layers.
6. Check calibration distribution, outliers, gripper postprocess/threshold,
   capture alignment, and subgroup uncertainty before changing precision.
7. Record the diagnosis and retained quantization coverage; do not silently
   widen tolerances or call parameter coverage latency.

## Gates

- FP replay must be deterministic within a declared tolerance.
- Every compared capture must have matched sample order, shape, and finite data.
- A rollback claim needs a matched broad parent and nonzero retained coverage.
- One low-error component-only trial does not prove safety under broad
  quantization; interactions require rollback evidence.

## ORT boundary

`OrtDebugCapture` augments a temporary graph copy and feeds the shared analyzer.
It must not modify the source ONNX or be treated as a quantization backend.

## Evidence output

Write candidate records plus a `SensitivityStudyRecord` that indexes hashes,
recovery ranking, calibration ablation, and the source/offline boundary. Raw
captures stay in the external artifact root.
