---
name: piquant-numerical-debug
description: Use for intermediate activation, action, ORT, and quantization-drift diagnosis.
---

# pi.quant Numerical Debugging

## Use when

An FP/candidate pair has shape, finite, cosine, SQNR, action, or task-loss drift.

## Steps

1. Re-run identical observation, language, history, seed, noise, and timestep.
2. Compare the earliest divergent capture point first.
3. Separate representation drift, decision drift, and temporal amplification.
4. Check outliers, calibration stages, gripper thresholds, and missing tensors.
5. Record the diagnosis as evidence; do not silently widen tolerances.

## ORT boundary

`OrtDebugCapture` augments a temporary graph copy and feeds the shared analyzer.
It must not modify the source ONNX or be treated as a quantization backend.
