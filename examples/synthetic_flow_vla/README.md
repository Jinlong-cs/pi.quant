# Synthetic flow-VLA example

This example is a deterministic SDK check, not a model benchmark. It creates a
small vision/language/history/action graph, runs a portable reference QDQ
candidate, compares named intermediate tensors and action outputs, and writes
an `EvidenceRecord` plus hashed NumPy captures to an external directory.

```bash
uv run python examples/synthetic_flow_vla/run.py \
  --recipe recipes/synthetic/flow-vla-int8.yaml \
  --output-dir /tmp/piquant-v0.1-evidence
```

The optional ModelOpt backend and ORT debug capture have independent dependency
and evidence gates. This example must not be quoted as Pi0.5, FastWAM, hardware,
TensorRT, or closed-loop evidence.
