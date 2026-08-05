# Pi0.5 LIBERO sensitivity

This example explicitly wires the real OpenPI PyTorch adapter, LIBERO manifest
provider, ModelOpt backend, streaming analyzer, and JSON evidence store. It is
a thin entry to reusable `piquant` APIs, not a second experiment framework.

Run it inside an identity-matched OpenPI environment with the `pi05` and
`modelopt` extras installed. Keep the identity JSON with the dataset,
checkpoint, captures, and evidence under the external artifact root; do not
commit experiment identities. Set `OPENPI_DATA_HOME` to the path recorded in
that external JSON.

```bash
export OPENPI_DATA_HOME=/path/to/openpi_cache
export ARTIFACT_ROOT=/path/to/external/piquant-pi05-study

uv run python examples/pi05_libero/run.py manifests \
  --identity "$ARTIFACT_ROOT/identity.json" \
  --dataset-root /path/to/physical-intelligence/libero \
  --output-dir "$ARTIFACT_ROOT/manifests"

uv run python examples/pi05_libero/run.py golden \
  --identity "$ARTIFACT_ROOT/identity.json" \
  --dataset-root /path/to/physical-intelligence/libero \
  --holdout-manifest "$ARTIFACT_ROOT/manifests/diagnostic_holdout.json" \
  --artifact-root "$ARTIFACT_ROOT/study" \
  --source-device "NVIDIA GPU identity from live inspection"

uv run python examples/pi05_libero/run.py trial \
  --identity "$ARTIFACT_ROOT/identity.json" \
  --dataset-root /path/to/physical-intelligence/libero \
  --plan recipes/pi05/libero-int8-broad.yaml \
  --calibration-manifest "$ARTIFACT_ROOT/manifests/calibration.json" \
  --holdout-manifest "$ARTIFACT_ROOT/manifests/diagnostic_holdout.json" \
  --golden-manifest "$ARTIFACT_ROOT/study/golden/manifest.json" \
  --artifact-root "$ARTIFACT_ROOT/study" \
  --trial-id int8-broad \
  --kind broad \
  --source-device "NVIDIA GPU identity from live inspection"

uv run python examples/pi05_libero/run.py study \
  --identity "$ARTIFACT_ROOT/identity.json" \
  --dataset-root /path/to/physical-intelligence/libero \
  --holdout-manifest "$ARTIFACT_ROOT/manifests/diagnostic_holdout.json" \
  --golden-manifest "$ARTIFACT_ROOT/study/golden/manifest.json" \
  --trial-index "$ARTIFACT_ROOT/trial-index.json" \
  --artifact-root "$ARTIFACT_ROOT/study" \
  --study-id pi05-libero-sensitivity-v1 \
  --source-device "NVIDIA GPU identity from live inspection"

uv run piquant validate-study "$ARTIFACT_ROOT/study/study.json"
```

Use `libero-fp-control.yaml` with `--kind fp_control` for deterministic golden
replay. Component-only and rollback trials are separate resolved plans with
semantic include/exclude patterns. Every trial starts from a fresh adapter and
writes a new candidate directory; existing evidence fails fast.

`trial-index.json` is an ordered JSON list of completed trial IDs. Finalization
hashes every candidate and derives rollback recovery plus calibration ablation;
it does not rerun or promote candidates.

The result is source/offline fake-quant evidence. It does not establish ONNX
parity, TensorRT precision, hardware latency, LIBERO closed-loop success, or an
accepted deployment.
