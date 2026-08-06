# FastWAM temporal diagnostics

This example documents the explicit wiring for a real FastWAM/WAM source
environment. The adapter is lazy: pi.quant does not import FastWAM or Torch
until the caller constructs `FastWAMSourceAdapter` with a model factory.

The audited source ABI is a single sample with 33 video frames, a 32-step,
7-dimensional action horizon, and ten inference steps. The adapter validates
that action output contract and records source module paths separately from
semantic IDs. It does not encode the historical five-node precision guard.

```python
from piquant.adapters import FastWAMSourceAdapter, FastWAMSourceConfig
from piquant.integrations import FastWAMCaptureRunner, FastWAMInferenceContract

config = FastWAMSourceConfig(
    revision="<external-source-revision>",
    model_factory=load_fastwam_model,
    inference=FastWAMInferenceContract(),
)
adapter = FastWAMSourceAdapter(config, runner=FastWAMCaptureRunner())
```

The source `infer_action` path uses MoT cache routines that manually invoke
the internals of video blocks, so a normal PyTorch forward hook cannot observe
video block outputs. To request the video block captures declared by the
adapter, inject a `FastWAMCaptureProvider` that runs the exact source path and
returns arrays keyed by logical capture ID. The provider is an explicit source
instrumentation boundary; it must not synthesize captures or change the model
ABI. Without it, an attempted video-block capture fails fast.

Use `recipes/fastwam/temporal-fp-control.yaml` for the FP control and
`recipes/fastwam/temporal-int8-broad.yaml` for an explicitly selected ModelOpt
INT8 fake-quant candidate. A real run must inject a temporal calibration
provider with episode-disjoint calibration, diagnostic holdout, and
promotion-reserved manifests, then call `TemporalSensitivityRunner` from
`piquant.temporal_study`. The provider must bind each sequence to its observation window,
language, proprio/history, action target, flow noise, timestep schedule, and
seed.

Iterative inference uses the source `infer_action` API. Teacher-forced
diagnostics require an explicit `teacher_forced_runner` callback because the
audited source training path does not expose a comparable final action. World
latent capture likewise requires an explicit adapter/integration callback;
missing callbacks fail fast and must remain pending rather than being
replaced with fabricated tensors.

Raw captures, manifests, model assets, logs, and evidence belong under the
external Task Contract artifact root. This repository example contains no
model identity, machine path, experiment record, hardware result, closed-loop
success claim, or accepted deployment decision.
