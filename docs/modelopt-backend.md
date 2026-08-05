# ModelOpt Backend

The optional backend pins `nvidia-modelopt==0.45.0` and imports
`modelopt.torch.quantization` only when the backend is instantiated. It uses the
public `mtq.quantize(model, config, forward_loop=...)` API and follows the
ordered `quant_cfg` precedence documented by ModelOpt.

Before calibration, pi.quant resolves semantic logical selectors to exact
backend module paths and fails on zero matches. It first disables every
ModelOpt quantizer, then enables only the input and weight quantizers for the
resolved Linear modules. After calibration, the enabled set must equal exactly
two quantizers per selected module; missing or unexpectedly enabled Conv,
Embedding, norm, head, or other quantizers reject the candidate.

Evidence includes candidate/matched/excluded modules and parameters, resolved
backend paths, inserted and enabled quantizer names, num bits, axis, amax,
scale, and pre-quant scale summaries. The backend currently requires
`backend=modelopt`, `quant_format=int8`, and `representation=fake_quant`.
ModelOpt fake quantization is never presented as a packed or target engine.

The backend requires a `TorchQuantizableAdapter` to expose `backend_model()`,
`forward_backend()`, and `with_backend_model()`. The adapter is rebuilt from
the frozen checkpoint for every trial. If ModelOpt 0.45.0 or its explicit
runtime is unavailable, the call fails; it does not fall back to reference
QDQ, another precision, CPU, or synthetic data.
