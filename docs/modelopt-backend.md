# ModelOpt Backend

The optional backend pins `nvidia-modelopt==0.45.0` and imports
`modelopt.torch.quantization` only when the backend is instantiated. It uses the
public `mtq.quantize(model, config, forward_loop=...)` API and follows the
ordered `quant_cfg` precedence documented by ModelOpt.

Before calibration, pi.quant performs its own module-name accounting and fails
on zero matches. ModelOpt quantizer coverage is recorded separately from
pi.quant module selection. The result explicitly reports `fake_quant` or
`real_quant`; a fake-quant model is never presented as a packed deployment
checkpoint.

The backend requires an adapter to expose `backend_model()` and
`forward_backend()` for the selected framework. v0.1's portable synthetic path
does not claim a GPU ModelOpt result when that optional environment is absent.
