"""Opt-in MLX CPU float64 BinomialLogpOp lowering.

Uses the installed PyTensor GammaLn lowering rather than a Python callback or
an independently copied approximation. Runtime evidence is required: float64
storage alone does not certify the precision of MLX transcendental kernels.
"""

from math import e

import mlx.core as mx
from pytensor.link.mlx.dispatch import mlx_funcify
from pytensor.scalar.math import GammaLn

from binomial_logp_op import BinomialLogpOp


_COUNT_DTYPES = (mx.int8, mx.int16, mx.int32, mx.int64, mx.uint8, mx.uint16, mx.uint32, mx.uint64)
_loggamma = mlx_funcify(GammaLn())


def _check_inputs(eta, trials, observed):
    if mx.default_device() != mx.cpu:
        raise NotImplementedError("BinomialLogpOp MLX lowering requires the CPU default device; no float32 substitution")
    if eta.ndim != 1 or eta.dtype != mx.float64:
        raise TypeError("eta must be a float64 vector")
    for name, value in (("trials", trials), ("observed", observed)):
        if value.ndim != 1 or value.dtype not in _COUNT_DTYPES:
            raise TypeError(f"{name} must be an integer vector (not bool)")
    if eta.shape != trials.shape or eta.shape != observed.shape:
        raise ValueError("BinomialLogpOp inputs must have equal length")


def _parts(trials, observed):
    nu, yu = trials.astype(mx.uint64), observed.astype(mx.uint64)
    valid = (trials >= 0) & (observed >= 0) & (yu <= nu)
    zero = mx.array(0, dtype=mx.uint64)
    safe_n, safe_y = mx.where(valid, nu, zero), mx.where(valid, yu, zero)
    return valid, safe_n.astype(mx.float64), safe_y.astype(mx.float64), (safe_n - safe_y).astype(mx.float64)


def _score(eta, trials, observed):
    valid, _, y, remainder = _parts(trials, observed)
    # MLX 0.32.2 exp/logaddexp use reduced precision even on float64 CPU
    # arrays. power with an explicitly float64 Euler constant preserves the
    # precision contract (also used by PyTensor's native MLX helper).
    # where, rather than abs, chooses the correct smooth branch derivative
    # at eta == 0; exponent inputs stay nonpositive in both tails.
    small = mx.power(mx.array(e, dtype=mx.float64), mx.where(eta >= 0, -eta, eta))
    positive = mx.where(eta >= 0, 1.0 / (1.0 + small), small / (1.0 + small))
    negative = mx.where(eta >= 0, small / (1.0 + small), 1.0 / (1.0 + small))
    return mx.where(valid & mx.isfinite(eta), y * negative - remainder * positive, mx.array(float("nan"), dtype=mx.float64))

@mx.custom_function
def _density(eta, trials, observed):
    valid, n, y, remainder = _parts(trials, observed)
    zero = mx.array(0.0, dtype=mx.float64)
    z = mx.where(mx.isfinite(eta), eta, zero)
    correction = mx.log1p(mx.power(mx.array(e, dtype=mx.float64), -mx.abs(z)))
    positive = mx.maximum(z, zero) + correction
    negative = mx.maximum(-z, zero) + correction
    # C(n, 0) = C(n, n) = 1 exactly, including n == 0. Do not inherit the
    # Lanczos approximation's nonzero logGamma(1) residual at these endpoints.
    log_coefficient = mx.where(
        (y == 0) | (remainder == 0),
        zero,
        _loggamma(n + 1.0) - _loggamma(y + 1.0) - _loggamma(remainder + 1.0),
    )
    finite_logp = log_coefficient - y * negative - remainder * positive
    result = mx.where(valid & mx.isfinite(eta), finite_logp, mx.array(-float("inf"), dtype=mx.float64))
    at_mass = (mx.isposinf(eta) & (remainder == 0)) | (mx.isneginf(eta) & (y == 0))
    result = mx.where(valid & at_mass, zero, result)
    return mx.where(valid & mx.isnan(eta), mx.array(float("nan"), dtype=mx.float64), result)


@_density.vjp
def _density_vjp(primals, cotangent, output):
    eta, trials, observed = primals
    # Counts have no continuous score: never invent a finite zero derivative.
    undefined = mx.full(eta.shape, float("nan"), dtype=mx.float64)
    return cotangent * _score(eta, trials, observed), undefined, undefined


def binomial_logp(eta, trials, observed):
    """Native CPU density; continuous reverse AD only for the eta input."""
    _check_inputs(eta, trials, observed)
    return _density(eta, trials, observed)


@mlx_funcify.register(BinomialLogpOp)
def mlx_funcify_binomial_logp(op, node=None, **kwargs):
    if mx.default_device() != mx.cpu:
        raise NotImplementedError("BinomialLogpOp MLX lowering requires the CPU default device; no float32 substitution")
    return binomial_logp
