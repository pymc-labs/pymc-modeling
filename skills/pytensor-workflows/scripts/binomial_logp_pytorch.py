"""Opt-in native CPU PyTorch lowering for BinomialLogpOp.

All signed counts and uint8/uint16/uint32 widen losslessly to int64. uint64 is
rejected as a dtype, even when a particular input happens to fit in int64:
casting it unconditionally would change support decisions above 2**63 - 1.
The custom backward preserves the Op's undefined-score (NaN) boundary.
"""

import torch
from pytensor.link.pytorch.dispatch import pytorch_funcify

from binomial_logp_op import BinomialLogpOp


_COUNT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16, torch.uint32)


def _check_inputs(eta, trials, observed):
    if eta.ndim != 1 or eta.dtype != torch.float64:
        raise TypeError("eta must be a float64 vector")
    for name, value in (("trials", trials), ("observed", observed)):
        if value.dtype == torch.uint64:
            raise NotImplementedError("BinomialLogpOp PyTorch lowering does not support uint64 counts; no lossy int64 cast")
        if value.ndim != 1 or value.dtype not in _COUNT_DTYPES:
            raise TypeError(f"{name} must be a supported integer vector (not bool)")
    if any(value.device.type != "cpu" for value in (eta, trials, observed)):
        raise NotImplementedError("BinomialLogpOp PyTorch lowering is restricted to CPU tensors")
    if eta.shape != trials.shape or eta.shape != observed.shape:
        raise ValueError("BinomialLogpOp inputs must have equal length")


def _parts(trials, observed):
    n, y = trials.to(torch.int64), observed.to(torch.int64)
    valid = (n >= 0) & (y >= 0) & (y <= n)
    # Compute the integer remainder before float conversion, and never subtract
    # invalid signed extremes (e.g. int64 max minus int64 min).
    safe_n = torch.where(valid, n, 0)
    safe_y = torch.where(valid, y, 0)
    return valid, safe_n.to(torch.float64), safe_y.to(torch.float64), (safe_n - safe_y).to(torch.float64)


class _BinomialLogp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eta, trials, observed):
        ctx.save_for_backward(eta, trials, observed)
        valid, n, y, remainder = _parts(trials, observed)
        z = torch.where(torch.isfinite(eta), eta, 0.0)
        zero = torch.zeros_like(z)
        finite_logp = (
            torch.lgamma(n + 1.0) - torch.lgamma(y + 1.0) - torch.lgamma(remainder + 1.0)
            - y * torch.logaddexp(zero, -z) - remainder * torch.logaddexp(zero, z)
        )
        result = torch.where(valid & torch.isfinite(eta), finite_logp, -torch.inf)
        at_mass = (torch.isposinf(eta) & (remainder == 0)) | (torch.isneginf(eta) & (y == 0))
        result = torch.where(valid & at_mass, 0.0, result)
        return torch.where(valid & torch.isnan(eta), torch.nan, result)

    @staticmethod
    def backward(ctx, cotangent):
        eta, trials, observed = ctx.saved_tensors
        valid, _, y, remainder = _parts(trials, observed)
        score = y * torch.sigmoid(-eta) - remainder * torch.sigmoid(eta)
        score = torch.where(valid & torch.isfinite(eta), score, torch.nan)
        return cotangent * score, None, None


def binomial_logp(eta, trials, observed):
    """Native traceable CPU function; tensor arguments must already be typed."""
    _check_inputs(eta, trials, observed)
    return _BinomialLogp.apply(eta, trials, observed)


@pytorch_funcify.register(BinomialLogpOp)
def pytorch_funcify_binomial_logp(op, node=None, **kwargs):
    """Refuse unsupported graph dtypes before torch.compile sees a computation."""
    if node is not None and any(value.dtype == "uint64" for value in node.inputs[1:]):
        raise NotImplementedError("BinomialLogpOp PyTorch lowering does not support uint64 counts; no lossy int64 cast")
    return binomial_logp
