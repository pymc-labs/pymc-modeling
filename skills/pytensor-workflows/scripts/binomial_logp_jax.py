"""Opt-in JAX lowering for BinomialLogpOp; no Python callback or NumPy fallback.

Import before constructing a JAX-linked function. Require JAX x64 *before*
converting inputs. Count comparisons and subtraction stay integer-exact; only
log-gamma evaluation and the continuous score use float64. CPU is the declared
execution target; this module does not establish accelerator compatibility.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from pytensor.link.jax.dispatch import jax_funcify

from binomial_logp_op import BinomialLogpOp


def _check_inputs(eta, trials, observed):
    if not jax.config.x64_enabled:
        raise TypeError("BinomialLogpOp JAX lowering requires jax_enable_x64=True before input conversion")
    if eta.ndim != 1 or eta.dtype != jnp.float64:
        raise TypeError("eta must be a float64 vector")
    for name, value in (("trials", trials), ("observed", observed)):
        if value.ndim != 1 or np.dtype(value.dtype).kind not in "iu":
            raise TypeError(f"{name} must be an integer vector (not bool)")
    # JAX array shapes are static during tracing. This catches unequal shapes
    # before elementwise broadcasting, including after a different-shape retrace.
    if eta.shape != trials.shape or eta.shape != observed.shape:
        raise ValueError("BinomialLogpOp inputs must have equal length")


def _parts(eta, trials, observed):
    _check_inputs(eta, trials, observed)
    nu = trials.astype(jnp.uint64)
    yu = observed.astype(jnp.uint64)
    valid = (trials >= 0) & (observed >= 0) & (yu <= nu)
    # Eager branches must never calculate negative-count gamma functions or
    # subtract signed integers across their representable range.
    safe_n = jnp.where(valid, nu, jnp.uint64(0))
    safe_y = jnp.where(valid, yu, jnp.uint64(0))
    remainder = (safe_n - safe_y).astype(jnp.float64)
    return valid, safe_n.astype(jnp.float64), safe_y.astype(jnp.float64), remainder


@jax.custom_jvp
def binomial_logp(eta, trials, observed):
    """Native pointwise density with the same finite/nonfinite support contract."""
    valid, n, y, remainder = _parts(eta, trials, observed)
    z = jnp.where(jnp.isfinite(eta), eta, jnp.float64(0))
    finite_logp = (
        gammaln(n + jnp.float64(1))
        - gammaln(y + jnp.float64(1))
        - gammaln(remainder + jnp.float64(1))
        - y * jnp.logaddexp(jnp.float64(0), -z)
        - remainder * jnp.logaddexp(jnp.float64(0), z)
    )
    result = jnp.where(valid & jnp.isfinite(eta), finite_logp, -jnp.inf)
    at_mass = (jnp.isposinf(eta) & (remainder == 0)) | (jnp.isneginf(eta) & (y == 0))
    result = jnp.where(valid & at_mass, jnp.float64(0), result)
    return jnp.where(valid & jnp.isnan(eta), jnp.nan, result)


@binomial_logp.defjvp
def _binomial_logp_jvp(primals, tangents):
    eta, trials, observed = primals
    eta_tangent, _, _ = tangents  # Integer tangents are JAX float0, not continuous scores.
    valid, _, y, remainder = _parts(eta, trials, observed)
    score = y * jax.nn.sigmoid(-eta) - remainder * jax.nn.sigmoid(eta)
    score = jnp.where(valid & jnp.isfinite(eta), score, jnp.nan)
    return binomial_logp(*primals), eta_tangent * score


@jax_funcify.register(BinomialLogpOp)
def jax_funcify_binomial_logp(op, node=None, **kwargs):
    """Register the Op class; return a traceable function, not a compiled graph."""
    if not jax.config.x64_enabled:
        raise TypeError("BinomialLogpOp JAX lowering requires jax_enable_x64=True before input conversion")
    return binomial_logp
