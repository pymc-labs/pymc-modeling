"""Opt-in, nopython NUMBA lowering for the educational BinomialLogpOp.

Import this module before compiling a graph with mode="NUMBA". The dispatch
registration is process-local. It does not install or enable any other backend.
"""

import math

import numba
import numpy as np
from pytensor.link.numba.dispatch import numba_funcify

from binomial_logp_op import BinomialLogpOp


@numba.njit(fastmath=False)
def _binomial_logp(eta, trials, observed):
    if len(eta) != len(trials) or len(eta) != len(observed):
        raise ValueError("BinomialLogpOp inputs must have equal length")
    result = np.empty(len(eta), dtype=np.float64)
    for i in range(len(eta)):
        z = eta[i]
        n_int = trials[i]
        y_int = observed[i]
        if n_int < 0 or y_int < 0 or np.uint64(y_int) > np.uint64(n_int):
            result[i] = -np.inf
        elif math.isnan(z):
            result[i] = np.nan
        elif math.isinf(z):
            at_mass = (z > 0 and np.uint64(y_int) == np.uint64(n_int)) or (z < 0 and y_int == 0)
            result[i] = 0.0 if at_mass else -np.inf
        else:
            n = float(n_int)
            y = float(y_int)
            remainder = float(np.uint64(n_int) - np.uint64(y_int))
            log_choose = math.lgamma(n + 1.0) - math.lgamma(y + 1.0) - math.lgamma(remainder + 1.0)
            correction = math.log1p(math.exp(-abs(z)))
            softplus_positive = max(z, 0.0) + correction
            softplus_negative = max(-z, 0.0) + correction
            result[i] = log_choose - y * softplus_negative - remainder * softplus_positive
    return result


@numba_funcify.register(BinomialLogpOp)
def numba_funcify_binomial_logp(op, node=None, **kwargs):
    # No objmode/perform callback, fastmath, or custom persistent-cache key.
    return _binomial_logp
