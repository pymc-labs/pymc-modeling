"""Educational external likelihood boundary; use pm.Binomial for ordinary models.

Public interface: BinomialLogpOp()(eta, trials, observed) returns one float64
log probability per observation. Import binomial_logp_numba separately before
NUMBA compilation; importing this module does not require Numba.
"""

import numpy as np
import pytensor.tensor as pt
from pytensor.gradient import DisconnectedType, disconnected_type, grad_undefined
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.raise_op import CheckAndRaise
from scipy.special import gammaln


def _equal_length(length, trials_length, observed_length):
    return CheckAndRaise(ValueError, "BinomialLogpOp inputs must have equal length")(
        length, pt.eq(length, trials_length), pt.eq(length, observed_length)
    )


class BinomialLogpOp(Op):
    """Pointwise binomial log density parametrized by log odds.

    eta: float64 vector; trials and observed: signed/unsigned integer vectors.
    No scalar broadcasting, bool counts, or implicit float-to-count casts.
    Out-of-support counts return -inf. At infinite eta the density is the
    limiting point mass; NaN eta propagates for otherwise valid counts.
    Derivatives are defined only for finite eta and valid counts. Integer
    derivatives are deliberately undefined, not zero or disconnected.
    """

    __props__ = ()

    def make_node(self, eta, trials, observed):
        eta, trials, observed = map(pt.as_tensor_variable, (eta, trials, observed))
        if eta.ndim != 1 or eta.dtype != "float64":
            raise TypeError("eta must be a float64 vector")
        for name, variable in (("trials", trials), ("observed", observed)):
            if variable.ndim != 1 or np.dtype(variable.dtype).kind not in "iu":
                raise TypeError(f"{name} must be an integer vector (not bool)")
        static_lengths = {
            variable.type.shape[0]
            for variable in (eta, trials, observed)
            if variable.type.shape[0] is not None
        }
        if len(static_lengths) > 1:
            raise ValueError("BinomialLogpOp inputs must have equal length")
        return Apply(self, [eta, trials, observed], [eta.type()])

    def perform(self, node, inputs, output_storage):
        eta, trials, observed = inputs
        if eta.shape != trials.shape or eta.shape != observed.shape:
            raise ValueError("BinomialLogpOp inputs must have equal length")
        result = np.full(eta.shape, -np.inf, dtype="float64")
        n_unsigned = trials.astype("uint64")
        y_unsigned = observed.astype("uint64")
        valid = (trials >= 0) & (observed >= 0) & (y_unsigned <= n_unsigned)
        finite = valid & np.isfinite(eta)
        n = trials[finite].astype("float64")
        y = observed[finite].astype("float64")
        remainder = (n_unsigned[finite] - y_unsigned[finite]).astype("float64")
        z = eta[finite]
        result[finite] = (
            gammaln(n + 1.0) - gammaln(y + 1.0) - gammaln(remainder + 1.0)
            - y * np.logaddexp(0.0, -z)
            - remainder * np.logaddexp(0.0, z)
        )
        result[valid & np.isposinf(eta) & (y_unsigned == n_unsigned)] = 0.0
        result[valid & np.isneginf(eta) & (observed == 0)] = 0.0
        result[valid & np.isnan(eta)] = np.nan
        output_storage[0][0] = result

    def infer_shape(self, fgraph, node, shapes):
        # A shape-only query must not silently accept lengths that perform rejects.
        length = _equal_length(shapes[0][0], shapes[1][0], shapes[2][0])
        return [(length,)]

    def pullback(self, inputs, outputs, cotangents):
        eta, trials, observed = inputs
        (cotangent,) = cotangents
        if isinstance(cotangent.type, DisconnectedType):
            return [disconnected_type() for _ in inputs]
        length = _equal_length(eta.shape[0], trials.shape[0], observed.shape[0])
        eta = pt.specify_shape(eta, (length,))
        # Elementwise switches are eager: sanitize integer arithmetic before
        # masking invalid support, so Python-linker casts cannot overflow.
        n_unsigned = pt.maximum(trials, pt.zeros_like(trials)).astype("uint64")
        y_unsigned = pt.maximum(observed, pt.zeros_like(observed)).astype("uint64")
        remainder = (n_unsigned - pt.minimum(n_unsigned, y_unsigned)).astype("float64")
        y = observed.astype("float64")
        valid = (
            (trials >= 0) & (observed >= 0)
            & (y_unsigned <= n_unsigned)
            & ~pt.isnan(eta) & ~pt.isinf(eta)
        )
        # Algebraically y - n * sigmoid(eta); this form avoids tail cancellation.
        score = y * pt.sigmoid(-eta) - remainder * pt.sigmoid(eta)
        score = pt.switch(valid, score, np.float64(np.nan))
        return [
            cotangent * score,
            grad_undefined(self, 1, trials, "trials is discrete; no continuous derivative"),
            grad_undefined(self, 2, observed, "observed is discrete; no continuous derivative"),
        ]
