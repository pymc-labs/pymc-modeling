"""Numerical and input-contract regressions for the educational Binomial Op."""

from importlib import import_module
from importlib.util import find_spec

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import expit
from scipy.stats import binom

from binomial_logp_op import BinomialLogpOp


@pytest.fixture(params=["FAST_COMPILE", pytest.param("NUMBA", marks=pytest.mark.numba)])
def value_mode(request):
    if request.param == "NUMBA":
        # Skip an absent optional backend, not a broken installed backend.
        if find_spec("numba") is None:
            pytest.importorskip("numba")
        import_module("binomial_logp_numba")
    return request.param


def compile_logp(mode, trials_dtype="int64", observed_dtype="int64", static_eta=False):
    eta = pt.TensorType("float64", shape=(3 if static_eta else None,))("eta")
    trials = pt.vector("trials", dtype=trials_dtype)
    observed = pt.vector("observed", dtype=observed_dtype)
    return pytensor.function(
        [eta, trials, observed], BinomialLogpOp()(eta, trials, observed), mode=mode
    )


def test_compiled_values_match_scipy(value_mode):
    eta = np.array([-8.0, -2.0, -0.1, 0.0, 1.5, 7.0])
    trials = np.array([0, 4, 11, 6, 20, 15], dtype="int64")
    observed = np.array([0, 0, 6, 3, 17, 15], dtype="int64")
    actual = compile_logp(value_mode)(eta, trials, observed)
    assert actual.shape == eta.shape
    assert actual.dtype == np.dtype("float64")
    np.testing.assert_allclose(
        actual, binom.logpmf(observed, trials, expit(eta)), rtol=1e-11, atol=1e-12
    )


def test_weighted_symbolic_gradient_and_curvature():
    eta = pt.dvector("eta")
    weights = pt.dvector("weights")
    trials = pt.lvector("trials")
    observed = pt.lvector("observed")
    objective = (weights * BinomialLogpOp()(eta, trials, observed)).sum()
    score = pt.grad(objective, eta)
    curvature = pt.grad(score.sum(), eta)
    evaluate = pytensor.function(
        [eta, trials, observed, weights], [score, curvature], mode="FAST_COMPILE"
    )
    z = np.array([-4.0, -0.5, 0.0, 1.0, 5.0])
    n = np.array([7, 9, 0, 12, 20], dtype="int64")
    y = np.array([0, 4, 0, 10, 20], dtype="int64")
    w = np.array([2.0, -0.75, 3.0, 0.0, 1.25])
    actual_score, actual_curvature = evaluate(z, n, y, w)
    p = expit(z)
    np.testing.assert_allclose(actual_score, w * (y - n * p), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        actual_curvature, -w * n * p * (1.0 - p), rtol=1e-12, atol=1e-13
    )


def test_support_nonfinite_logits_and_finite_tails(value_mode):
    eta = np.array(
        [
            np.inf,
            np.inf,
            -np.inf,
            -np.inf,
            np.inf,
            -np.inf,
            np.nan,
            np.nan,
            0.0,
            0.0,
            1000.0,
            -1000.0,
        ]
    )
    trials = np.array([3, 3, 3, 3, 0, 0, 3, -1, 3, 3, 1, 1], dtype="int64")
    observed = np.array([3, 2, 0, 1, 0, 0, 1, 0, -1, 4, 0, 1], dtype="int64")
    expected = np.array(
        [
            0.0,
            -np.inf,
            0.0,
            -np.inf,
            0.0,
            0.0,
            np.nan,
            -np.inf,
            -np.inf,
            -np.inf,
            -1000.0,
            -1000.0,
        ]
    )
    with np.errstate(over="raise", invalid="raise"):
        actual = compile_logp(value_mode)(eta, trials, observed)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "trials, observed, eta, expected",
    [
        pytest.param(
            np.array([-(2**63), 0, 2**63 - 1, 2**63 - 1], dtype="int64"),
            np.array([0, -(2**63), 2**63 - 1, 0], dtype="int64"),
            np.array([0.0, 0.0, np.inf, -np.inf]),
            np.array([-np.inf, -np.inf, 0.0, 0.0]),
            id="signed-extremes",
        ),
        pytest.param(
            np.array([2**64 - 1, 2**64 - 1, 0], dtype="uint64"),
            np.array([2**63 - 1, -1, -(2**63)], dtype="int64"),
            np.array([np.inf, 0.0, 0.0]),
            np.array([-np.inf, -np.inf, -np.inf]),
            id="unsigned-trials-signed-observed",
        ),
        pytest.param(
            np.array([2**63 - 1, -1, 0], dtype="int64"),
            np.array([2**64 - 1, 0, 0], dtype="uint64"),
            np.array([0.0, np.nan, 0.0]),
            np.array([-np.inf, -np.inf, 0.0]),
            id="signed-trials-unsigned-observed",
        ),
    ],
)
def test_integer_extremes_do_not_wrap(value_mode, trials, observed, eta, expected):
    evaluate = compile_logp(value_mode, str(trials.dtype), str(observed.dtype))
    with np.errstate(over="raise", invalid="raise"):
        actual = evaluate(eta, trials, observed)
    np.testing.assert_array_equal(actual, expected)


def test_invalid_support_gradient_does_not_overflow_before_masking():
    eta = pt.dvector("eta")
    trials = pt.lvector("trials")
    observed = pt.lvector("observed")
    score = pt.grad(BinomialLogpOp()(eta, trials, observed).sum(), eta)
    evaluate = pytensor.function([eta, trials, observed], score, mode="FAST_COMPILE")
    n = np.array([-(2**63), 3, 2**63 - 1, 1, 1], dtype="int64")
    y = np.array([2**63 - 1, 4, -(2**63), 0, 1], dtype="int64")
    with np.errstate(over="raise", invalid="raise"):
        actual = evaluate(np.array([0.0, 0.0, 0.0, np.nan, np.inf]), n, y)
    np.testing.assert_array_equal(actual, np.full(5, np.nan))


def test_empty_vectors(value_mode):
    actual = compile_logp(value_mode)(
        np.empty(0, dtype="float64"),
        np.empty(0, dtype="int64"),
        np.empty(0, dtype="int64"),
    )
    assert actual.shape == (0,)
    assert actual.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "static_eta", [False, True], ids=["unknown-eta", "eta-length-three"]
)
@pytest.mark.parametrize("trials_length, observed_length", [(1, 3), (3, 2)])
def test_value_evaluation_requires_equal_lengths(
    value_mode, static_eta, trials_length, observed_length
):
    # Counts retain unknown symbolic lengths even when eta has a static length.
    # Static shape-only queries are not input validation; evaluate the values.
    evaluate = compile_logp(value_mode, static_eta=static_eta)
    with pytest.raises(ValueError):
        evaluate(
            np.zeros(3, dtype="float64"),
            np.ones(trials_length, dtype="int64"),
            np.zeros(observed_length, dtype="int64"),
        )


@pytest.mark.parametrize(
    "position, bad_input",
    [
        pytest.param(0, np.zeros(3, dtype="float32"), id="eta-float32"),
        pytest.param(0, np.float64(0), id="eta-scalar"),
        pytest.param(1, np.ones(3, dtype="float64"), id="trials-float"),
        pytest.param(1, np.ones(3, dtype="bool"), id="trials-bool"),
        pytest.param(1, np.ones((1, 3), dtype="int64"), id="trials-matrix"),
        pytest.param(2, np.zeros(3, dtype="float64"), id="observed-float"),
        pytest.param(2, np.zeros(3, dtype="bool"), id="observed-bool"),
        pytest.param(2, np.int64(0), id="observed-scalar"),
    ],
)
def test_rejects_wrong_input_dtype_or_rank(position, bad_input):
    inputs = [np.zeros(3), np.ones(3, dtype="int64"), np.zeros(3, dtype="int64")]
    inputs[position] = bad_input
    with pytest.raises(TypeError):
        BinomialLogpOp()(*inputs)


def test_rejects_conflicting_static_lengths():
    with pytest.raises(ValueError):
        BinomialLogpOp()(
            np.zeros(3), np.ones(2, dtype="int64"), np.zeros(3, dtype="int64")
        )
