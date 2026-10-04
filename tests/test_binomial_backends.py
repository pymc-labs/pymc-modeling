"""CPU lowering contracts; native AD is not the base Op's symbolic pullback."""

import importlib

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from pytensor.compile.mode import Mode
from scipy.special import expit
from scipy.stats import binom

from binomial_logp_op import BinomialLogpOp


def _finite_case():
    eta = np.array([-40.0, -2.0, 0.0, 0.7, 3.0, 40.0])
    trials = np.array([9, 12, 8, 11, 7, 9], dtype="int64")
    observed = np.array([0, 3, 4, 8, 5, 9], dtype="int64")
    return eta, trials, observed


def _support_case():
    limits = np.iinfo("int64")
    eta = np.array(
        [
            np.inf,
            np.inf,
            -np.inf,
            -np.inf,
            np.nan,
            np.nan,
            0.0,
            0.0,
            0.0,
            2.0,
            np.inf,
            -np.inf,
        ]
    )
    trials = np.array([4, 4, 4, 4, 4, -1, limits.max, 2, -1, 0, 0, 0], dtype="int64")
    observed = np.array([4, 3, 0, 1, 2, 0, limits.min, 3, 0, 0, 0, 0], dtype="int64")
    expected = np.array(
        [
            0.0,
            -np.inf,
            0.0,
            -np.inf,
            np.nan,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            0.0,
            0.0,
            0.0,
        ]
    )
    return (eta, trials, observed), expected


def _forward_cases():
    eta, trials, observed = _finite_case()
    yield (eta, trials, observed), binom.logpmf(observed, trials, expit(eta))
    yield _support_case()


def _derivative_reference():
    eta, trials, observed = _finite_case()
    weights = np.array([1.5, -0.7, 2.0, -1.1, 0.25, -2.0])
    direction = np.array([-0.5, 1.0, 1.5, -2.0, 0.75, 2.5])
    # Stable analytic derivatives, independent of any backend implementation.
    score = weights * (observed * expit(-eta) - (trials - observed) * expit(eta))
    hvp = -weights * trials * expit(eta) * expit(-eta) * direction
    return weights, direction, score, hvp


def _assert_logpmf(actual, expected):
    actual = np.asarray(actual)
    assert actual.dtype == np.dtype("float64")
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True)


def _linked_forward(linker):
    inputs = [
        pt.vector("eta", dtype="float64"),
        pt.lvector("trials"),
        pt.lvector("observed"),
    ]
    return pytensor.function(
        [pytensor.In(value, strict=True) for value in inputs],
        BinomialLogpOp()(*inputs),
        mode=Mode(linker=linker, optimizer="fast_run"),
    )


@pytest.fixture(scope="module")
def jax_backend():
    jax = pytest.importorskip("jax", exc_type=ModuleNotFoundError)
    previous_x64 = jax.config.x64_enabled
    # PyTensor dispatch initializes JAX x64 from floatX when first imported.
    with pytensor.config.change_flags(floatX="float64"):
        jax.config.update("jax_enable_x64", True)
        try:
            with jax.default_device(jax.devices("cpu")[0]):
                module = importlib.import_module("binomial_logp_jax")
                yield jax, module.binomial_logp
        finally:
            jax.config.update("jax_enable_x64", previous_x64)


@pytest.fixture(scope="module")
def pytorch_backend():
    torch = pytest.importorskip("torch", exc_type=ModuleNotFoundError)
    with torch.device("cpu"):
        module = importlib.import_module("binomial_logp_pytorch")
        yield torch, module.binomial_logp


@pytest.fixture(scope="module")
def mlx_backend():
    mx = pytest.importorskip("mlx.core", exc_type=ModuleNotFoundError)
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        module = importlib.import_module("binomial_logp_mlx")
        yield mx, module.binomial_logp
    finally:
        mx.set_default_device(previous_device)


@pytest.mark.jax
def test_jax_compiled_forward_and_support(jax_backend):
    jax, logp = jax_backend
    compiled = jax.jit(logp)
    for inputs, expected in _forward_cases():
        _assert_logpmf(
            compiled(*(jax.numpy.asarray(value) for value in inputs)), expected
        )


@pytest.mark.jax
def test_jax_native_weighted_score_and_curvature(jax_backend):
    jax, logp = jax_backend
    eta, trials, observed = map(jax.numpy.asarray, _finite_case())
    weights, direction, expected_score, expected_hvp = _derivative_reference()
    weights, direction = map(jax.numpy.asarray, (weights, direction))
    score = jax.grad(lambda z: jax.numpy.sum(weights * logp(z, trials, observed)))
    actual_score = jax.jit(score)(eta)
    actual_hvp = jax.jit(lambda z, v: jax.jvp(score, (z,), (v,))[1])(eta, direction)
    # No absolute tolerance: the endpoint-tail scores must not collapse to zero.
    np.testing.assert_allclose(actual_score, expected_score, rtol=2e-10, atol=0)
    np.testing.assert_allclose(actual_hvp, expected_hvp, rtol=2e-10, atol=0)

    inputs, _ = _support_case()
    z, n, y = map(jax.numpy.asarray, inputs)
    invalid_score = jax.jit(jax.grad(lambda value: logp(value, n, y).sum()))(z)
    expected = np.full(z.shape, np.nan)
    expected[9] = 0.0  # Only finite eta with zero trials has a defined score here.
    np.testing.assert_allclose(invalid_score, expected, equal_nan=True)


@pytest.mark.jax
def test_jax_rejects_length_one_broadcast(jax_backend):
    jax, logp = jax_backend
    inputs = list(map(jax.numpy.asarray, _finite_case()))
    compiled = jax.jit(logp)
    for index in (1, 2):
        mismatched = inputs.copy()
        mismatched[index] = mismatched[index][:1]
        with pytest.raises(ValueError, match="equal length"):
            compiled(*mismatched)


@pytest.mark.jax
def test_jax_pytensor_linked_forward(jax_backend):
    from pytensor.link.jax.linker import JAXLinker

    linked = _linked_forward(JAXLinker())
    for inputs, expected in _forward_cases():
        _assert_logpmf(linked(*inputs), expected)


@pytest.mark.pytorch
def test_pytorch_compiled_forward_and_support(pytorch_backend):
    torch, logp = pytorch_backend
    compiled = torch.compile(logp, fullgraph=True)
    for inputs, expected in _forward_cases():
        result = compiled(*(torch.as_tensor(value, device="cpu") for value in inputs))
        _assert_logpmf(result.detach().numpy(), expected)


@pytest.mark.pytorch
def test_pytorch_native_weighted_score_and_curvature(pytorch_backend):
    torch, logp = pytorch_backend
    eta, trials, observed = [
        torch.as_tensor(value, device="cpu") for value in _finite_case()
    ]
    eta.requires_grad_()
    weights, direction, expected_score, expected_hvp = _derivative_reference()
    weights, direction = [
        torch.as_tensor(value, device="cpu") for value in (weights, direction)
    ]

    # Compiled forward and first backward; compiled double backward is not claimed.
    compiled = torch.compile(logp, fullgraph=True)
    compiled_score = torch.autograd.grad(
        (weights * compiled(eta, trials, observed)).sum(), eta
    )[0]
    np.testing.assert_allclose(
        compiled_score.detach().numpy(), expected_score, rtol=2e-10, atol=0
    )

    eager_score = torch.autograd.grad(
        (weights * logp(eta, trials, observed)).sum(),
        eta,
        create_graph=True,
    )[0]
    actual_hvp = torch.autograd.grad((eager_score * direction).sum(), eta)[0]
    np.testing.assert_allclose(
        eager_score.detach().numpy(), expected_score, rtol=2e-10, atol=0
    )
    np.testing.assert_allclose(
        actual_hvp.detach().numpy(), expected_hvp, rtol=2e-10, atol=0
    )

    inputs, _ = _support_case()
    z, n, y = [torch.as_tensor(value, device="cpu") for value in inputs]
    z.requires_grad_()
    invalid_score = torch.autograd.grad(logp(z, n, y).sum(), z)[0]
    expected = np.full(z.shape, np.nan)
    expected[9] = 0.0
    np.testing.assert_allclose(invalid_score.detach().numpy(), expected, equal_nan=True)


@pytest.mark.pytorch
def test_pytorch_rejects_broadcast_and_uint64(pytorch_backend):
    torch, logp = pytorch_backend
    inputs = [torch.as_tensor(value, device="cpu") for value in _finite_case()]
    # Test public input validation eagerly: compiler exception wrappers vary.
    for index in (1, 2):
        mismatched = inputs.copy()
        mismatched[index] = mismatched[index][:1]
        with pytest.raises(ValueError, match="equal length"):
            logp(*mismatched)
    with pytest.raises(NotImplementedError, match="uint64"):
        logp(inputs[0], inputs[1].to(torch.uint64), inputs[2])


@pytest.mark.pytorch
def test_pytorch_pytensor_linked_forward(pytorch_backend):
    from pytensor.link.pytorch.linker import PytorchLinker

    linked = _linked_forward(PytorchLinker())
    for inputs, expected in _forward_cases():
        _assert_logpmf(linked(*inputs), expected)


def _mlx_arrays(mx, values):
    # MLX otherwise narrows NumPy float64 inputs to float32 during conversion.
    return tuple(
        mx.array(value, dtype=getattr(mx, str(value.dtype))) for value in values
    )


@pytest.mark.mlx
def test_mlx_compiled_forward_and_support(mlx_backend):
    mx, logp = mlx_backend
    compiled = mx.compile(logp)
    for inputs, expected in _forward_cases():
        _assert_logpmf(compiled(*_mlx_arrays(mx, inputs)), expected)


@pytest.mark.mlx
def test_mlx_native_weighted_score_and_curvature(mlx_backend):
    mx, logp = mlx_backend
    eta, trials, observed = _mlx_arrays(mx, _finite_case())
    weights, direction, expected_score, expected_hvp = _derivative_reference()
    weights, direction = _mlx_arrays(mx, (weights, direction))
    score = mx.grad(lambda z: mx.sum(weights * logp(z, trials, observed)))
    actual_score = mx.compile(score)(eta)
    # Reverse over reverse, not an unsupported forward-mode JVP or symbolic pullback.
    actual_hvp = mx.grad(lambda z: mx.sum(score(z) * direction))(eta)
    np.testing.assert_allclose(
        np.asarray(actual_score), expected_score, rtol=2e-10, atol=0
    )
    np.testing.assert_allclose(np.asarray(actual_hvp), expected_hvp, rtol=2e-10, atol=0)

    inputs, _ = _support_case()
    z, n, y = _mlx_arrays(mx, inputs)
    invalid_score = mx.grad(lambda value: logp(value, n, y).sum())(z)
    expected = np.full(z.shape, np.nan)
    expected[9] = 0.0
    np.testing.assert_allclose(np.asarray(invalid_score), expected, equal_nan=True)


@pytest.mark.mlx
def test_mlx_rejects_length_one_broadcast(mlx_backend):
    mx, logp = mlx_backend
    inputs = list(_mlx_arrays(mx, _finite_case()))
    compiled = mx.compile(logp)
    for index in (1, 2):
        mismatched = inputs.copy()
        mismatched[index] = mismatched[index][:1]
        with pytest.raises(ValueError, match="equal length"):
            compiled(*mismatched)


@pytest.mark.mlx
def test_mlx_pytensor_linked_forward(mlx_backend):
    from pytensor.link.mlx.linker import MLXLinker

    linked = _linked_forward(MLXLinker(use_compile=True))
    for inputs, expected in _forward_cases():
        _assert_logpmf(linked(*inputs), expected)
