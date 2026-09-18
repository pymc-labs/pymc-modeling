# Optional custom-Op backends

Use these examples only when a custom numerical boundary genuinely needs native
lowering. Prefer built-in `pm.Binomial` for ordinary binomial modeling. A native
log density is neither a random generator nor a complete inference workflow.

## Choose and import one implementation

Keep the chosen module beside `binomial_logp_op.py` on the Python import path.
Import it **before** compiling the Op for that backend; the base Op does not
register optional lowerings or require their packages.

- [JAX](../scripts/binomial_logp_jax.py): JAX/jaxlib, float64 density and a custom
  continuous JVP. Enable x64 before imports, input conversion and graph construction.
- [PyTorch](../scripts/binomial_logp_pytorch.py): PyTorch, CPU float64 density
  and custom autograd backward. This implementation rejects uint64 counts.
- [MLX](../scripts/binomial_logp_mlx.py): MLX with CPU float64 support, native
  density and a custom continuous VJP. This implementation requires the CPU
  default device; it does not silently substitute float32 or move devices.

Numba's separate implementation is described with the base Op. Dependencies are
needed only for the selected example; do not install every backend to use this skill.

## Lowering versus representation

`*_funcify` maps an **Op class** to a backend-compatible callable. Register with,
for example, `@jax_funcify.register(BinomialLogpOp)`, not on one tensor or Op
instance. The callback receives graph metadata and returns native operations;
native execution must not call SciPy, `perform`, or a Python likelihood callback.

`*_typify` converts concrete representations. It does not implement an Op or its
derivative. Reuse existing ndarray/tensor registrations rather than globally
overwriting them. Keep shape/dtype checks in `make_node`, strict function inputs,
and the lowering itself; a typifier is not necessarily an arbitrary-object validator.

Use an explicit linker in `Mode(..., optimizer="fast_run")` when diagnosing a
backend. A successful `FAST_RUN` call does not identify the selected executor.
Use `In(..., strict=True)` if implicit float32-to-float64 input conversion would
conceal a contract violation.

## JAX

PyTensor 3.3.0's JAX dispatch initializes `jax_enable_x64` from PyTensor's
`floatX`; configure both before importing dispatch or constructing arrays. The
supplied lowering rejects disabled x64. Enabling it after a count array narrowed
cannot recover lost information. Inspect the installed JAX configuration API
rather than assuming an older experimental re-export still exists.

The density uses native `gammaln` and `logaddexp`. Its custom JVP uses the stable
score

\[
\frac{\partial\ell_i}{\partial\eta_i}
=y_i\sigma(-\eta_i)-(n_i-y_i)\sigma(\eta_i).
\]

Compute the count remainder in integer space before converting to float64.
This avoids both integer support errors and cancellation from subtracting a
rounded sigmoid from one. JAX integer tangents use `float0` semantics, not a
continuous count score. The core PyTensor integer-gradient request is undefined.
Continuous scores at invalid support or nonfinite eta are NaN, not fabricated zeros.

Check native `jax.jit` values, reverse-mode scores, weighted VJPs and JVPs of
the score independently. Native AD is distinct from compiling the base Op's
symbolic pullback graph.

**Shape-only caution:** the linked PyTensor 3.3.0 JAX dispatcher warns and drops
dynamic `CheckAndRaise` conditions. An optimized `op(...).shape` query may remove
the density node, bypassing its native length check. Do not use shape-only output
as input validation or override global assertion dispatch to hide the warning.
Check installed behavior before assuming this restriction persists.

## PyTorch

The linked dispatcher uses `torch.as_tensor` for NumPy arrays and tensors, so
storage may be shared. The supplied lowering never mutates inputs. The linker
uses `torch.compile` and converts outputs to NumPy arrays: those arrays do not
retain a native autograd graph. Differentiate the callable returned by
`pytorch_funcify`, not a NumPy result from `pytensor.function`.

The custom `torch.autograd.Function` implements a stable analytic backward.
Check native eager AD, compiled forward/first-backward behavior and higher-order
AD separately. `torch.compile(..., fullgraph=True)` exposes graph breaks when
that guarantee matters; do not infer compiled double-backward support from an
eager Hessian-vector product or a working first derivative.

Signed 8/16/32/64-bit and unsigned 8/16/32-bit counts widen losslessly to int64.
The example rejects uint64 even if a particular batch fits int64: blind casting
would change support above `2**63 - 1`. This is a limitation of this lowering,
not a claim that every Torch uint64 operation is absent. Non-CPU tensors are
also rejected rather than copied inside the likelihood.

The base symbolic pullback uses uint64 intermediates even for signed counts,
outside this Torch arithmetic subset. A forward registration does not provide
support for arbitrary downstream shape-only or symbolic derivative graphs.

## MLX

MLX CPU float64 support differs from Metal support. The linked PyTensor 3.3.0
source preserves float64 on CPU but narrows it on Metal with a warning. Do not
infer that MLX universally lacks float64 or that float32 preserves this Op.
[Official installation guidance](https://ml-explore.github.io/mlx/build/html/install.html#cpu-only-linux)
describes CPU-only Linux packages; check its current platform requirements.

`MLXLinker(use_compile=True)` uses `mx.compile`; `False` still typifies and
executes the graph. Disabling compilation does not supply missing lowerings.
The module uses the installed public `mlx_funcify(GammaLn())` approximation and
[`custom_function.vjp`](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html)
for the eta score. Check forward values, compiled first gradients and
reverse-over-reverse Hessian-vector products separately; no native JVP is supplied.

Float64 storage alone does not establish transcendental accuracy. The supplied
implementation follows the linked PyTensor MLX precision helper's Euler-power
form with an explicit float64 constant, nonpositive exponent arguments, and
`where` branches rather than an `abs` derivative at zero. Preserve the exact
identity `C(n,0) = C(n,n) = 1` at endpoints instead of clipping density errors.
Compare installed kernels at the precision required by the calculation.

## Shared mathematical and error contract

All examples require equal-length rank-one float64 logits and integer count
vectors, with no scalar/length-one broadcasting. They return a pointwise float64
log PMF in observation order. Float/bool counts and wrong ranks/dtypes are errors.

- Invalid counts give `-inf`; signed extremes must not overflow subtraction
  before masking. Exact support decisions do not imply accurate gamma differences
  at enormous counts.
- For valid counts, NaN logits propagate NaN. Infinite logits give limiting
  point masses at all failures/successes; impossible events remain `-inf`.
- Zero trials give zero log density for finite/infinite eta; empty vectors give
  empty output. These value limits do not make nonfinite eta differentiable.
- Missing registrations raise rather than silently executing another likelihood.
- Check shape, dtype, support classification and storage aliasing separately from
  numeric tolerances. Compare values with an independent log PMF and derivatives
  with analytic weighted scores, curvature and finite differences inside support.
- Log-gamma cancellation worsens at large counts. Choose precision and tolerances
  from the intended range; do not claim uniform accuracy across integer widths.

## Wrapping an existing JAX function

`pytensor.wrap_jax(jax_function=None, *, allow_eval=True)` accepts a JAX-jittable
function and nested input/output PyTrees, including VJPs. Static Python leaves
do not become differentiable inputs. This is an alternative authoring route,
not a way to make arbitrary Python JAX-compatible or register an existing Op.

Use `allow_eval=False` when unknown symbolic dimensions must fail instead of
being evaluated to infer example shapes. Supply real static shape information,
for example `pt.vector(dtype="float64", shape=(6,))`, only when it is known.
Data-dependent filtering such as `x[x > 0]` still has a non-static output shape
and is unsuitable for ordinary JAX JIT, even with fixed input length. A continuous
Normal factor is a suitable first value/VJP comparison; integer Binomial counts
do not have meaningful continuous VJPs.

## Official sources

These pinned PyTensor 3.3.0 sources ground version-specific cautions, not a
promise about another package version or device:

- [JAX dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/jax/dispatch/basic.py),
  [linker](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/jax/linker.py),
  [wrapper](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/jax/ops.py).
- [PyTorch dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/pytorch/dispatch/basic.py)
  and [linker](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/pytorch/linker.py).
- [MLX dispatch and precision helpers](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/mlx/dispatch/basic.py)
  and [linker](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/mlx/linker.py).
