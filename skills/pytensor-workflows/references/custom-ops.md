# Custom Ops, differentiation, and backend boundaries

## Choose the smallest necessary extension

For an ordinary binomial observation model, use PyMC's built-in `Binomial`.
The example here deliberately reimplements a known likelihood so that an
**external numerical likelihood boundary** can be checked against an independent
reference before it is trusted in a model. It is not a faster or more accurate
replacement for the built-in distribution.

1. Express the calculation with existing `pytensor.tensor` operations when possible.
   This preserves symbolic derivatives and existing backend implementations.
2. Use `OpFromGraph` to encapsulate a reusable symbolic calculation. It does not
   make unsupported inner Ops supported by another backend.
3. Use `wrap_py` for a deliberately value-only Python boundary. It supplies no
   derivative automatically. `as_op` is deprecated in PyTensor 3.3.0.
4. Implement an `Op` when an external calculation needs a controlled
   shape/dtype contract, analytic derivatives, or explicit backend lowering.
5. A log density is not a random generator. A custom likelihood integration must
   separately define predictive generation, or clearly state that it is absent.
   Keep observation-level log probabilities for later pointwise diagnostics;
   sum only where a scalar objective is actually required.

Verify derivatives before integrating the Op into an inference algorithm.

## API compatibility

The supplied modules use PyTensor 3.3.0's `pullback` interface. Check installed
signatures against the linked official source before adapting an extension.
Numerical examples do not guarantee support across package versions or devices.

## Contract: a pointwise BinomialLogpOp

The reusable [implementation](../scripts/binomial_logp_op.py) requires PyTensor,
NumPy and SciPy. With this skill's `scripts/` directory on the Python import path:

```python
from binomial_logp_op import BinomialLogpOp

pointwise_logp = BinomialLogpOp()(eta, trials, observed)
```

| Input/output | Exact contract |
|---|---|
| `eta` | One-dimensional `float64` tensor of log odds, not probabilities |
| `trials`, `observed` | One-dimensional signed or unsigned integer tensors; bool and floating-point counts are rejected |
| Shape | All three lengths equal; no scalar or length-one broadcasting |
| Output | One `float64` vector with one log probability per input position, in the same order |
| Counts outside support | `trials < 0`, `observed < 0`, or `observed > trials` gives `-inf` at that position |
| Zero trials | `trials = observed = 0` gives log probability zero for finite or infinite eta |
| Infinite eta | Limiting point mass at all failures (`-inf`) or all successes (`+inf`); other counts have log probability `-inf` |
| NaN eta | Propagates NaN for otherwise valid counts; invalid counts remain `-inf` |
| Empty vectors | Valid empty pointwise result |
| Derivatives | Finite eta and valid counts only; undefined count derivatives, NaN score at invalid support or nonfinite eta |

The observation model is conditionally independent binomial counts with known
trials. At finite eta, write `p = sigmoid(eta)` and compute

\[
\ell_i=\log\Gamma(n_i+1)-\log\Gamma(y_i+1)-\log\Gamma(n_i-y_i+1)
-y_i\operatorname{softplus}(-\eta_i)-(n_i-y_i)\operatorname{softplus}(\eta_i).
\]

The Python implementation uses SciPy `gammaln` and NumPy `logaddexp`. The NUMBA
implementation uses `math.lgamma` and stable `log1p(exp(-abs(eta)))` branches.
Neither converts large positive eta to a rounded probability before evaluating
the density. Explicit infinite-eta branches avoid `0 * inf` at valid endpoints.
Signed/unsigned support comparisons and count differences are performed in
integer space before conversion to floating point.

The log-gamma difference can lose accuracy at enormous trial counts. Exact
signed/unsigned support rejection does not imply accurate 64-bit-count
densities. For large-count use, compare against an appropriate high-precision
or specialized implementation; do not clip positive log probabilities to zero
or quietly round counts.

### Op and Type authoring responsibilities

`make_node` converts inputs into symbolic variables, rejects invalid dtype/rank
and incompatible static lengths, and returns `Apply(self, inputs, outputs)`.
It does not run NumPy on symbolic variables. The example uses `eta.type()` to
preserve its static vector type while promising float64 output. Static types
cannot establish equality between three unknown runtime lengths: `perform`,
the gradient graph, and shape inference each preserve that condition.

`perform(node, inputs, output_storage)` receives concrete arrays. Write each
output into its existing one-element storage cell, not by replacing the cell
list. Do not mutate inputs or alias a new output to an input without declaring
the corresponding `destroy_map` or `view_map`. This implementation allocates its
own output, has no view/inplace contract, and is deterministic. If recycling
storage in a different Op, account for changing shape and arbitrary strides.

Use immutable `__props__` for every attribute that changes the calculation.
The empty tuple here means all instances implement the same operation; equality
and hashing promise equivalent computation, not merely matching names.
`default_output` controls single-versus-multiple-output convenience return
behavior. `itypes`/`otypes` can express a simple fixed signature; `make_node` is
clearer for these rank, dtype, and equality checks. `connection_pattern` reports
input/output dependence, **not** whether a derivative was convenient to write.
Counts affect this likelihood, so they are not declared disconnected.

`infer_shape(fgraph, node, shapes)` returns one shape tuple per output without
needing the expensive density calculation. The supplied equality guard means a
shape-only query cannot silently accept mismatched lengths. `do_constant_folding`
controls compile-time evaluation on constants; the default is appropriate for
this pure Op. `debug_perform` is an optional alternate diagnostic implementation,
not an error-hiding fallback. `flops` estimates profiling work, not correctness.
`prepare_node`, `make_py_thunk`, `make_thunk`, and the `ThunkType` storage/compute
protocol are lower-level extension hooks: keep the defaults for a normal Python
Op. There is no need to manipulate a linker's storage map manually.

Use an existing `TensorType` for arrays. A PyTensor `Type` is a symbolic value
contract, not Python's `type` object. Its `filter(data, strict, allow_downcast)`
validates concrete values; `filter_variable`/`convert_variable` handle symbolic
compatibility. `is_super` expresses a set-of-values relation, whereas
`in_same_class` has type-specific rules (including broadcasting for tensors).
`make_variable`, `make_constant`, and `Type()` create graph variables/constants;
`clone`, equality, and hashing must preserve immutable type semantics.
`values_eq`/`values_eq_approx` are value-comparison contracts, not a reason to
silently relax likelihood tolerances. `HasDataType`/`HasShape` expose dtype,
rank, and partial static shape. Check strict filtering and static shape narrowing
with actual invalid inputs.

`pt.scalar` creates a **zero-dimensional tensor**, not an internal `ScalarType`.
In PyTensor 3.3.0, `ScalarSharedVariable` is a deprecated alias for
`TensorSharedVariable`; use the latter and check `ndim == 0`.

A genuinely new runtime value type must implement its own filtering,
comparison, and storage semantics, including alias detection for DebugMode and
appropriate backend representation. Optional extension hooks such as
`Type.filter_inplace`, `Op.infer_shape`, or `Op.__props__` may be documented
subclass protocols without existing as attributes on the abstract base class.

## Differentiation: reverse, forward, and higher order

Implement `pullback(self, inputs, outputs, cotangents)` with one symbolic
cotangent per input. For this Op, multiply the output cotangent by the eta
score and explicitly mark the count derivatives undefined.

The pointwise score is

\[
\frac{\partial\ell_i}{\partial\eta_i}=y_i-n_i p_i
=y_i\sigma(-\eta_i)-(n_i-y_i)\sigma(\eta_i),
\qquad
\frac{\partial^2\ell_i}{\partial\eta_i^2}=-n_i p_i(1-p_i).
\]

The second form of the score avoids cancellation at saturated logits. The
Jacobian is diagonal, so the VJP is `cotangent * score`; returning just `score`
would be wrong for a weighted objective. The score is itself a symbolic graph,
allowing higher-order derivatives with respect to eta. No finite differences
are used inside sampling or inside `pullback`.

| Interface family | Use and boundary |
|---|---|
| `grad(cost, wrt)` | Reverse-mode derivative of a scalar cost; `known_grads` supplies boundary cotangents and `consider_constant` stops selected paths |
| `pullback(f, wrt, cotangents)` | Public VJP helper, including non-scalar outputs |
| `pushforward(f, wrt, tangents)` | Public JVP helper; PyTensor 3.3.0 defaults to applying pullback twice |
| `pushforward(..., use_op_pushforward=True)` | Requires direct `pushforward(self, inputs, outputs, tangents)` methods; BinomialLogpOp does not supply one |
| `jacobian`, `hessian` | Dense derivatives for small checks; potentially quadratic storage/large generated graphs |
| `hessian_vector_product` | Curvature-vector product without materializing the Hessian; compare with analytic curvature |
| `subgraph_grad` | Stage reverse differentiation at intermediate variables; do not count a cost both in `start` cotangents and again as `cost` |
| `verify_grad` | Random scalar projections with finite differences; fix integer counts and perturb only floating inputs |
| `numeric_grad`, `GradientError` | Numerical comparison machinery/error details; inspect absolute and relative error, step size and problem scale rather than just relaxing tolerances |

When checking a selected mode, consider
`verify_grad(..., mode=mode, no_debug_ref=False)`: in PyTensor 3.3.0 the default
`no_debug_ref=True` can use a different reference compilation mode. Check both
the reference and derivative paths rather than inferring backend parity.

A JVP constructed through reverse-mode differentiation is not a specialized
Op-level forward implementation. Use the public `pushforward` options to choose
the intended route. In the linked version, `Lop`/`Rop`, `Op.L_op`/`Op.R_op`,
and `Op.grad` are legacy interfaces; use `pullback`/`pushforward` for new code.

### Undefined is not disconnected or zero

- `grad_undefined` constructs an uncomputable `NullType` for a mathematically
  undefined derivative. It is returned for trials and observed: these are
  discrete inputs, and this likelihood defines no continuous extension in them.
  Requesting their gradients should raise `NullTypeGradError`.
- `grad_not_implemented` also constructs an uncomputable gradient, but means the
  derivative has not been implemented, not that it is mathematically undefined.
- `DisconnectedType` means the cost does not depend on that path. Return it for
  disconnected output cotangents. It is not a way to silence an integer or
  missing likelihood derivative. `DisconnectedInputError` makes an unexpected
  disconnected request visible.
- `zero_grad` keeps the value but substitutes a zero derivative;
  `disconnected_grad` removes the gradient connection; `undefined_grad` raises
  when differentiation reaches it. These have different graph semantics.
- `grad_clip` and `grad_scale` alter the **backpropagated derivative**, not the
  function value. Supplying that result as the derivative of an unchanged log
  density breaks the force calculation used by gradient samplers.

A derivative at `eta = inf`, `eta = -inf`, or an out-of-support count is not an
ordinary real derivative even when a limiting score can be written down. The
example intentionally emits NaN for these eta scores instead of fabricating
finite differentiability. Finite-difference checks are performed away from
these nonfinite-domain boundaries.

## OpFromGraph and value-only boundaries

`OpFromGraph([inputs], [outputs], inline=...)` packages existing symbolic graphs.
In PyTensor 3.3.0, shared variables must be explicit inputs; hidden capture
raises `MissingInputError`, and `updates`/`givens` are unsupported. Compare
inline and retained inner-graph values and gradients without assuming a speedup.
`HasInnerGraph` exposes the graph boundary; compiler integration hooks are not
user-managed caches.

In that version, the constructor `pullback` override takes
`(inputs, outputs, cotangents)`, while `pushforward` takes
**`(inputs, tangents)`**, unlike `Op.pushforward`'s three arguments.
Check the linked builder for the installed API. Overrides must implement the
actual chain rule, including non-unit cotangents.

`wrap_py`/`FromFunctionOp` wraps a Python function with declared Types and an
optional shape function, but does not infer derivatives through NumPy/SciPy.
A wrapped NumPy square can evaluate correctly and still lack a symbolic
derivative. For gradient-based modeling, use symbolic primitives or implement
a real differentiable Op; do not substitute zero gradients.

## Backends: lowering is a separate contract

| Backend | Supplied implementation and restriction |
|---|---|
| FAST_COMPILE | Python `perform` and symbolic derivatives |
| CVM (`Mode(linker="cvm", optimizer="fast_run")`) | Python Op boundary in a mixed C/Python graph, not a C kernel |
| NUMBA | Explicit lowering; import its module before compiling |
| C-only linker | No `c_code` implementation in this example |
| JAX | Opt-in native density/JVP; x64 required |
| PyTorch | Opt-in CPU density/autograd; uint64 rejected |
| MLX | Opt-in CPU float64 density/VJP; no silent device or precision substitution |

`FAST_RUN` may select a configured Numba linker rather than CVM. Check the
actual linker and import its registration module before compilation.

Enable only the implemented optional lowering:

```python
import binomial_logp_numba  # registers BinomialLogpOp with numba_funcify
from binomial_logp_op import BinomialLogpOp
```

The registration returns a real `numba.njit(fastmath=False)` implementation.
There is no `objmode`/Python `perform` callback and no fabricated speedup claim.
Inspect fallback warnings and nopython specializations before claiming native
execution: upstream Numba dispatch can call Python in object mode.
The supplied lowering repeats length/support checks before indexing and
preserves NaN/infinity semantics. Its registration has no custom persistent
PyTensor cache key; separate cold compilation from steady-state timing.

User-facing extension points are `numba_funcify`, `jax_funcify`,
`pytorch_funcify`, and `mlx_funcify` (in their respective `link.*.dispatch`
modules). Register on the **Op class** and return a backend function whose
input/output count, dtype, rank, support and side effects match `make_node` and
`perform`. The `*_typify` dispatchers convert concrete data representations;
they do not implement an Op's computation. Extending a non-tensor `Type` may
require both representation support and lowering. Linker `fgraph_convert` and
`jit_compile` organize those conversions; ordinary authors select a `Mode`
rather than editing internal linker schedules, thunks, rewrite sets or storage.

`pytensor.wrap_jax` wraps a JAX-jittable function, including PyTrees, into
symbolic inputs/outputs. `allow_eval` governs attempts to infer unknown shapes
by evaluation; wrapping does not make arbitrary Python JAX-compatible.
See [optional backends](custom-backends.md) for native AD, dynamic shapes and device
restrictions. Missing optional packages do not imply those extension APIs were
removed, and float64 support must be checked on the actual device.

### C and scalar implementation boundary

`COp`/`CLinkerOp` remain legitimate native extension interfaces. Their contract
includes `c_code` and failure propagation, apply/struct support and cleanup
hooks, headers/libraries/search directories/compile flags, initialization,
parameter extraction, and cache-version methods. A `CLinkerType` additionally
owns C declaration, extraction, initialization, synchronization and cleanup;
`CType` is the combined Type/C interface. Cache versions must change when
semantics or ABI-dependent generated code change. A Python `perform` or NUMBA
lowering supplies none of this automatically. Toolchain/OpenMP setup requires
its own portability and numerical checks, not an unimplemented C skeleton.

Use tensor-level numerical functions for modeling. Internal scalar kernels and
`Composite` support elementwise execution/fusion; they are not an alternate
modeling API or an automatic backend bridge.

## Check an extension before use

- Compare pointwise log densities to an independent reference, preserving
  observation order and log-space tails. Check the intended large-count range
  separately from exact support classification.
- Check weighted VJPs, JVPs and Hessian-vector products against analytic formulas.
  A gradient of an unweighted sum cannot detect a missing cotangent factor.
  Use finite differences only for continuous inputs away from invalid support,
  nonfinite values and singularities; choose step sizes from problem scale.
- Check dtype/rank, static and dynamic lengths, especially length-one inputs
  that could accidentally broadcast. Include empty and noncontiguous arrays,
  zero trials, mixed signed/unsigned extremes and limiting point masses.
- Verify nonmutation and absence of undeclared output aliasing. Check value,
  shape-only and derivative graphs separately because rewrites can remove
  the expensive forward Op.
- Undefined integer gradients, disconnected inputs, opaque Python gradients,
  unsupported direct JVPs and missing backend kernels must remain visible errors.
- Require genuine native execution when that is the goal; a selected mode or
  successful forward value does not establish it. Do not weaken numerical
  tolerances simply because another backend agrees.

## Authoritative sources

These pinned sources explain PyTensor 3.3.0 contracts and limitations. Consult
the matching installed version when adapting an implementation.

- [Released Op contract](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/doc/extending/op.rst)
  and [implementation](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/graph/op.py).
- [Released Type source](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/graph/type.py)
  and [Type contract](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/doc/extending/type.rst).
- [Released autodiff implementation](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/gradient.py),
  [OpFromGraph](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/builders.py),
  and [wrap_py / auxiliary Ops](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/ops.py).
- [Released Numba dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/numba/dispatch/basic.py)
  and [backend-extension guide](https://pytensor.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html).
- [JAX dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/jax/dispatch/basic.py),
  [wrap_jax](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/jax/ops.py),
  [PyTorch dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/pytorch/dispatch/basic.py),
  and [MLX dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/mlx/dispatch/basic.py).
- [C extension contracts](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/c/interface.py)
  and [scalar publicness/implementation boundary](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/scalar/basic.py).
- [SciPy binomial reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
  and [gammaln](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaln.html).
