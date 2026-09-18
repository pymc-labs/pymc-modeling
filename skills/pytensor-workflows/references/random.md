# Symbolic random computation

## Own the random state

Use explicit Generator state for positional or named-dimension symbolic draws.
API cautions below refer to PyTensor 3.3.0 and PyMC 6.3.1; check the official
sources against the installed version.

## State is an input and an output

Given the same Generator state, parameters, shape, algorithm, dtype, and execution
environment, a random variable's evaluation is reproducible. A deterministic
transform of an already drawn node uses that same
draw: `2 * x + 1` does not resample `x`. Evaluating a random expression and
persisting its *returned state* are separate operations.

```python
import pytensor
import pytensor.tensor.random as pr

initial = pr.shared_rng(seed=160923)
next_rng, x = initial.normal(0.0, 1.0, size=(4, 3), dtype="float64")
final_rng, y = next_rng.normal(0.0, 1.0, size=(4, 3), dtype="float64")
draw = pytensor.function([], [x, y, 2 * x + 1], updates={initial: final_rng})
first = draw()
second = draw()
initial.set_value(seed=160923)  # deliberate replay, not a per-call operation
replayed_first = draw()
```

Thread the returned RNG through successive random operations. Reusing `initial`
for both nodes is a state fork, not the same as sequential independent draws.
PyTensor 3.3.0's method wrapper warns on reuse; warning absence is not proof
that a rewritten graph cannot fork state. Update the shared RNG to the **last**
state in the chain. Updating only to `next_rng` discards the second operation's
progression and can reuse its state on the next call.

Three explicit initialization choices have different ownership semantics:

- `pr.shared_rng(seed=...)`: creates a NumPy `default_rng` state retained by the
  graph. Specify `seed=None` explicitly if nondeterministic initialization is
  intended. Calling `shared_rng()` with neither `value` nor `seed` is an error.
- `pr.shared_rng(np.random.default_rng(seed))`: copies the supplied Generator by
  default. `borrow=True` shares the object; external mutation and compiled
  in-place updates may then interact. Prefer isolated ownership unless sharing
  is intentional. `set_value(seed=...)` resets a shared RNG used by **all**
  functions that refer to it. Supplying both a Generator value and a seed is an
  error, not a precedence rule. Legacy NumPy `RandomState` is rejected.
- `root = pr.rng("state")`: creates a symbolic Generator input. Compile a
  function returning `[next_rng, sample]` and pass its returned Generator into
  the next call. With immutable inputs, check that the caller's original state
  is preserved. `pytensor.In(root, mutable=True)` explicitly permits mutation.

`pr.default_rng(symbolic_seed)` is an RNG-constructor **Op**, not shared storage.
A function rebuilding it from the same integer input reproduces the first draw
on every call. This is useful for intentionally indexed reproducibility, but
is not a progressing simulation stream. Do not confuse it with NumPy's eager
constructor or `pr.shared_rng`.

Setting `np.random.seed`, Python's `random.seed`, or a model sampler's seed is
not a substitute for seeding this explicitly owned Generator. Reproducibility
also depends on graph construction order, call order, draw sizes, backend,
BitGenerator and library versions. A seed is not a promise of identical draws
across hardware or algorithms. For intentionally separate simulations, create
child streams with `SeedSequence.spawn`; do not repeatedly reconstruct the same
seed and call the resulting samples independent replications.

### Diagnose omitted updates

`pytensor.function([], x)` without an explicit update on a Generator graph can
repeatedly evaluate the same stored state. Compare the state before and after
evaluation, then compile with `updates={initial: next_rng}` when progression is
intended. Do not inject a wall-clock seed or require every pair of random values
to differ: discrete draws can legitimately repeat. Check state transitions and
whole-sequence reconstruction instead.

## Distribution and shape families

Use `pr.<name>(..., rng=state, return_next_rng=True)` or `state.<name>(...)`.
The generated methods automatically supply these two keywords and return
`(next_rng, draw)`. Functional wrappers such as `choice`, `chisquare`, and
`standard_normal` can build several graph nodes; do not assume every result's
immediate owner is a RandomVariable. Keep the returned RNG instead of extracting
`draw.owner.outputs[0]` by hand.

`size` is the complete requested **batch** shape, not automatically an extra
leading Monte Carlo axis. With scalar-support parameters shaped `(3,)`, request
`size=(5, 3)` for five replicates of those three parameter values; `size=(5,)`
is incompatible. `size=None` infers the broadcast batch shape. `size=()` requests
a scalar batch, not a zero-length array. Support/core axes come after the batch
axes: Dirichlet concentrations `(3,)` with `size=(5,)` yield `(5, 3)`;
multivariate-normal mean `(2,)` and covariance `(2, 2)` yield `(5, 2)`.
Integer or integer-sequence sizes are appropriate; do not rely on coercion of
fractional sizes. A symbolic size vector needs a statically known length even
when its dimension values vary. Negative extents and incompatible parameter
broadcasting require fixing the declared shape, not flattening observations.

Dtypes are part of the Op contract. Continuous RVs ordinarily resolve `floatX`;
use a supported `dtype="float32"`/`"float64"` when needed and check the actual
output. Discrete RVs remain integer-valued and are not made continuous by
requesting a floating dtype. Choice/permutation preserve the selected values'
dtype. Check the requested and actual dtype separately from state replay.

Choose the law by parameterization and support, not by similarity of names:

| Family / exports | Parameters and appropriate use | Important boundaries |
| --- | --- | --- |
| `normal`, `standard_normal` | location/scale; standardized noise and deterministic location-scale transforms | Scale is standard deviation, not variance; NumPy bitwise comparisons require the same algorithm and consumption order. |
| `cauchy`, `t`, `laplace`, `logistic`, `gumbel` | real-valued location/scale noise; `t(df, loc, scale)` | Positive scale and positive t degrees of freedom. Heavy tails need not have finite moments; do not impose a universal sample-mean gate. |
| `uniform`, `triangular`, `beta` | finite-support draws; `triangular(left, mode, right)`, `beta(alpha, beta)` | Ordered interval/mode; positive beta shapes. Beta support is [0,1], not an arbitrary interval. |
| `halfnormal`, `halfcauchy` | positive-side location/scale draws | In the low-level API both accept `loc, scale`; this is not PyMC's one-parameter HalfNormal interface. |
| `exponential`, `gamma`, `chisquare`, `invgamma`, `gengamma` | positive/skewed quantities | Exponential uses **scale**; positional `gamma(shape, second)` binds deprecated **rate**, so write `gamma(shape, scale=...)`. `invgamma(shape, scale)` reciprocates a gamma draw with reciprocal scale. `gengamma(alpha,p,lambd)` maps to SciPy shapes `(alpha/p,p)` and scale `lambd`; do not substitute SciPy's first shape directly. |
| `lognormal`, `pareto`, `wald`, `weibull`, `truncexpon` | positive or bounded-positive quantities | Lognormal `mean,sigma` are log-scale normal parameters. Pareto `b,scale` uses SciPy's lower bound `scale`, not NumPy's shifted Pareto output. Wald uses `mean,scale`. Weibull takes unit-scale `shape`; scale by a deterministic transform. Truncated exponential `b,loc,scale` has upper endpoint `loc+b*scale`. |
| `vonmises` | circular data via `mu,kappa` | Nonnegative concentration; angular wrap matters. Check support under the chosen location/wrap convention, not a linear-noise assumption. |
| `bernoulli`, `binomial`, `betabinom` | binary/success counts; `(p)`, `(n,p)`, `(n,a,b)` | Valid probabilities, nonnegative trial counts, positive beta shapes; count support [0,n]. |
| `poisson`, `geometric`, `negative_binomial` / `nbinom` | event counts, trials until first success, failures before n successes | Poisson uses nonnegative `lam`; geometric starts at 1. Negative binomial uses `(n,p)`, not PyMC's `(mu,alpha)`; `nbinom` is an alias, not a different law. |
| `hypergeometric` | finite-population sampling counts `(ngood,nbad,nsample)` | Population and sample counts must be consistent; draws are bounded by available good items and sample size. |
| `integers`, `categorical` | discrete selection | Integer upper bound is exclusive. Categorical consumes a final category probability axis and outputs category indices, not a vector of counts; use normalized nonnegative probabilities. |
| `multinomial`, `dirichlet` | count vectors and simplex vectors | Core category axis retained; count vectors sum to n, simplex vectors sum to 1. Positive concentrations and normalized probabilities. |
| `multivariate_normal` | correlated vector-valued noise | Matching mean/covariance core axes; positive-definite covariance for Cholesky. `method` can select Cholesky/SVD/eigh; algorithm changes can change individual draws. |
| `choice`, `permutation` | resampling/selecting or shuffling values | `choice(replace=False)` produces dependent selected items and requires sufficient population; weighted selection requires probability length matching the first axis. `permutation` shuffles the first axis, retaining its values; it is not iid sampling. |

PyMC's `rv_op` / `rv_type` aliases expose low-level sampling kernels/classes.
Use them for inspection, not as replacements for model constructors, log
densities, transforms, or observed-data handling. PyMC rate, precision and
mean/dispersion parameterizations may be converted before reaching the kernel.

## Named-dimension random graphs

`pytensor.xtensor.random` provides the corresponding named-dimension wrappers,
with `xr.rng` / `xr.shared_rng` and generated methods on
`XRandomGeneratorVariable` / `XRandomGeneratorSharedVariable`. Their state chain
and explicit shared update contract match the positional interface. Parameters
are XTensorVariables with named dimensions; `extra_dims={"replicate": 4}` adds
batch dimensions and `core_dims` identifies support/parameter core dimensions.
Align by name and explicitly transpose before converting to positional `.values`.
This does not attach xarray coordinate labels or infer a scientific meaning for a
dimension name.

Named normal parameters broadcast by dimension; Dirichlet uses a category core.
Most named scalar constructors are positional `*params` wrappers, not
keyword-compatible copies of the positional signature. In PyTensor 3.3.0,
**named `gamma(shape, scale)` uses scale**, unlike the positional legacy
gamma helper's second argument. `standard_normal`, `chisquare`, and `rayleigh`
are composite wrappers; `rayleigh(scale)` scales a square root of a chi-square
draw. `nbinom` aliases `negative_binomial`. Check the namespace before assuming
named `choice` or `permutation` is available.

`as_xrv(core_op, core_inps_dims_map=None, core_out_dims_map=None, name=None)` is
an extension **factory**: it returns a constructor accepting `core_dims`,
`extra_dims`, `rng`, and `return_next_rng`. The maps identify which core dimension
names correspond to each input/output. A multivariate normal requires two core
names for covariance, one matching the mean/output axis; the specialized named
wrapper aligns the mean axis and accepts `method="cholesky"|"svd"|"eigh"`.
A missing or wrong number of core names is a model-shape error, not random noise.

Use module-level `as_xrv`, not `rng.as_xrv`: PyTensor 3.3.0's automatic method
population also wraps that factory, but injects unsupported RNG keywords.
See the linked named Generator source when checking another version.

PyMC dims adapters include a `TruncatedNormal` wrapper around PyMC's truncated
kernel; verify its declared bounds. `Flat` and `HalfFlat` are **improper**
densities without a random generator and raise `NotImplementedError`.
Do not invent finite uniform draws as a substitute. If simulation is needed,
formulate an appropriate proper distribution. Absence of a sampler does not
establish that a posterior using an improper prior is proper.

## Dynamic RandomStream compatibility protocol

In PyTensor 3.3.0, `RandomStream` emits a FutureWarning. Prefer the explicit
Generator API for new code. When maintaining an existing stream:

1. The constructor creates `SeedSequence(seed)`, stores `rng_ctor`,
   `default_instance_seed`, `namespaces`, and an initially empty `state_updates`.
   The default namespace is `pytensor.tensor.random.basic`; a supplied namespace
   **substitutes** its `__all__`, rather than extending the defaults.
2. Missing attribute lookup searches admitted names, wraps the namespace callable
   through `gen`, and caches the wrapper on that instance. Unknown names raise
   AttributeError. `__all__` membership is an interface admission rule, not proof
   that any arbitrary user callable satisfies the random-output protocol.
3. `gen` spawns one child SeedSequence for each constructed RV, creates its shared
   Generator, calls the random callable, records the RNG-to-next-RNG pair, and
   sets `default_update`. Its `rng=` argument is forbidden because the stream
   owns that state. Adding or reordering random nodes changes child assignments.
4. A compiled function applies applicable legacy default updates.
   `updates()` returns explicit pairs. Preserve explicit-update precedence;
   avoid the deprecated `no_default_updates` argument.
5. `seed(seed)` recreates the child states of existing nodes and resets the
   spawning sequence. `seed()` reuses the constructor's default seed; a default
   seed of None is not a reproducible reset to a fixed integer. It is not
   equivalent to setting every node to `default_rng(seed)`.

For a custom namespace, check admitted names, child-stream construction,
successive calls and replay with `seed`. A custom `rng_ctor` must consume the
child SeedSequence and return a compatible Generator. Reordering constructed
nodes changes child assignments even if the top-level seed is unchanged.

## Random Op inspection and extension boundary

`RandomVariable` is an `RNGConsumerOp`. Its symbolic contract is an input RNG,
size and distribution parameters, and output RNG plus draw. `update(node)`
provides the symbolic RNG mapping; `rng_param`, `size_param` and `dist_params`
provide role-aware access without hard-coded offsets. `signature`, `inputs_sig`,
`output_sig`, `ndim_supp` and `ndims_params` distinguish batch from support;
`batch_ndim(node)` and `infer_shape` describe shape, not sampling evidence.
When defining a new kernel, use a gufunc-like `signature` rather than deprecated
constructor `ndim_supp` / `ndims_params` arguments.

`__call__` builds a node via `make_node`; the Python `perform` calls `rng_fn` and
casts the result to the declared dtype. `inplace` controls copying versus
mutation of the incoming state; the compiler manages legal destructive reuse.
Do not flip `destroy_map` or force in-place Ops to cure a stale update.
`name` and `dtype` describe the Op, not a model variable's probabilistic meaning.
RNG variables' `owner_op` / `owner_op_and_inputs` are graph-inspection shortcuts;
a root/shared state has no creating Apply node. Walk the returned state chain
when debugging; don't expect root state ownership to identify the draw.

In the linked PyTensor 3.3.0 source, raw RandomVariable `pullback` reports
undefined gradients and `pushforward` returns disconnected tangents.
Random sampling is not automatically a differentiable reparameterization or a log density.
Use an explicit deterministic transform of base noise or an appropriate custom
probabilistic/derivative implementation when that is the intended mathematics;
see the differentiation/custom-Op workflow for its own derivative verification.

Supply correct state graphs and let the compilation mode apply random inplace
and unused-consumer rewrites. Do not invoke compiler passes as user RNG controls.

## Backend and reproducibility checks

- Within each backend, check state progression and exact replay after resetting
  the initial state. Preserve deterministic transforms of each draw.
- Cross-backend draws need not be bitwise equal: algorithms, vectorization and
  consumption order can differ. Compare appropriate support/distributional
  properties, not state dictionaries from incompatible RNG representations.
- Check immutable-input preservation, missing/ambiguous initialization,
  selection-without-replacement and permutation invariants, requested
  dtype/shape, count sums and simplex sums. Heavy tails may not have finite
  moments; there is no universal sample-mean check.
- Numba has per-Op dispatch; in the linked 3.3.0 source HyperGeometric is
  explicitly unimplemented. A Python/SciPy draw does not establish Numba
  support. Inspect fallback warnings and nested graphs before claiming native
  execution, including when using PyMC distribution wrappers.
- Named XRV wrappers may require optimization before execution; do not assume
  an unoptimized Python linker can run them directly.
- JAX converts Generator state to keys and splits those keys. JIT random
  generation requires supported static or shape-derived sizes, and some
  families require NumPyro. Check installed dispatch and dependencies.
- PyTorch/MLX support and device precision must be established for the actual
  graph; backend names alone do not establish random-kernel compatibility.

## Authoritative sources

These pinned sources describe PyTensor 3.3.0 behavior; consult the installed
version when an API or support limitation differs:

- [Released Generator variables and method generation](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/random/variable.py)
- [Released random kernels and parameterizations](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/random/basic.py)
- [Released RandomStream and shape utilities](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/random/utils.py)
- [Released RandomVariable state/shape/derivative contract](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/random/op.py)
- [Released named constructors](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/xtensor/random/basic.py) and [named Generator methods](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/xtensor/random/variable.py)
- [Released rewrite imports/passes](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/random/rewriting/basic.py)
- [Numba random dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/numba/dispatch/random.py) and [JAX random dispatch](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/link/jax/dispatch/random.py)
- [PyMC 6.3.1 named distribution adapters](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/dims/distributions/scalar.py) and [proper/improper scalar kernels](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/continuous.py)
- [NumPy Generator](https://numpy.org/doc/stable/reference/random/generator.html), [SeedSequence](https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html), and [compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html): current explanatory references, not evidence of cross-version bitwise guarantees.

