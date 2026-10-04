# Shapes, dimensions, and structured symbolic computation

## Define the calculation

Use this workflow to express a model's **deterministic computation**. Write the
observation unit, predictor units, group-code mapping, time ordering, and intended
output axes before building a graph.

API cautions below refer to PyTensor 3.3.0 and PyMC 6.3.1; check the linked
official source against the installed version. `pymc.dims` and
`pytensor.xtensor` are experimental in those versions.

## Choose the representation

| User decision | Recommended representation | Invariant to establish |
|---|---|---|
| Dense design matrix, group indexing, ordinary likelihood inputs | `pytensor.tensor` (`pt`) | Explicit positional axes and dtypes |
| Alignment by dimension name materially clarifies the expression | `pytensor.xtensor` (`px`), with `pymc.dims.math` for named math | Unique names; equal sizes for a shared dimension; explicit conversion boundaries |
| Same computation for independent scenarios/observations | Broadcasting, batched contractions, or `pt.vectorize` | Distinguish batch axes from core axes |
| State depends on previous state | `pytensor.scan` | Step order, initial-state shape/dtype, sequence alignment |
| Elementwise masking | `pt.where` / `pt.switch` | Both branch expressions are valid over their evaluation domains |
| Scalar choice of a whole expression, including expensive/invalid unselected branch | `pytensor.ifelse` with a lazy VM/CVM linker | Branch count, rank and dtype agree; verify actual laziness |
| Ragged blocks with one element type/rank | `pytensor.typed_list.TypedListType` | Homogeneous element type; ragged lengths do not mean mixed dtypes/ranks |

Prefer direct array operations over a Python loop that builds thousands of graph
nodes. Do not replace a genuinely sequential recurrence by independent mapping.
Do not claim vectorization always accelerates a graph: graph construction,
compilation, memory traffic, and the selected backend must be measured separately.

## Types, construction, and shape contracts

`pt.tensor(name, dtype="float64", shape=(None, 3))` declares rank two and three
columns. `None` means an unknown length, not a missing observation. A known
length-one axis is broadcastable. `pt.vector`, `matrix`, `tensor3` through
`tensor7`, `row`, and `col` are conveniences. Typed prefixes select dtypes:
`b/w/i/l` mean signed 8/16/32/64-bit integers, `f/d` float32/64, and `c/z`
complex64/128. In particular, **`bvector` is not boolean**; request
`dtype="bool"`. Plural constructors create several independent input variables.
`TensorLike` is a typing annotation, not a numeric constructor.

Use `as_tensor_variable` / `as_tensor`, `constant`, or `as_symbolic` at explicit
Python/NumPy-to-symbolic boundaries. `as_symbolic` also permits supported
non-dense values; the exposed `as_sparse_or_tensor_variable` alias does not imply
conversion of all inputs to a sparse matrix. Construct design arrays with
`stack`, `ones_like`, `arange`, grids, or array creation (`zeros`, `ones`, `full`,
`eye`, `identity_like`). `empty` and `empty_like` contain uninitialized values:
only use them if every element is populated before it can affect an output.
`alloc` broadcasts a value into a shape; it is not random initialization.
`fill`/`second` carry a value over another tensor's shape. `pi`, `e`,
`euler_gamma`, `inf`, `nan`, and `newaxis` are constants, not fitted quantities.

Distinguish `x.type.shape` (static information) from `x.shape` (symbolic runtime
lengths). `ndim`, `dtype`, `broadcastable`, `numpy_dtype`, and `size` answer
different questions. `get_vector_length` can return a Python length only when
the graph makes it inferable; otherwise it raises. Scalar-constant extraction
helpers likewise must not be treated as evaluation of an arbitrary graph.
`shape_of_variables`/static-shape and normalized-axis utilities are for inspecting
shape relations, not proving the numeric expression is valid. A shape-only graph
may optimize away the very numerical operation whose error you wanted to check.

`specify_shape(x, expected_shape)` establishes/checks a runtime shape contract;
`specify_broadcastable(x, axis)` requires the indicated axis to have length one.
Neither fixes incorrectly aligned scientific data. `astype`/`cast` change values'
representation; they are not permission to truncate real-valued group labels to
integers. `TensorType.filter` and related conversion methods implement input
acceptance, including rank/static length and unsafe-downcast restrictions.
`values_eq`/`values_eq_approx` serve type-level comparisons, not posterior checks.
Use `pt.eq` for elementwise symbolic equality, not object identity or constant
hash/signature comparison. `copy`/`tensor_copy`/`identity` preserve values; do not
infer that all optimized intermediates are independent memory allocations.

Inspect a variable's `owner`, `owner_op`, and `owner_op_and_inputs` when tracing
which expression produced it; leaf inputs have no producing Apply node. The
inherited `OpFromGraph` inner-input/output and function properties describe an
encapsulated graph, not an extra array API. See the custom-Op and compilation
references for extension and graph transformation contracts.

Advanced scalar extensions can use `Elemwise(scalar_op)` and `CAReduce`.
`Elemwise.outer(x, y)` expands both operands and applies a binary scalar kernel:
`pt.add.outer(a, b)` is an outer **sum**, unlike `pt.outer(a, b)`'s product.
Unary kernels reject `.outer`. `CAReduce` requires a commutative, associative
single-output binary kernel because iteration order is unspecified; subtraction
and division are invalid choices. Specify reduction axes and accumulator/output
dtypes separately. `DimShuffle(input_ndim=..., new_order=...)` constructs an
axis transformation; ordinary model authors should use variable `dimshuffle`.
These constructors do not automatically make custom scalar kernels
backend-compatible.

## Design matrices, indexing, and broadcasting

For `N` observations, `P` features, and `G` groups:

- `X`: `(N, P)`, floating predictors with declared units/scaling;
- `beta`: `(P,)`, compatible floating dtype;
- `group_idx`: `(N,)`, integer, one consistent mapping into `[0, G)`;
- `alpha`: `(G,)`; `X @ beta + alpha[group_idx]`: `(N,)`.

```python
import pytensor.tensor as pt

X = pt.matrix("X", dtype="float64")          # (observation, feature)
beta = pt.vector("beta", dtype="float64")    # (feature,)
alpha = pt.vector("alpha", dtype="float64")  # (group,)
group_idx = pt.vector("group_idx", dtype="int64")
eta = X @ beta + alpha[group_idx]           # (observation,)
```

Check the concrete group mapping and feature ordering before evaluation;
equal array lengths alone cannot establish alignment.

Build intercept, linear and quadratic columns symbolically and compare the
predictor against NumPy. Reject negative group codes when they are invalid
for the model. Raw NumPy/PyTensor indexing supports
negative indices; a missing-group code of `-1` otherwise silently selects the
last group. Preserve that distinction rather than globally outlawing Python
index semantics. Out-of-range positive indices and floating index arrays must
fail; never clip or cast them to conceal an error.

`take`, `take_along_axis`, `choose`, basic slices, masks, `compress`, `nonzero`,
`flatnonzero`, `flip`, and row permutations select different axes/positions.
Check whether advanced indices pair elementwise or create an outer product;
insert singleton axes explicitly when an outer product is intended. Boolean
selection has data-dependent output length. That may be acceptable on CPU but
conflict with a compiler requiring static output sizes. `ravel_multi_index` and
`unravel_index` require a matching shape/order and explicit out-of-range policy.
`to_one_hot`/`bincount` need valid integer categories and an intentional class
count. One-hot matrices can allocate much more than direct group gathering.

`x[idx].set(y)` / `set_subtensor` and `.inc(y)` / `inc_subtensor` return symbolic
updated arrays; they are not Python assignment into `x`. With repeated indices,
`inc_subtensor(..., ignore_duplicates=False)` accumulates every occurrence like
`np.add.at`; advanced-indexed NumPy `x[idx] += y` has different repeated-index
semantics. Do not request `ignore_duplicates=True` or assume unique indices
without proving the index property. Repeated overwriting and repeated
accumulation are different operations.

Broadcasting is right-aligned by positional axis. For scenario coefficients
`B: (S, P)`, a prediction surface is `(S, N)`, not an accidental `(N, P)`:
`B @ X.T`, or vectorized row/core contraction with batch shapes `(1, N)` and
`(S, 1)`. `pt.vectorize(function, signature="(p),(p)->()")` creates a batched
**symbolic graph**, unlike NumPy's general Python-loop convenience wrapper.
In PyTensor 3.3.0, `vectorize` does not assert equality of core lengths from
the signature. Add an explicit shape contract; the signature is not an input
validator. Inspect backend lowering when an Op cannot be vectorized rather than
silently changing the scientific calculation.

`broadcast_arrays`, `broadcast_to`, `broadcast_shape`, and `broadcast_shape_iter`
make intended broadcasting explicit. `broadcast_to` cannot make incompatible
lengths agree. `reshape`, `ravel`/`flatten`, `dimshuffle`, `transpose`/`mT`,
`swapaxes`, `moveaxis`, `expand_dims`, `atleast_*`, `squeeze`, and `shape_pad*`
change axis layout, not coordinate identity. `dimshuffle` can add singleton axes
and remove only eligible singleton axes. Removing an axis is not summing over it.

The newer `join_dims` / `split_dims` preserve consecutive-axis structure;
`pack` / `unpack` retain shape descriptors for reversible parameter-block
packing. Pass the same `keep_axes` to both; preserved axes must agree across
inputs. Not every arbitrary axis subset is admissible: positive kept axes must
be contiguous from zero and negative kept axes contiguous to the right edge.
Check both reshape and pack/unpack round trips.

`concatenate`/`join` extend an existing axis; `stack`/`stacklists` introduce a new
one. Horizontal/vertical stacking follow rank-specific convenience conventions.
`repeat` repeats elements; `tile` repeats blocks. `roll` wraps, `diff` shortens an
axis, and cumulative operations preserve its length. `pad` supports constant,
edge, ramp, statistical, wrap, symmetric and reflect modes in the inspected
signature: the chosen mode encodes a boundary assumption, not a neutral fix for
misaligned time series. Check exact keyword compatibility for that mode.

## Arithmetic and transcendental math

`pymc.math` mostly exposes positional PyTensor operators; `pymc.dims.math`
re-exports named `xtensor.math` operations. Names that happen to match NumPy are
not permission to call a NumPy function on a symbolic graph. Method forms on
`TensorVariable`/`TensorConstant` and facade aliases usually express the same
operator.

| Family | User decision and boundary |
|---|---|
| `add`, `sub`/`subtract`, `mul`/`multiply`, `neg`/`negative`, division, reciprocal, power/square/sqrt | Check units, integer vs real division, overflow and support. A real square root of a negative argument is invalid; casting is not a scientific repair. |
| `abs`, `sign`/`sgn`, `clip`, elementwise minimum/maximum, rounding/truncation | Nondifferentiable boundaries matter for gradients. `round_half_to_even` and `round_half_away_from_zero` encode different tie policies. Reducing `min/max` is not elementwise `minimum/maximum`. |
| `exp`, `exp2`, `log`, `log2`, `log10` | Check log support and overflow. Large finite logits may require a stable link rather than raw exponentiation. |
| `expm1`, `log1p`, `softplus`/`log1pexp`, `sigmoid`/`expit`/`invlogit` | Use dedicated stable forms near zero or in log-space. Compare to NumPy `logaddexp` and SciPy `expit`; choose tolerances for the magnitudes and dtypes. |
| trig/inverse trig, hyperbolic/inverse hyperbolic, `deg2rad`/`rad2deg` | State angle units and real-domain restrictions; radians are not degrees. |
| `logit`, `probit`, `invprobit`, `logdiffexp` | Probabilities lie in `(0,1)` for finite inverse links; `logdiffexp(a,b)` needs `a >= b` for a real log difference. Equality permits `-inf`, not an arbitrary finite value. |
| `complex`, `complex_from_polar`, `conj`/`conjugate`, real/imaginary parts, `angle` | Preserve complex dtype and distinguish conjugation from transpose. Complex arithmetic support does not imply a real-valued differentiable likelihood. |

**Constant precision:** PyMC 6.3.1's `probit` and `invprobit` construct
`sqrt(2.0)` from a Python constant. Under PyTensor's `custom` cast policy,
this intermediate can be float32 even with float64 inputs. If precision
depends on it, inspect the graph's constants and use an explicit local policy
when constructing it:

```python
with pytensor.config.change_flags(cast_policy="numpy+floatX", floatX="float64"):
    transformed = pm.math.probit(probability)
```

Compare with an independent numerical reference; do not loosen tolerances to
conceal unintended precision loss. See the version-specific
[PyMC math source](https://github.com/pymc-devs/pymc/blob/da8fc472989122c0441e03789fba9f36ffcd9a98/pymc/math.py).

**Log-complement sign:** that source's `log1mexp` body delegates to
`pt.log1mexp(x)`, meaning `log(1-exp(x))`, despite opposite-sign docstring
wording. Use nonpositive arguments; for strictly negative values compare
against `np.log(-np.expm1(x))`. Check installed behavior before using legacy
polarity flags; do not negate twice.

### Special functions

Use SciPy's corresponding special function as an independent reference,
checking its parameter convention:

- Gamma/beta families: `gamma`, `gammaln`, digamma/`psi`, `polygamma`/`tri_gamma`,
  regularized incomplete gamma/complement/inverses, `gammal`/`gammau` (unregularized
  forms), incomplete beta/inverse, and `chi2sf`. Shape parameters and integration
  limits have support restrictions; poles and complements are not interchangeable.
- Error/normal-tail families: `erf`, `erfc`, `erfcx`, inverse error functions,
  `ndtri_exp`, and `owens_t`. Scaled or log-input forms exist to address tail
  cancellation/underflow; an ordinary inverse-CDF reference can itself underflow.
- Bessel families: `i0/i1/iv/ive`, `j0/j1/jv`, `kn/kv/kve`, plus `hyp2f1`.
  Order, positive-real argument restrictions of modified Bessel K, scaling,
  branches, and parameter singularities require function-specific checks.

Check relevant orders, tails, domains, derivatives and compiler lowerings.
A scalar C or Python implementation does not guarantee JAX/MLX/Torch support.

`pm.math.logbern(log_p, rng=...)` is an **eager NumPy random decision**, not a
symbolic random variable or a distribution log density. Its input is a log
probability, not a logit. Explicitly supply an RNG for reproducibility; NaN raises.
Do not use it inside graph construction expecting a new decision at each graph
execution. Its endpoint/error boundaries can be checked without inference.

## Reduction and statistics

Choose the observation/reduction axis before selecting a function. `sum`,
`prod`, `mean`, `var`, `std`, `median`, `ptp`, extrema and `argmin/argmax` have
shape-changing semantics. `keepdims=True` preserves a reduced singleton axis
for centering/normalization; omitting it may silently align against the wrong
axis when lengths coincide. Accumulator dtype can differ from input dtype;
integer reductions can overflow. A zero-length sum/product has an identity,
whereas an empty mean, maximum or variance is not an interchangeable operation.

For variance/covariance specify `ddof` and ensure `N > ddof`. For an
`(observations, features)` matrix, `rowvar=False` treats columns as variables.
A posterior sample covariance is not an observational covariance model.
`norm` requires an intentional order/axis. `logsumexp` is a stable reduction; `softmax` must normalize
over the category axis (or named `dim`), not the observation axis. Cumulative
sum/product and differencing retain temporal order; they are not exchangeable
summaries. `max_and_argmax` returns a value/index pair, not two independent
latent variables. `sort`/`argsort`, `unique` and inverse/count outputs can support
preprocessing, but sorting time-dependent observations destroys time alignment.
`searchsorted` requires ordered input and an intentional left/right tie policy.

## Comparison and boolean math

Use `pt.eq`/`neq`, ordered comparisons, `isfinite`/`isnan`/infinity predicates,
`isclose`/`allclose`, logical operations and `all`/`any`. Python `and`, `or`,
chained comparisons, and `if symbolic_condition` demand concrete truth values
and cannot express elementwise graph decisions. Parenthesize symbolic
comparisons around `&`/`|`. Bitwise operators on integers manipulate bits, not
merely truth; `bvector` is signed int8. Match named logical/bitwise aliases to
the desired dtype.

`nan_to_num` is an explicit numerical transformation. It is not a general fix
for invalid support, uninitialized tensors, a broken gradient, or divergence.
Finite predicates help diagnose invalid inputs; they do not define missing-data
mechanisms. Check boolean output dtypes and symbolic truth-value misuse.

## Contractions and structured arrays

For two-dimensional matrix-vector multiplication use `dot`/`matmul` as appropriate.
For batched work, `matmul`, `matvec`, `vecmat`, `vecdot`, `tensordot` and `einsum`
have different core-axis and conjugation conventions; write the intended equation
and compare to a matching NumPy reference. `dense_dot` is not a sparse-format
preservation promise. `outer` introduces axes rather than contracting them.
Use the numerical reference for solver/determinant/inverse/matrix-exponential
conditioning and dtype restrictions; an exposed `det`/`inv` alias through math
or a tensor method does not add a new solver guarantee.

`cartesian` materializes a **NumPy** grid with earlier arrays varying more slowly;
it is suitable for a fixed prediction design, not symbolic runtime mesh creation.
`mgrid`/`ogrid`, `linspace`, `geomspace` and `logspace` have different endpoint,
spacing and broadcasting conventions. `bartlett` creates a taper/window; it does
not fit a temporal covariance or resolve edge bias.

`diag`/`diagonal`, `trace`, triangular constructors/masks/indices, diagonal fill,
`batched_diag`, and `expand_packed_triangular` encode specific storage/order
conventions. `expand_packed_triangular(n, packed)` expects `n*(n+1)//2` entries
for a full triangle; it does not make a matrix positive definite.
`batched_diag` turns rank-two diagonals into rank-three matrices or extracts
rank-three diagonals; other ranks raise. `flatten_list` concatenates flattened
blocks and discards their shapes; `pack` retains descriptors for a round trip.

`kronecker`/`kron`, `flat_outer`, `kron_diag`, and `kron_matrix_op` support separable
structures. `kron_dot` and triangular `kron_solve_lower/upper` avoid explicitly
materializing the full Kronecker matrix for their supported inputs; validate the
factor order and solve orientation. In PyMC 6.3.1 a vector right-hand side
is treated as a one-column matrix, retaining output shape `(N,1)`. Compare
against a small explicit NumPy Kronecker reference before scaling up.

## Scan and conditional behavior

`scan` step arguments are sequences, recurrent outputs, then non-sequences.
`outputs_info` declares initial states; their dtype/rank must match recurrent
outputs. Use typed constants instead of relying on integer literal inference.
Pass non-sequences explicitly and use `strict=True` when that makes graph
capture errors detectable. A sequence supplies a leading time axis; multiple
sequences otherwise stop at the shortest length, so assert equality for aligned
scientific series. Taps describe actual lag dependencies and must agree with
initial-history length. `go_backwards` reverses iteration order; it does not
relabel forecast times. `truncate_gradient` changes derivative propagation and
therefore can change inference; do not use it as an invisible speed fix.

`map` produces all independent step outputs. `reduce`/`foldl` return final
accumulators; `foldr` iterates in reverse. Use an order-sensitive recurrence to
check the direction against a Python loop. `until(condition)` stops **after
including the triggering step**. If empty sequences are allowed, specify their
output convention: `states[-1]` is not valid for one.

`scan_checkpoints` trades recomputation for saved intermediates during gradients;
it is not a generic drop-in Scan. The source documents equal-length sequences,
singly recurrent/nonrecurrent outputs and consumption of the last step only.
Tap dictionaries are explicitly rejected in PyTensor 3.3.0. Check the final
state and its gradient against ordinary Scan before claiming memory savings.

`where`/`switch` select elementwise and can evaluate both expressions. A mask
around an invalid indexed expression or division is not lazy protection.
`ifelse` has a scalar condition and matching branch outputs; its laziness
requires a suitable VM/CVM linker. For example, with
`Mode(linker="vm", optimizer="fast_compile")`, an invalid index in an
unselected branch should remain unevaluated and should raise when selected.
Check this on the intended linker rather than silently falling back.

## Named dimensions and homogeneous lists

Ordinary `pm.Model` coords/dims label variables and may determine distribution
shapes, but **do not change positional tensor algebra into name-aligned algebra**.
Named `XTensorVariable` operations align by dimension name instead. Construct
with `px.xtensor(..., dims=(...))` or `px.as_xtensor(array, dims=(...))` and
verify sizes. The inspected facade exports `px.math.softmax`, not `px.softmax`.
`pymc.dims.math` re-exports that named-math module.

Use `dot(..., dim="feature")` for an explicit contraction; `isel(group=index)`
for named positional gathering; `rename` before an intentional outer operation;
`transpose` for output ordering; `broadcast_like`/`broadcast`, `concat`,
`expand_dims`, `squeeze`, `head`/`tail`/`thin`, and named reductions as appropriate.
Compare named gathering, outer operations, softmax and stack/unstack with
positional references. `unstack` requires sizes because coordinates are absent;
the reshape order is C-style.

`.values` returns a positional **symbolic TensorVariable**, not an evaluated
NumPy array. In PyTensor 3.3.0, `.as_numpy` is a no-op, not execution; `.sel`,
`.loc`, and `.coords` raise `NotImplementedError`. `dims` are names and `sizes`
are symbolic lengths. Do not assume label lookup, coordinate reindexing, or
full xarray compatibility; see the XTensor source below.
`isel` with an unknown dimension raises unless a different `missing_dims` policy
is intentionally requested; do not hide misspelled dimension names with ignore.

Typed lists permit vectors of unequal lengths but the same TensorType. Use
`.length()` instead of Python `len`, indexing/slicing for selection, and explicit
returned results from append/extend/insert/remove/reverse. `index`/`count`
compare list elements, not tensor coordinates. Check ragged vector blocks
without admitting a matrix into a vector list. Type matching, aliasing and
ownership are real contracts; avoid inplace operations merely for convenience.
C implementations do not imply accelerator support.

## Distribution shape interfaces and structural assumptions

`shape` includes batch and support axes; distribution `size` describes batch
replication. `change_dist_size(dist, new_size, expand=True)` prepends batch
sizes; it returns a new distribution rather than reshaping realized samples.
`to_tuple` normalizes a shape representation; `rv_size_is_none` recognizes an
unspecified size. Check expansion structurally, not by reshaping realized draws.

Advanced distribution authors may encounter `convert_dims`, the ellipsis-aware
variant, `convert_shape`, `convert_size`, `shape_from_dims`, `find_size`,
`get_support_shape`/`get_support_shape_1d`, `implicit_size_from_params`, and
`maybe_resize`. These normalize dimension arguments, infer support/batch sizes,
and broadcast log-density/moment results; they do not supply missing scientific
shape information. `get_support_shape` adds symbolic consistency assertions when
explicit and inferred support shapes are supplied. Prefer the public distribution
constructor for ordinary models; preserve `ndim_supp`/parameter-core semantics
when extending distributions. `change_rv_size`/`change_specify_shape_size` are
registered implementation handlers, not alternate user resize entry points.

`from pytensor.assumptions import assume` attaches structural facts such as
triangular, symmetric, positive definite, orthogonal, permutation, selection, or
unique indices. It is a runtime no-op annotation for rewrites, **not a validator**.
Incorrect assumptions can justify incorrect rewrites. Establish the property
from construction or check the concrete input first. Unique indices must account
for negative aliases (`-1` and `n-1` identify the same position). Establish
uniqueness before adding the annotation.
`ConflictingAssumptionsError` diagnoses contradictory provable/declarative facts,
not a universal data-validation mechanism.

## Import and backend boundaries

Use `matmul`, `tensordot` or `vectorize` for an explicit batch equation rather
than relying on old `batched_dot`/`batched_tensordot` documentation. Check the
installed facade; do not invent compatibility aliases. Import `as_symbolic`
from `pytensor.basic`, Scan views from `pytensor.scan.views`, and printers from
the printing API when the root module does not re-export them.

Built-in C kernels, inplace rewrite registrations, dtype/ABI registries and
compiler storage bookkeeping are not alternative modeling operations. Use
tensor functions and public mode selection; extend a custom Op only when the
desired computation genuinely lacks a supported primitive.

`transfer(var, target)` only supports registered transfer targets; the inspected
implementation recognizes CPU and otherwise consults a registry. It is not an
API for magically enabling GPU/JAX/MLX/Torch. Treat backend-specific targets as
unavailable until an appropriate implementation/environment is actually present;
never silently replace the target by CPU.

## Check the shape contract

Compare numerical values **and exact output shapes/dtypes** over representative
arrays. Include mismatched group alignment, invalid indices, unsafe downcasts,
unequal vectorized core lengths, Scan state dtype mismatches, selected/unselected
lazy branches, duplicate/unknown dimension names, and wrong list element ranks.
Use dtype- and scale-appropriate tolerances. A shape-only evaluation can remove
the numerical Op, so evaluate values as well.

## Authoritative sources

Read the matching installed source when a docstring and implementation disagree.
The pinned links below describe version-specific behavior, not future guarantees.

- [Released tensor creation/math reference](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/doc/library/tensor/basic.rst),
  [live tensor guide](https://pytensor.readthedocs.io/en/latest/library/tensor/basic.html).
- [Released tensor math](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/math.py),
  [PyMC math](https://github.com/pymc-devs/pymc/blob/da8fc472989122c0441e03789fba9f36ffcd9a98/pymc/math.py).
- [Reshape/pack source](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/reshape.py),
  [vectorize contract](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/functional.py),
  [indexing/update source](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/subtensor.py).
- [Scan guide](https://pytensor.readthedocs.io/en/latest/library/scan.html),
  [checkpoint restrictions](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/scan/checkpoints.py),
  [IfElse linker contract](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/ifelse.py).
- [PyMC dims guide](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dims_module.html),
  [XTensor type/selection source](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/xtensor/type.py),
  [shape utilities](https://github.com/pymc-devs/pymc/blob/da8fc472989122c0441e03789fba9f36ffcd9a98/pymc/distributions/shape_utils.py).
- [Structural assumptions](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/assumptions/specify.py),
  [typed-list operations](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/typed_list/basic.py).
- [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html),
  [NumPy advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html),
  [SciPy special functions](https://docs.scipy.org/doc/scipy/reference/special.html).
