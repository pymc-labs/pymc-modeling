# Numerical, sparse, and spectral computations

Use this reference when a model needs a linear solve, covariance calculation,
sparse design matrix, stable special function, implicit equation, interpolation,
or spectral/filter operation. First state the mathematical quantity, input units,
core and batch dimensions, dtype, domain, and a reference identity.

## API and numerical boundaries

Version-specific cautions below refer to PyTensor 3.3.0; use the linked official
source to check the installed version. Availability of an operation and
availability of its gradient or backend lowering are separate questions.
Choose an independent NumPy/SciPy reference, equation residual, or reconstruction
identity before trusting numerical output.

## Dense solves and Gaussian calculations

Prefer `pytensor.tensor.linalg` (`la`) to the deprecated `nlinalg`/`slinalg`
import facades. PyMC's `math` aliases share the underlying numerical operations.

For an SPD covariance `K` and residual `r`, factor once:

```python
L = la.cholesky(K, lower=True)
z = la.solve_triangular(L, r, lower=True, b_ndim=1)
logdet_K = 2 * pt.log(pt.diagonal(L)).sum()
log_density = -0.5 * (r.size * np.log(2 * np.pi) + logdet_K + z @ z)
alpha = la.cho_solve((L, True), r, b_ndim=1)
```

This computes the quadratic form and log determinant without explicitly forming
`K**-1`. Compare with `la.solve(K, r)` and check the residual derivative
`-solve(K, r)`; covariance-hyperparameter derivatives need separate checks.
The Cholesky input must be positive definite, not merely semidefinite.
Adding diagonal jitter changes the covariance: justify its scale and assess its
impact; do not keep increasing it to conceal a wrong kernel or duplicated data.

| Family and interface choices | Model-author restrictions |
|---|---|
| `solve`, `solve_triangular`, `cho_solve`; `Solve`, `SolveTriangular`, `CholeskySolve` core Ops | Use `assume_a` only when symmetry, positive definiteness, or band structure is guaranteed. An assumption can cause a solver to ignore entries. For triangular solves select `lower`, transpose, and unit diagonal deliberately. |
| `lu`, `lu_factor`, `lu_solve`, `pivot_to_permutation`; LU-related core Ops | General nonsymmetric systems; reuse a factor across multiple RHS. Pivot vectors encode row swaps, not automatically a final permutation. Verify `A = P L U` or a residual rather than comparing pivots across implementations. |
| `tridiagonal_lu_factor`, `tridiagonal_lu_solve` | Factor the full matrix and pass the five arrays (`dl, d, du, du2, ipiv`) to the solve. Assert the matrix is tridiagonal; ignored off-band entries alter the equation. Compare with a general solve. |
| `slogdet`, `det`, `trace`, `norm` and their Ops | `slogdet` supplies a sign and log absolute determinant; sign matters for non-SPD matrices. A determinant is not a stable route to its logarithm. Choose vector/matrix norm and axes explicitly; a small residual does not alone guarantee small forward error in an ill-conditioned system. |
| `inv`/`matrix_inverse`, `pinv`, `tensorinv`, `tensorsolve` | Inverse output is occasionally the actual target, not a substitute for a solve. Pseudoinverse tolerance changes effective rank. Tensor equations require matching contracted dimensions; PyTensor 3.3.0's `TensorSolve` declares a matrix output, so check output rank instead of assuming arbitrary tensor support. |

A batched `A` has shape `(..., n, n)`. For a batch of vector RHS with shape
`(..., n)`, set `b_ndim=1`; a two-dimensional RHS otherwise defaults to matrix
core rank. For multiple RHS set `b_ndim=2`. Check a nontrivial batch, not only the
unbatched case.

PyTensor 3.3.0's dense solver policy maps inputs to LAPACK float32/float64 or
complex64/complex128. Small integers and float16 may promote; other Ops have
their own output rules. Check actual dtypes, reject unintended downcasts with
`allow_input_downcast=False`, and choose tolerances accordingly. Complex
forward execution does not establish complex differentiation.

In that version, several wrappers accept but ignore `check_finite`, and failed
Cholesky or singular solves may return NaNs rather than exceptions. Check
finite outputs and equation residuals; do not use a keyword as an input validator.
Nonsquare shapes and non-positive-definite values are different error cases.

Sources: [general solves](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/solvers/general.py),
[triangular](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/solvers/triangular.py),
[Cholesky](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/decomposition/cholesky.py),
[dtype policy](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/dtype_utils.py),
[least squares/tensor solve](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/solvers/lstsq.py).

## Decompositions, matrix functions, and structured systems

Use QR for least squares without squaring the condition number in normal
equations; use reduced/economic output unless a full basis is needed. Pivoted QR
and SVD help diagnose rank. PyTensor 3.3.0's `lstsq` requires an explicit `rcond`
and declares a float64 matrix solution; check its current RHS/output contract.
Rank-deficient coefficients are not individually identified merely because a
pseudoinverse returns a value. Check projection and reconstruction identities.

`eigh`/`eigvalsh` exploit symmetric or Hermitian structure; `eig` is for a general
matrix and may yield complex results. Eigenvectors and singular vectors have
sign/phase ambiguity and can rotate within a repeated-eigenvalue subspace.
Compare reconstructed matrices, orthogonality, residuals, and spectra, not raw
vector coordinates. Derivatives involving repeated eigenvalues/singular values
may be singular or nonunique; finite forward output is insufficient for HMC.

`schur`, `qz`, and `ordqz` express ordinary/generalized invariant subspaces.
`ordqz` is a wrapper around `qz`, with return conventions to inspect rather than
assuming SciPy's exact tuple. Select real versus complex form and sorting domain
(e.g. inside the unit circle) explicitly. These are useful for equilibrium
systems; lack of a separated stable subspace is not an optimization nuisance.
Check Schur and generalized Schur reconstruction.

`expm` is the matrix exponential, not elementwise `pt.exp`; it propagates a
linear continuous-time system. `matrix_power` uses integer powers and
`matrix_dot` composes products. `block_diag` and `kron` express block-independent
or separable structure, but these are dense constructors: materializing a large
Kronecker matrix can defeat the purpose of separability. Reorganize contractions
only after establishing vectorization order and matching the intended model.

### Covariance/control equations: check signs from implementations

For the linked PyTensor 3.3.0 implementations, the residual contracts are:

- `solve_sylvester(A, B, Q)`: `A @ X + X @ B = Q`.
- `solve_continuous_lyapunov(A, Q)`: `A @ X + X @ A.conj().T = Q`.
  For stationary covariance with stable continuous-time drift and positive
  innovation covariance `Q_noise`, supply `-Q_noise`.
- `solve_discrete_lyapunov(A, Q)`: `X - A @ X @ A.conj().T = Q`.
  A stationary covariance with positive `Q` requires stable transition dynamics.
  `direct` constructs a larger Kronecker system; `bilinear` uses a transformed
  continuous equation and becomes delicate near problematic eigenvalues such as
  `-1`. There is no hardware-independent crossover-size claim.
- `solve_discrete_are(A, B, Q, R)`: check the Riccati equation and closed-loop
  stability, not only agreement with another implementation. Stabilizability,
  detectability, symmetry, and positive-definiteness conditions belong to the
  scientific formulation; not every input admits the desired solution.

The linked version's Lyapunov docstrings disagree with these implementation
signs. Check residuals and the installed source rather than copying a sign.
An algebraic solution with a positive drift matrix does not establish a
stationary covariance.

Sources: [linalg facade](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/__init__.py),
[eigen](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/decomposition/eigen.py),
[Schur/QZ](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/decomposition/schur.py),
[matrix products](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/products.py),
[control equations](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/linalg/solvers/linear_control.py).

## Sparse storage and arithmetic

The sparse interface is based on SciPy **sparse matrices**, not a promise of
support for the newer sparse-array API or arbitrary-rank sparse tensors.
`SparseTensorType` supports matrix formats CSR, CSC, and BSR at the type level;
individual operations often support only CSR/CSC. For a sparse design matrix,
keep the data sparse through the matrix product and accept a dense predictor.
Sparse shape alone does not establish sparsity of intermediate results.

| Family | Semantics and choice |
|---|---|
| `csr_matrix`, `csc_matrix`, `bsr_matrix`, `matrix`, `*_fmatrix`/`*_dmatrix`, `as_sparse_variable`, `shared` | Specify format and dtype. The shared constructor accepts `strict`, `allow_downcast`, and `borrow`; borrowed storage requires an explicit mutation/aliasing policy. Dense-to-sparse conversion costs memory/time and does not make a dense problem inherently sparse. |
| `CSR`, `CSC`, `CSM`, `csm_properties`, `csm_data/indices/indptr/shape` | Compressed storage separates differentiable data from integer structural indices. CSR indptr partitions rows, CSC partitions columns. Indices/indptr must be consistent with shape, monotone pointer structure, and bounds. The property Ops use int32 structural outputs; do not assume graphs can handle arbitrary 64-bit index ranges. |
| `cast` and typed cast Ops, `csr_from_dense`, `csc_from_dense`, `dense_from_sparse` | Dtype conversion and densification are explicit model decisions. Reject lossy casts unless scientifically justified; dense output may be unaffordable even if the source is sparse. |
| `clean`, `ensure_sorted_indices`, `remove0`; `sp_ones_like`, `sp_zeros_like` | Stored zeros count toward `nnz`. Sorting is not the same as combining duplicates. `remove0` removes explicit zeros; `sp_ones_like` fills stored positions, while `sp_zeros_like` creates empty storage with the same shape, not the same stored pattern. Canonicalize duplicates before nonlinear transforms. |
| `structured_exp/log/sigmoid/pow/minimum/maximum/add`, other structured elementwise functions and their decorator | The implementation transforms **stored data entries**, including explicit zeros. An implicit zero remains zero even if `f(0) != 0`. This is not dense elementwise semantics. Structured `exp` maps a stored zero to one but leaves an unstored zero at zero. |
| `add`, `subtract`, `multiply`, comparisons and specialized SD/SS/SV Ops | Determine dense versus sparse result and broadcasting from the operation. `multiply` is elementwise, not matrix multiplication. Non-zero-preserving comparisons/addition can produce dense or nearly full sparse storage. `sub`/`mul` are deprecated spellings; use the nondeprecated functions. |
| `dot`, `structured_dot`, `true_dot`, `sampling_dot`, `usmm`, `sp_sum` | `dot` is documented to return dense output; `true_dot` provides sparse-result semantics. `structured_dot` restricts gradients with respect to the sparse argument to its stored pattern. `sampling_dot(x,y,p)` computes `p * (x @ y.T)` at the sampled pattern, not a new probability sampler. `usmm` forms `alpha * x @ y + z`. Check whether each operand and gradient remains sparse. |
| row/column scale, `hstack`/`vstack`, transpose, `diag`/`square_diagonal`, get-item/construct-from-list Ops | These build sparse design structures. Index pairing versus Cartesian selection, output format, and shape changes matter. Do not equate transpose with merely swapping shape metadata. `row_scale`/`col_scale` have format-specific implementations. |

Include both stored and implicit zeros when checking sparse nonlinear
operations. Compare structured with dense exponentiation, inspect `indices`
and `indptr`, and differentiate a linear predictor with respect to stored
coefficients. A BSR type does not imply BSR `structured_exp` support in the
linked version. Check formats, complex gradients, batching and indexing on
the actual operation rather than inferring them from the type constructor.

Sources: [sparse type](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/sparse/type.py),
[storage/constructors](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/sparse/basic.py),
[arithmetic/structured gradients](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/sparse/math.py),
[shared storage](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/sparse/sharedvar.py).

## Stable special functions

Use `logsumexp`/`logaddexp` for adding positive terms represented in log space;
use `log_softmax` for log category probabilities rather than logging an
underflowed softmax. Set the reduction axis explicitly: `axis=None` normalizes
or reduces the entire tensor, not automatically each observation. Check large
positive and negative logits and the `logsumexp` derivative against softmax.
All-minus-infinity logits have no normalized probability distribution;
stable algebra does not rescue their gradient.

`betaln` works in log space using log gamma; `beta`, `factorial`, and `poch`
use gamma/products/ratios in PyTensor 3.3.0 and can overflow. For likelihood
normalizers prefer log-gamma or log-beta expressions in their valid domain.
A log-gamma difference is not automatically a stable substitute across poles
or where the gamma ratio has a sign; first establish positive arguments.
`logit` requires an interior probability: endpoints give signed infinities and
out-of-support inputs NaNs. This is not an automatic support exception.
`xlogy` and `xlog1py` correctly return zero at zero coefficient, including the
boundary used for count terms; their derivatives need not be finite there.
Check zero-boundary values separately from derivatives.

Source: [special functions](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/special.py).

## Spectral, convolution, and interpolation choices

`rfft` expects a real floating array with a **leading batch dimension**. For
shape `(batch, n)` it returns `(batch, n//2+1, 2)`, storing real and imaginary
parts in the last axis, not native complex dtype. `irfft` needs `is_odd=True`
for odd original length; packed size alone cannot distinguish odd/even input.
Match normalization: default normalizes the inverse, `ortho` normalizes both,
and `no_norm` is unnormalized (a forward/inverse pair then multiplies by the
transformed size). Do not import NumPy's other normalization spellings blindly.
Integer FFT inputs are not suitable merely because construction accepts them:
the linked implementation stores outputs in the input dtype. Check odd/even
lengths, float32/float64, packed shape, NumPy spectra and inverse reconstruction.

`convolve1d`/`convolve2d` express linear filtering, not automatically circular
convolution. Decide full/valid/same alignment, padding (`fill`, `wrap`, `symm`
for 2D), and kernel origin. `same` keeps the first input's size, which need not
match NumPy's 1D behavior when the second input is longer. Method `fft` versus
`direct` is not a performance guarantee; boundary conditions are scientific
assumptions. Compare alignment against SciPy. The PyTensor 3.3.0 wrappers do not
consistently reject unknown mode strings; check user-selected modes rather than
assuming type annotations enforce them.

`interp(query, knots, values)` clamps beyond endpoints by default, like the
nonperiodic NumPy interface. `interpolate1d` constructs a symbolic interpolator,
sorts knots with ordinates, and defaults to **extrapolation**. Methods are linear,
nearest, first, last, and mean, not cubic splines. Sorting is not duplicate-knot
validation: supply finite, distinct knots, matching one-dimensional ordinates,
and enough knots for the chosen rule. Duplicate knots can divide by zero.
Nearest/step methods have discontinuities and knot locations/sorting can make
derivatives unsuitable for continuous-gradient inference. `period` is documented
unsupported in `interp` and silently unused in the linked PyTensor 3.3.0
implementation. Check installed behavior; do not assume it gives periodic
interpolation or an exception. Compare clamping and extrapolation explicitly.

Sources: [FFT](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/fft.py),
[convolution](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/signal/conv.py),
[interpolation](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/interpolate.py).

## Implicit solutions inside a graph

`pytensor.tensor.optimize.minimize/minimize_scalar/root/root_scalar` wrap SciPy
routines with symbolic objective/equation inputs. They are not posterior
samplers, and placing an optimizer in a graph does not integrate out its
solution. All return a solution and a success flag. A false flag means the
returned terminal state need not be a minimum/root. Check flags **and** residuals
or first-order conditions, domain, branch selection, and sensitivity to starts.
Method-specific bounds/brackets/Jacobian/Hessian options depend on the installed
interface and SciPy method.

Implicit derivatives need a regular solution: a nonsingular root Jacobian or
appropriate local Hessian, and a stable branch. Nonsmooth objectives, flat
minima, competing roots, boundary optima, and solver failure invalidate a casual
differentiability claim. For example, the positive root of `x**2-theta=0` has
derivative `1/(2*sqrt(theta))` only for `theta > 0` on that branch. Check both
residual and derivative; local agreement is not global convergence.

Source: [optimization Ops and public functions](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/tensor/optimize.py).

## Named numerical dimensions and backend boundaries

`pytensor.xtensor.linalg.cholesky` requires two core dimension names.
`xtensor.linalg.solve` requires two names for a matrix/vector equation and three
for a matrix/matrix equation; the shared dimension is contracted. Names do not
establish matrix symmetry or positive definiteness. Named convolution requires
a different core name for each input and retains the first input's core name.
Check named dimensions and numerical values, including repeated-name errors.

`Mode.optimizer` and `optimizer`, `optimizer_including/excluding/requiring`,
`optimizer_verbose`, and `optimizer_verbose_ignore` select/inspect graph rewrite
behavior; they do not optimize a scientific objective. `blas__ldflags` and
`blas__check_openmp` concern linking/runtime compatibility, not numerical API
availability. Change flags only in a controlled project environment after
inspecting the selected backend and profiling representative shapes. Do not
suggest global BLAS changes based on a tiny calculation.

Express matrix products symbolically and let the selected compiler lower them.
Generated C-BLAS kernels and alias-analysis internals are not alternate
numerical APIs. Custom Op storage and derivative contracts require explicit
extension work, not manipulation of a built-in kernel.

Sources: [named linalg](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/xtensor/linalg.py),
[named convolution](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/xtensor/signal.py),
[configuration registration](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/configdefaults.py),
[Mode](https://github.com/pymc-devs/pytensor/blob/aa1bc7772eb11ed8def2814374aeb5da03da590c/pytensor/compile/mode.py).

## Check numerical behavior

- Compare solves using residuals, not just agreement with another solver.
  Assess conditioning; a small residual alone does not imply small forward error.
- For decompositions compare reconstructed matrices and subspaces rather than
  arbitrary vector signs/phases. Check stationarity or closed-loop stability
  when those are part of the scientific claim.
- Check exact shapes, output dtypes and sparse structural arrays separately
  from floating-point tolerances. Float32, complex arithmetic and ill-conditioned
  systems need their own error budgets.
- For implicit solvers require success flags and residual/first-order checks.
  Test branch selection and derivatives away from singularities.
- Include non-SPD, nonsquare and singular systems, unsafe casts, unsupported
  sparse formats, malformed structural indices, FFT rank/normalization errors,
  duplicate interpolation knots and invalid named dimensions.
- Preserve mathematical NaN/infinity behavior when that is the API contract,
  such as logit endpoints and out-of-support probabilities. Do not fabricate
  exceptions, clip failures away, or infer gradient health from finite values.
