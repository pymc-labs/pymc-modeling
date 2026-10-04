# Gaussian processes: choose the target before the approximation

Declare input/output units, observation noise, dependence and whether predictions
concern latent f or future noisy y. A GP prior does not eliminate grouping,
temporal leakage or extrapolation assumptions. Use `X.shape=(n,input_dim)` even
in one dimension, and retain training transformations for new inputs.

## Choose the formulation

| Interface | Scientific and computational meaning |
|---|---|
| Marginal | Gaussian observations integrated over latent f; exact Gaussian conditioning conditional on hyperparameters. Random hyperparameters still require inference. |
| Latent | Explicit latent GP with any appropriate likelihood. `conditional` conditions on f, not directly on noisy y. `n_outputs` repeats independent GPs, not cross-output covariance. |
| MarginalApprox | DTC/FITC/VFE inducing approximations for Gaussian observations; VFE objective differs from an exact likelihood. Use this name, not deprecated MarginalSparse. |
| HSGP | Finite Laplacian eigenbasis with spectral weights on a bounded domain; attach the actual likelihood. |
| HSGPPeriodic | One-active-dimension Fourier basis for a Periodic kernel instance; amplitude is separate `scale`, not an arbitrary product kernel. |
| LatentKron, MarginalKron | Exact separable covariance on a full Cartesian training grid; prediction inputs need not form a grid. Do not invent a grid for irregular data. |
| TP | Student-t process with joint tail dependence, not a GP with independent robust observation noise. |

There is no universal exact-GP observation-count cutoff. Dense Cholesky is O(n³)
time/O(n²) storage; inducing algebra commonly costs O(nu²+u³). An HSGP multiplication
costs O(nM), M=product(m), but Gaussian coefficient conditioning can cost
O(nM²+M³). Latent sampling geometry, prediction covariance and stored arrays also
matter. Measure actual build/compile, gradients, inference and prediction.

## Mean and covariance assumptions

Lengthscales have input units; amplitude/noise have response units. Use either
`ls` or `ls_inv`, and match anisotropic scales to `active_dims`. Validate finite
inputs, symmetry, diagonals and PSD. Jitter is a numerical stabilizer, not noise
or a repair for an invalid covariance.

| Kernel/mean family | Interpretation and caution |
|---|---|
| mean.Zero/Constant/Linear | Centering, level or extrapolating trend. Mean and long-scale GP can be weakly identified. Custom Mean must return correct shape. |
| ExpQuad | Infinitely smooth stationary prior; eta² multiplies variance. |
| Matern12/32/52, Exponential | Different roughness. PyMC 6.3.1 Exponential is exp(-r/2), Matern12 exp(-r); ls is not numerically interchangeable. |
| RatQuad | Positive-alpha scale mixture of squared exponentials, not general nonstationarity. |
| Constant, WhiteNoise | Shared random offset versus diagonal sigma². Avoid double-counting likelihood noise; cross-input WhiteNoise calls return zero even at repeated coordinates. |
| Linear, Polynomial | Nonstationary covariance. Nonnegative integer elementwise powers preserve PSD; arbitrary real powers need not. |
| Cosine, Periodic, WrappedPeriodic, Circular | Distinct periodic/circular assumptions. Periodic uses exp(-sin²(pi*delta/T)/(2*ls²)); conventions elsewhere differ. Circular uses geodesic distance and documented tau>=4. |
| WarpedInput, Gibbs, ScaledCov | Validate warps, positive local lengthscales or nonnegative response scaling at train/test inputs. Gibbs in 6.3.1 accepts one active dimension. ScaledCov scales both input locations. |
| Coregion | B=W W.T+diag(kappa) or explicit PSD B. Validate integer task labels before the implementation casts them. |
| Kron | Product over separate input blocks with ordering matching `pm.math.cartesian(*Xs)`. |
| Add/Prod/Exponentiated | Sum kernels model sums of independent functions; product kernels do not mean products of GP draws. Use callable kernel interface; inherited `.full`/`.diag` can be abstract. |

BaseCovariance/Covariance/Stationary are extension contracts, not ready-made priors.
Provide full/diagonal behavior and required spectral implementation. A valid dense
kernel need not support HSGP: PyMC 6.3.1 supplies power spectra for ExpQuad,
Matern32/52 and RatQuad; compatible stationary sums and scalar amplitudes work,
but arbitrary covariance products lack a generic spectral implementation. Periodic
has separate Fourier coefficients. Do not switch kernels merely to compile.

## Exact prediction and uncertainty

```python
# X, y, Xnew and hyperparameter values are supplied by the project.
with pm.Model() as model:
    gp = pm.gp.Marginal(cov_func=eta**2 * pm.gp.cov.ExpQuad(X.shape[1], ls=ell))
    gp.marginal_likelihood("y", X=X, y=y, sigma=sigma)
    mean_f, cov_f = gp.predict(Xnew, point=hyperparameter_point)
```

For fixed mean m and covariance K,

\[
\mu_*=m_*+K_{*X}(K_{XX}+\sigma^2I)^{-1}(y-m_X),\qquad
V_*=K_{**}-K_{*X}(K_{XX}+\sigma^2I)^{-1}K_{X*}.
\]

Use linear solves, not materialized inverses. Independent future noise adds
sigma² I; match jitter placement for tight checks. A point-conditioned prediction
is not hyperparameter-integrated uncertainty. For random hyperparameters, fit/save,
create `gp.conditional("f_new",Xnew=...)` and generate posterior predictions for
each draw. Averaging covariances alone omits covariance of conditional means.

`given` for Latent uses X/f; Marginal uses X/y/sigma. Additive component conditioning
also needs the total gp, and component posteriors are generally dependent. A joint
GP observation density does not automatically supply valid rowwise LOO; define
the held-out unit and conditioning target.

## Inducing points

Choose Xu from training input geometry, never held-out responses. The k-means
utility needs sensible units, nondegenerate columns and controlled randomness.
With `Q=K_Xu K_uu^-1 K_uX`, DTC uses Q+sigma²I, FITC adds diag(K_XX-Q), and VFE
uses the DTC Gaussian objective minus trace(K_XX-Q)/(2 sigma²). VFE is represented
by a Potential: prior simulation cannot turn it into a generative likelihood or
one pointwise observation factor. Fixed-hyperparameter DTC/VFE predictions can
agree while their hyperparameter objectives differ.

## HSGP domain and truncation

Choose a boundary covering training and intended prediction inputs before fitting.
The first training range midpoint becomes the retained center. L is half-width;
c scales training half-range. Supply exactly one of L/c and one m/L per active
dimension. Do not recenter on prediction inputs. Finite output at/outside the
boundary is not valid extrapolation.

```python
with model:
    gp = pm.gp.HSGP(m=basis_counts, L=domain_halfwidths, cov_func=covariance)
    f = gp.prior("f", X=X)
    # Attach the project's observation likelihood before fitting.
    f_new = gp.conditional("f_new", Xnew=Xnew)
```

`prior_linearized(X)` returns phi/sqrt_psd, without a mean function:
`f=mean(X)+phi@(beta*sqrt_psd)` for standard-Normal beta. Reuse coefficients,
center, boundary and covariance convention when updating data. Centered versus
non-centered coefficients is a geometry choice. Avoid deprecated `drop_first`
as an automatic identification fix.

Compare covariance, latent means and uncertainty over plausible lengthscales and
training/interpolation/extrapolation inputs, varying m and L together. Larger L
at fixed m reduces spectral resolution. Errors unchanged when m rises can indicate
boundary padding, not basis implementation; widening L alone can worsen truncation.
An exact finite-basis Gaussian posterior is still not the infinite-basis GP.
Heuristic m/c recommendations are starting points, not accuracy guarantees.

## Version-specific numerical checks

Before relying on the following PyMC 6.3.1 interfaces, compare against an independent
formula and inspect the installed source; these cautions do not assert current
versions retain the same implementation.

- **HSGPPeriodic:** for Periodic's exponent, Fourier variance weights use
  `a=1/(4*ls²)` and `(2 if j>0 else 1)*ive(j,a)`; standard-Normal coefficients
  need square-root weights. Check both lengthscale and variance-versus-SD
  conventions in Periodic/HSGPPeriodic source. Internal agreement of two public
  paths does not establish fidelity to the declared covariance. A fixed period
  also assumes scientific stability under extrapolation.
- **TP:** if the prior uses MvStudentT scale K, conditional scale is
  `(nu+beta)/(nu+n)` times the Schur complement. A covariance-parameterized formula
  with minus-two terms cannot be passed unchanged as scale. Independent univariate
  t coordinates rotated by Cholesky do not generally yield the same elliptical
  multivariate t as a shared random mixing scale; check reparameterized priors.
- **MarginalKron noisy predictions:** verify observation noise contributes sigma²,
  not sigma, in full/diagonal covariance and conditional builders.

`util.stabilize` only adds jitter. `plot_gp_dist` requires ordered input and an
explicit interval interpretation. Latent/HSGP objects use their latent interfaces, not
unimplemented inherited `predict`/`marginal_likelihood` methods.

Sources: [GP API](https://www.pymc.io/projects/docs/en/stable/api/gp.html),
[GP implementation](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/gp/gp.py),
[covariances](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/gp/cov.py),
[HSGP](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/gp/hsgp_approx.py),
[HSGP methodology](https://arxiv.org/abs/2004.11408).
