# Spline smoothing and distributional regression

Distributional regression uses ordinary PyMC likelihoods with separate linked
predictors for distribution parameters; it needs no special GAMLSS class.
Spline utilities described here are in `pymc_extras.utils.spline` for extras 0.14.0.
That release requires PyMC >=6.2,<6.3, PyTensor >=3.2.3,<3.3 and PreliZ >=0.27,<0.28.
Check current APIs/dependencies rather than override incompatible package bounds.

## Use the actual spline interface

| API | Contract |
|---|---|
| numpy_bspline_basis(eval_points,k,degree=3) | Numeric matrix with k **basis coefficients**, not k knots. Use floating inputs to avoid integer-cast output. |
| BSplineBasis(sparse=True)(eval_points,k,degree) | PyTensor Op with rank-one floating evaluation points, integer k/degree, dense or CSR output; not a fitted estimator. |
| bspline_basis(n,k,degree=3,...) | Symbolic basis at n regular points on [0,1], not irregular observed x. |
| bspline_interpolation(x,n=... or eval_points=...,...) | Multiplies the first coefficient axis by a basis. Supply exactly one evaluation specification; it does not solve interpolation through coefficient values. |

There is no `pmx.BSplineBasis(n_knots=...).build(x)` or exported ZeroSumNormalBasis
in that release. Use the module functions rather than invent compatibility objects.
For degree d and coefficient count k, its clamped knot sequence is

```python
knots = np.r_[np.zeros(d), np.linspace(0, 1, k-d+1), np.ones(d)]
```

There are k+d+1 knots including repeats and endpoint multiplicity d+1. Require
k>=d+1 and enough distinct predictor locations. The 0.14 basis Op has no coordinate
gradient or JAX/Numba lowering: precompute a fixed observed-x matrix and sample
coefficient expressions, not uncertain x through an unsupported derivative.

```python
from pymc_extras.utils.spline import numpy_bspline_basis
u = (train_x - lower_bound) / (upper_bound - lower_bound)
B = numpy_bspline_basis(u.astype(float), k=basis_count, degree=3)
# Apply the identifiable coefficient prior below before using B in a likelihood.
```

## Difference order is a prior choice

For D_r, the r-th row difference of the k-by-k identity, roughness is ||D_r w||².
First differences are `w[j+1]-w[j]`; second differences are
`w[j+2]-2*w[j+1]+w[j]`. GaussianRandomWalk coefficients with independent increments
penalize **first**, not second differences. First-order smoothing shrinks toward
constant coefficients, second-order toward coefficient-index linear trends.
With clamped boundary knots an index-linear sequence need not be exactly linear
in physical x near boundaries. Neither penalty automatically equals integrated
squared physical curvature, especially with irregular knots.

## Identify and normalize the smooth

Partition of unity makes a free constant coefficient direction redundant with a
separate intercept. Let `Bc=B-training_column_means`. For nonzero SVD part
`D_r=U diag(s) V.T`, define `H=V diag(1/s)` and training normalization
`c=sqrt(mean(sum((Bc@H)**2,axis=1)))`.

- Put z ~ Normal(0,1) in k-r penalized dimensions and a proper positive smooth
  amplitude tau in response units.
- For first differences use `w=tau*(H/c)@z`.
- For second differences additionally model the nonconstant nullspace direction
  `w += beta_trend*v`, with v orthogonal to the constant and normalized to unit
  training RMS of Bc@v. Give beta_trend a proper response-scale prior.
- Add a separate proper intercept prior and no sampled constant coefficient
  direction. Check augmented design rank and valid normalization.

Conditional training-average smooth variance is tau²; difference innovation scale
is tau/c and roughness precision `(c/tau)²`. Increasing tau weakens smoothing.
Save c: normalization changes the prior, not just code representation. Distinct
penalty orders are different prior candidates, not automatically equivalent
parameterizations. Orthogonal coefficients do not imply posterior independence.

## Reuse training transformations

Save physical bounds, unit mapping, full knot vector, degree/coefficient count,
training column means, whitening matrix, c and nullspace directions. New x must use
the same transformation, not new batch min/max, centering or SVD. A subset evaluated
alone must reproduce its rows from a full prediction batch. This catches accidental
redefinition of the intercept or smooth and misuse of regular-grid constructors.

Update all shared design arrays/coords together with likelihood shape tied to the
new design, save predictions, and restore training data in finally. Preserve row
identity and check linked parameters draw-wise. Extrapolation beyond training
bounds is a scientific choice; polynomial extrapolation by an underlying basis
is not a guarantee of adequacy. Reject or explicitly model extrapolation, never
silently clip inputs. Uncertain knots/locations require a different model.

## Link distribution parameters in their units

| Likelihood | Linked parameters and variance |
|---|---|
| Normal | mu=X beta_mu, sigma=exp(Z beta_sigma); sigma is SD, not variance. |
| Beta | mu=sigmoid(X beta_mu), phi=exp(Z beta_phi); alpha=mu*phi, beta=(1-mu)*phi; Var=mu(1-mu)/(phi+1). |
| NegativeBinomial | mu=exp(X beta_mu), alpha=exp(Z beta_alpha); Var=mu+mu²/alpha. alpha is inverse overdispersion, not variance. |

```python
with model:
    beta_mu = pm.Normal("beta_mu", 0, mean_link_scale, dims="mean_feature")
    beta_sigma = pm.Normal("beta_sigma", 0, scale_link_scale, dims="scale_feature")
    mu = pm.math.dot(X, beta_mu)
    sigma = pm.math.exp(pm.math.dot(Z, beta_sigma))
    pm.Normal("response", mu=mu, sigma=sigma, observed=y, dims="obs_id")
```

Fit both predictors jointly and preserve joint posterior draws. Beta endpoints
need an explicit zero/one or measurement model rather than jitter. Counts need
integer support; varying exposure belongs as a known log-mu offset. More flexible
scale/shape predictors can be weakly identified and need prior/PPC sensitivity.

If log q=a+b*z with independent Normal coefficients,

\[
\operatorname{median}(q)=e^{m_a+m_bz},\qquad
E(q)=e^{m_a+m_bz+(s_a^2+z^2s_b^2)/2}.
\]

A log-scale intercept centers a **median**, not a mean. Variance exp(2*log_sigma)
has a different, more skewed prior. exp(b) is a multiplicative effect per unit z;
logit b is a log odds ratio, not fixed percentage points. Priors can widen strongly
at design extremes. Choose scales from physical units and plausible endpoint
ratios; there is no universally weak coefficient SD across links.

Before fitting, inspect native-unit quantiles, dispersion, conditional variances,
endpoint ratios, Beta boundary mass and NB zero probability
`(alpha/(alpha+mu))**alpha`. Compare to domain expectations, not post-hoc bounds
chosen from outcomes. Failed plausibility should prompt a reasoned prior/model
revision, not clipping or seed searches.

Sources: [spline implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.14.0/pymc_extras/utils/spline.py),
[extras dependencies](https://github.com/pymc-devs/pymc-extras/blob/v0.14.0/pyproject.toml),
[PyMC continuous families](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/continuous.py),
[discrete families](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/discrete.py).
