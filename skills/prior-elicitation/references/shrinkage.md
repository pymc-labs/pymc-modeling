# Scale-aware shrinkage

Continuous shrinkage retains uncertainty when many coefficients may be small.
Neither horseshoe nor R2D2 creates a point mass at zero. An interval spanning
zero does not establish irrelevance, and `P(abs(beta)>practical_threshold)` is not
a model-inclusion probability. Selection requires an explicit decision loss and
leakage-safe validation.

## Regularized horseshoe

For centered, unit-variance predictors in Gaussian regression, a sparsity
reference is `tau0 = p0/(p-p0) * sigma/sqrt(n)`, with `0<p0<p` and residual SD
`sigma`. Omitting `sigma` requires fixing that scale to one or including the
factor elsewhere. This approximately orthogonal-regression reference is not an
exact sparsity law for correlated predictors or arbitrary links. A HalfCauchy
prior with this scale does not fix the expected active count at `p0`.

For local scale `lambda_j` and standard-Normal coefficient `z_j`,

```
lambda_tilde_j**2 = c2 * lambda_j**2 / (c2 + tau**2 * lambda_j**2)
beta_j = z_j * tau * lambda_tilde_j
c2 ~ InverseGamma(nu/2, nu * slab_scale**2 / 2)
```

Conditional coefficient variance is
`1/(1/(tau*lambda_j)**2 + 1/c2)`, bounded by `c2`. Coefficients themselves remain
unbounded: the slab soft-regularizes, not hard-truncates. For standardized
predictors, `c2` has squared outcome units. An illustrative slab scale of 2 mg/L
and `nu=4` gives `InverseGamma(2,8)`, not an unexplained `InverseGamma(2,1)`.

An inline construction, with `X` centered and unit-variance and all scale choices
elicited in outcome units:

```python
import numpy as np
import pymc as pm

n, p = X.shape
with pm.Model(coords={"feature": feature_names}) as model:
    alpha = pm.Normal("alpha", mu=intercept_mean, sigma=intercept_scale)
    sigma = pm.HalfNormal("sigma", sigma=noise_scale)
    tau = pm.HalfCauchy("tau", beta=p0 / (p - p0) * sigma / np.sqrt(n))
    local = pm.HalfCauchy("local", beta=1, dims="feature")
    c2 = pm.InverseGamma("c2", alpha=nu / 2, beta=nu * slab_scale**2 / 2)
    z = pm.Normal("z", 0, 1, dims="feature")
    local_regularized = local * pm.math.sqrt(c2 / (c2 + tau**2 * local**2))
    beta = pm.Deterministic("beta", z * tau * local_regularized, dims="feature")
    y = pm.Normal("y", mu=alpha + pm.math.dot(X, beta), sigma=sigma,
                  observed=y_obs)
```

Global/local scales can still create difficult geometry. Inspect all latent
scales as well as coefficients: divergences, rank R-hat, bulk/tail ESS, MCSE and
trace/rank plots. Non-centering and higher `target_accept` are not convergence
certificates. A prior change must have scientific justification, not merely make
a warning disappear.

When `X.T@X=n*I` and the intercept is orthogonal to X,
`kappa_j=1/(1+n*beta_sd_j**2/sigma**2)` describes **conditional** shrinkage
relative to least squares. `sum(1-kappa)` is continuous complexity, not a discrete
selected-variable count. With correlated X, this scalar interpretation is only
approximate; marginal posterior shrinkage is not one fixed kappa.

## R2D2M2CP in pymc-extras 0.14.0

This optional helper returns a named tuple with **`eps` and `beta`**, not
`(beta,r2)`. It accepts scale information, not positional design/response arrays:

```python
import pymc_extras as pmx

with pm.Model(coords={"feature": ["a", "b", "c"]}):
    output_sigma = pm.HalfNormal("output_sigma", sigma=2.0)  # illustrative units
    allocation = pmx.R2D2M2CP(
        "beta", output_sigma=output_sigma, input_sigma=np.ones(3),
        dims="feature", r2=0.5, r2_std=0.2,
        variables_importance=np.full(3, 0.5),
        positive_probs=0.5, centered=False,
    )
    # Linear predictor uses allocation.beta; Normal noise SD is allocation.eps.
```

`output_sigma` is total outcome scale, not residual scale. `input_sigma` contains
positive predictor SDs. If predictors were already divided by SD, use ones to
avoid double scaling. Center predictors and give the intercept a separate prior.
An outcome-derived `y.std()` is an empirical prior choice, not a required input.
Respect the package's dependency bounds; do not force-install an incompatible
PyMC/PyTensor combination.

With `positive_probs=.5`, normalized allocation `phi` and variance budget `R2`,

```
eps = output_sigma * sqrt(1-R2)
beta_j = z_j * output_sigma * sqrt(R2*phi_j) / input_sigma_j
```

`r2=.5, r2_std=.2` implies Beta shapes 2.625 and 2.625. Dirichlet concentration
controls how unequal allocations are; it does not fix a sparse count. The R2
budget is not the exact realized sample variance explained in every draw,
especially with correlated predictors. R2D2 is not restricted to dense effects.

**Version-specific caution:** in 0.14.0, omitting all allocation arguments makes
`_phi` return ones rather than normalized `1/p`. If a shared variance budget is
intended, provide normalized `variance_explained` or explicit positive Dirichlet
concentrations through `variables_importance`. `r2_std=None` fixes R2 rather than
eliciting uncertainty; `variance_concentration` is not an accepted argument.
Consult the version-matched source for direction-probability, centered and
multilevel broadcasting semantics; do not infer them from the symmetric example.

## Check implications and report the target

- Prior-predictive simulations should assess conditional scale, extremes and
  meaningful predictor profiles, not only coefficient marginals. Distinguish
  soft plausibility bounds from physical support and retain simulation uncertainty.
- Transform coefficients back to original units. With centered/scaled predictors
  and unscaled response, `beta_native=beta_standardized/predictor_scale` and
  `intercept_native=alpha-dot(beta_native,predictor_center)`. The native intercept
  may extrapolate outside meaningful profiles; alpha at the centers may be more useful.
- Compare reasonable sparsity, slab, scale and allocation assumptions while holding
  the observation model and target fixed. Large shrinkage can reflect the prior,
  correlated predictors or weak information, not demonstrated irrelevance.
- Separate latent-profile uncertainty from noisy new observations. For Gaussian
  new observations, total variance is
  `Var(mu_new | data) + E(sigma**2 | data)`, not only latent-mean variance.
- In-sample residual/extreme-value PPCs are model criticism, not external
  calibration. One simulated parameter value falling outside one credible
  interval does not establish miscalibration; repeated calibration needs a
  declared repeated-sampling experiment.
- Report practical coefficient probabilities and prediction intervals with MCSE.
  Include latent-scale exploration and any divergent trajectories in the
  assessment alongside coefficient summaries.

## Primary sources

- [Piironen and Vehtari: regularized horseshoe and sparsity calibration](https://arxiv.org/pdf/1707.01694),
  especially the Gaussian/scale assumptions behind equations 2.8, 2.11 and 3.12.
- [pymc-extras 0.14.0 R2D2M2CP source](https://github.com/pymc-devs/pymc-extras/blob/v0.14.0/pymc_extras/distributions/multivariate/r2d2m2cp.py).
- [R2D2M2 research](https://arxiv.org/abs/2208.07132) and
  [CP implementation proposal](https://github.com/pymc-devs/pymc-extras/pull/137).
  The CP implementation should not be assumed identical to every R2D2 prior.
