# Mixtures and supported marginalization

## Model the mixture

For conditionally independent observations,

\[
p(y_i\mid w,\theta)=\sum_k w_k f_k(y_i\mid\theta_k),\qquad
\log p(y_i\mid w,\theta)=\operatorname{logsumexp}_k[\log w_k+\log f_k(y_i\mid\theta_k)].
\]

A mixture selects a component; it is not the distribution of a weighted sum of
independent component realizations. Weights are nonnegative and sum to one along
the component axis. `pm.Mixture` requires unnamed `.dist()` components and clones
them; do not infer shared latent realizations from reused Python objects.

```python
# w and mu have component shape (K,); scales and y come from the model.
components = pm.Normal.dist(mu=mu, sigma=component_sd)
pm.Mixture("response", w=w, comp_dists=components, observed=y, dims="obs_id")
```

`NormalMixture` is a convenience constructor with sigma (SD) or tau (precision),
not both. For scalar events the last component batch axis is consumed; for vector
events the mixture axis precedes event axes. A `(K,D)` MvNormal collection produces
one D-vector, not D independently chosen components. Components must share support
dimensionality and be all discrete or all continuous; dedicated hurdles manage
their own mixed measure.

## Label symmetry and identification

Marginalizing allocations does **not** remove label switching. Exchangeable
component priors imply equivalent label permutations. Initialize across these
permutations and dispersed substantive modes. Diagnose label-invariant mixture
means/variances, predictive densities, sorted means and corresponding weights;
raw component traces can be misleading. Always permute weights, scales and
responsibilities with means. Near-coincident components remain hard to interpret.

Ordering or genuinely different component priors need scientific justification.
An ordering convention does not establish distinct biological or social groups.
Healthy invariant diagnostics do not prove exploration of non-label modes, identify
K, resolve empty components or prevent unknown-scale collapse. Diagnose geometry;
higher `target_accept` alone does not remove multimodality or nonidentification.

## Responsibilities are uncertain allocations

At each posterior draw,

\[
r_{ik}=\exp\{\log w_k+\log f_k(y_i\mid\theta_k)-\log p(y_i\mid w,\theta)\}.
\]

Normalize in log space and integrate over joint posterior draws; substituting
posterior mean parameters into this nonlinear expression is not equivalent.
Exchangeable `P(z_i=0|y)` does not identify an externally named class. For distinct
observations under iid allocations, co-clustering probability averages
`sum_k r_ik*r_jk`; a self-pair has probability one. Recovered categorical draws add
simulation uncertainty beyond the conditional probabilities.

## Choose supported elimination with pymc-extras

This section targets **pymc-extras 0.15.1** and its compatible PyMC 6.3/PyTensor 3.3
environment; see [extras selection and installation](pymc-extras.md).
`pmx.marginalize(model, rvs_to_marginalize=(), *, laplace_approx=None,
minimizer_kwargs=None)` clones a nonempty selection into a new model. Ordinary
selections request **exact** supported rewrites; `laplace_approx={name_or_rv: Q}`
explicitly requests **approximate** integration. Laplace keys need not also appear
in the exact selection. Unsupported exact cases raise rather than silently become
Laplace approximations.

| Strategy in 0.15.1 | Supported scope |
|---|---|
| Finite enumeration | Bernoulli, Categorical, DiscreteUniform; supported observationwise/batched graphs. Domains can remain symbolic. |
| Markov-chain rewrite | `pmx.DiscreteMarkovChain` with its supported transition/emission graph; not arbitrary HMM recognition. |
| Exact continuous conjugacy | Elementwise Normal prior → one Normal dependent with affine mean and latent-independent scale. |
| Laplace integration | A continuous Gaussian latent vector with supplied prior precision `Q`; mode/curvature approximation, not arbitrary exact conjugacy. |

Beta–Binomial, Gamma–Poisson, Dirichlet–Categorical and general MvNormal
conjugacy are **not** additional registered exact rewrites in this release.
An analytic integral available on paper is not a guarantee this graph API
recognizes it. Derive a valid explicit marginal likelihood if needed, preserving
its dependence structure and a separate generative/predictive model.

### Exact discrete marginalization and recovery

```python
import pymc as pm
import pymc_extras as pmx
from pymc_extras.marginal import conditional, recover, unmarginalize

with pm.Model() as original:
    w = pm.Dirichlet("w", a=[1.0, 1.0])
    comp = pm.Categorical("comp", p=w, shape=3)
    pm.Normal("response", mu=pm.math.switch(comp, 1.0, -1.0),
              sigma=0.5, observed=[-0.8, 0.9, 1.2])

marginal_model = pmx.marginalize(original, ["comp"])
idata = pm.sample(model=marginal_model, nuts_sampler="pymc",
                  tune=500, draws=500, chains=2, cores=1, random_seed=42)
idata.to_netcdf("mixture-posterior.nc")
allocations = recover(idata, model=marginal_model, var_names=["comp"],
                      extend_inferencedata=False, random_seed=43)
conditional_model = conditional(marginal_model, ["comp"])
log_conditional = conditional_model.compile_logp(
    vars=[conditional_model["comp"]], sum=False)
restored = unmarginalize(marginal_model, ["comp"])
```

The finite calculation is exact up to numerical arithmetic; the retained-variable
posterior still needs reliable sampling. The illustrative budgets here and below
are not convergence guarantees. `recover` draws eliminated variables from their
conditional posterior; it does not return probability arrays. It defaults to
extending/returning the supplied DataTree's `posterior`; with
`extend_inferencedata=False` it returns an xarray Dataset. Use `recover`, not the
deprecated `recover_marginals`. Empty selections can return their input unchanged.

`conditional` creates a refactorized joint model: select the recovered variable's
factor for conditional log probabilities, not the full joint logp. Multiple
recoveries follow a chain rule, not arbitrary independent full conditionals.
`unmarginalize` restores the original prior and generative graph, **not** the
conditional posterior; missing latent draws in that restored model will be
forward-sampled from the prior unless supplied/recovered.

Restrictions remain important:

- Marginalized variables must be free, not observations relabelled as latent.
- Registered dependent Deterministics/Potentials are rejected; an unrecorded
  expression such as `mu[comp]` is a different graph contract.
- Batched dependencies must use the supported elementwise/observationwise
  structure. Cross-observation mixing is not generic; splitting vector latents
  into scalars can create exponential graph growth.
- Infinite-support Poisson latents are not finite enumeration. Do not silently
  truncate or substitute a different model. A finite symbolic domain can still
  be too expensive to enumerate.
- Inspect warnings about dependent transform bounds and nonseparable likelihoods;
  do not suppress them or treat zero bookkeeping factors as observationwise logp.

### Exact Normal–Normal elimination

For `x_i ~ Normal(mu_i, s_i)` and `y_i ~ Normal(a_i+b_i*x_i, t_i)`,
the supported elementwise marginal is
`y_i ~ Normal(a_i+b_i*mu_i, sqrt(t_i**2+(b_i*s_i)**2))`.
Recovery uses the analytic Normal conditional, retaining posterior dependence
on the remaining parameters.

```python
import numpy as np
import pymc as pm
import pymc_extras as pmx
from pymc_extras.marginal import conditional, recover

y_obs = np.array([0.5, 1.2, 2.0])
with pm.Model() as normal_model:
    mu = pm.Normal("mu", 0.0, 2.0)
    x = pm.Normal("x", mu=mu, sigma=1.0, shape=3)
    pm.Normal("y", mu=1.0 + 2.0 * x, sigma=0.5,
              observed=y_obs)

normal_marginal = pmx.marginalize(normal_model, ["x"])
# Check only y's marginal factor at mu=1 (exclude the prior on mu).
expected = -0.5 * (((y_obs - 3.0) ** 2) / 4.25
                   + np.log(2 * np.pi * 4.25))
np.testing.assert_allclose(
    normal_marginal.compile_logp(vars=[normal_marginal["y"]])({"mu": 1.0}),
    expected.sum(),
)
# Check the exact conditional factor at x=0, mu=1.
normal_conditional = conditional(normal_marginal, ["x"])
post_mean = (1.0 + 8.0 * (y_obs - 1.0)) / 17.0
expected_cond = -0.5 * (post_mean**2 * 17.0 + np.log(2 * np.pi / 17.0))
np.testing.assert_allclose(
    normal_conditional.compile_logp(vars=[normal_conditional["x"]])(
        {"mu": 1.0, "x": np.zeros(3)}
    ),
    expected_cond.sum(),
)
normal_idata = pm.sample(model=normal_marginal, tune=500, draws=500,
                         chains=2, cores=1, random_seed=44)
joint_idata = recover(normal_idata, model=normal_marginal,
                      var_names=["x"], random_seed=45)
# Replicate responses for the SAME fitted latent units, using recovered x.
replicated = pm.sample_posterior_predictive(
    joint_idata, model=normal_model, var_names=["y"],
    sample_vars=["y"], freeze_vars=["mu", "x"], random_seed=46,
)
joint_idata.to_netcdf("normal-joint-posterior.nc")
```

Here each observation has its own latent `x_i`; sharing the hyperparameter `mu`
is allowed. The rewrite requires **exactly one dependent Normal RV node**, an
affine supported Add/Mul mean, and a scale independent of the eliminated variable.
It rejects broadcasting one scalar/size-one latent across many dependent draws:
those observations share a latent and their true marginal is correlated, not a
product of the elementwise Normals above. General matrix linear maps, nonlinear
means/scales and several dependent RV nodes are not this rewrite.
Check the marginal log density and recovered conditional moments against the
analytic formula on a small case before relying on a changed graph.

### Laplace elimination is approximate

`laplace_approx` supplies the **Gaussian prior precision** of a vector field,
not an arbitrary tuning matrix or a posterior covariance. The implementation
finds the conditional mode and uses `Q - Hessian(log_likelihood)` for curvature.
Supply the actual positive-definite precision (including parameter dependence)
and a compatible finite vector shape; this is not a general API for bounded,
non-Gaussian or arbitrary tensor latents. The rewrite does not validate that
your supplied `Q` matches the prior, so a wrong `Q` changes the calculation.

```python
import numpy as np
import pymc as pm
import pymc_extras as pmx

Q = np.eye(2)
with pm.Model() as field_model:
    theta = pm.Normal("theta", 0.0, 1.0)
    field = pm.MvNormal("field", mu=theta * np.ones(2), tau=Q)
    pm.Poisson("counts", mu=pm.math.exp(field), observed=np.array([1, 3]))

field_marginal = pmx.marginalize(
    field_model, laplace_approx={"field": Q},
    minimizer_kwargs={"method": "L-BFGS-B",
                      "optimizer_kwargs": {"tol": 1e-8}},
)
logp_at_zero = field_marginal.compile_logp()({"theta": 0.0})
field_idata = pm.sample(model=field_marginal, tune=500, draws=500,
                        chains=2, cores=1, random_seed=47)
```

This removes `field` from the sampled variables but approximates the target for
`theta`. More NUTS draws cannot remove integration error. The inner optimization
and dense Hessian run during marginal likelihood evaluation; initialization is
currently a deterministic all-ones vector. Check inner-solver sensitivity,
conditional skewness/multiple modes, precision/curvature and comparisons with
the original joint model or numerical integration. The experimental INLA helper
wraps this operation and `pm.sample`; see [INLA selection](approximate-inference.md#inla-select-a-latent-gaussian-structure-not-an-arbitrary-model).

**Recovery limit:** 0.15.1 has no conditional/recovery implementation for
Laplace-eliminated variables. `recover(..., var_names=["field"])` and
`conditional(..., ["field"])` raise `NotImplementedError`; INLA's
`return_latent_posteriors=True` also raises. `unmarginalize` can restore the
generative model but cannot manufacture posterior field draws. If those are
required, fit the original joint model (or implement and validate a separate
conditional inference calculation), rather than call prior draws recovered
posteriors.

### Likelihood and prediction after elimination

- Call `pm.compute_log_likelihood(idata, model=marginal_model)` explicitly when
  using its likelihood. Confirm the returned factors represent the intended
  prediction/validation units, not just the old observed-variable names.
- Exact independent allocation/Normal elimination can give observationwise
  marginal likelihoods. Shared eliminated latents induce joint factors:
  enumeration can assign a nonseparable joint term to one dependent variable
  and zeros to others; Laplace currently returns a summed joint term, not
  elementwise observation logp. Do not feed those bookkeeping entries to ordinary
  observationwise LOO. Choose a valid group/held-out unit and derive the needed
  factorization or refit.
- Conditional likelihood on recovered training latents and marginal likelihood
  integrating fresh latents answer different predictive questions. State whether
  prediction concerns the same fitted units or new units/groups. For same-unit
  response replication, recover exact latents and use the original model as
  above. For new units, build the corresponding prediction model and intentionally
  draw new latents conditional on retained posterior hyperparameters.
- The marginal model retains a generative graph that can draw eliminated latents
  from their prior conditional on retained parameters during forward prediction.
  That is appropriate for fresh exchangeable units, not automatic posterior
  recovery of a fitted field. In particular, a Laplace marginal posterior plus
  prior field draws is not a same-field PPC.
- Compare prior simulation, log densities, likelihood grouping and predictions
  with the intended original model on a tractable case. Elimination changes
  computation; it does not repair label switching, nonidentification or model
  misspecification.

## Zero inflation and hurdles

`psi` means body-component probability, not structural-zero probability.

| Family | Observation mechanism |
|---|---|
| ZeroInflatedPoisson/Binomial/NegativeBinomial | P(0)=1-psi+psi*g(0); positive mass psi*g(y). Body can itself generate zeros. |
| HurdlePoisson/NegativeBinomial | P(0)=1-psi; positive mass psi*g(y)/(1-g(0)). mu is the untruncated mean. |
| HurdleGamma/LogNormal | Atom 1-psi at zero, positive density psi*f(y). Gamma beta is rate; LogNormal mu/sigma are log-scale. |

For ZIP, mean is psi*mu and variance psi*mu+psi*(1-psi)*mu². Continuous positive
bodies have no atom at zero and do not require discrete zero-truncation
normalization. In PyMC 6.3.1 source the continuous hurdle body is not truncated at
machine epsilon despite inconsistent docstrings. Model exact, rounded and censored
zeros differently. Mixed atom/continuous free variables are not ordinary NUTS
parameters; an observed hurdle likelihood for continuous parameters is different.

Sources: [PyMC mixture implementation](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/mixture.py),
[extras marginalization](https://www.pymc.io/projects/extras/en/stable/api/marginalization.html),
[release marginalization source](https://github.com/pymc-devs/pymc-extras/tree/v0.15.1/pymc_extras/model/marginal),
[exact Normal rewrite](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/marginal/distributions/normal.py),
[Laplace rewrite](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/marginal/distributions/laplace.py),
[conditional recovery](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/marginal/conditional.py),
[release conjugacy tests](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/tests/model/marginal/test_normal.py),
[PyMC prediction source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/sampling/forward.py),
[extras dependency metadata](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pyproject.toml).
