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

## Exact discrete marginalization with pymc-extras

Check the installed extras release and dependency requirements. In extras 0.14.0,
PyMC <6.3 and PyTensor <3.3 are required; do not override bounds to combine packages.
The following interface description is version-specific, not a compatibility promise.

```python
import pymc_extras as pmx
from pymc_extras.marginal import conditional, recover, unmarginalize

# original contains comp ~ Categorical(w) and its observed dependent likelihood.
marginal_model = pmx.marginalize(original, ["comp"])
idata = pm.sample(model=marginal_model, nuts_sampler="pymc",
                  tune=tune, draws=draws, chains=chains, random_seed=42)
idata.to_netcdf("posterior.nc")
allocations = recover(idata, model=marginal_model, var_names=["comp"],
                      extend_inferencedata=False, random_seed=43)
conditional_model = conditional(marginal_model, ["comp"])
log_conditional = conditional_model.compile_logp(
    vars=[conditional_model["comp"]], sum=False)
restored = unmarginalize(marginal_model, ["comp"])
```

For a nonempty selection, marginalize returns a new model. `recover` defaults to
extending/returning the supplied DataTree; with `extend_inferencedata=False` it
returns an xarray Dataset. Empty selections can return their input unchanged.
`conditional` creates a refactorized joint model: select comp's factor for its
conditional probabilities, not the full joint logp. Multiple recoveries follow
a chain rule, not arbitrary independent full conditionals. `unmarginalize`
restores the original prior, not the conditional. Use `recover`, not deprecated
`recover_marginals`; `return_samples=False` is not a probability-array API.

Enumerable rewrites in 0.14 recognize Bernoulli, Categorical and DiscreteUniform,
with constant-foldable support sizes/bounds. A separate supported Markov-chain
rewrite is not a guarantee for arbitrary HMM graphs.

- Marginalized variables must be free, not observations relabelled as latent.
- Registered dependent Deterministics/Potentials can be rejected; an unrecorded
  expression such as `mu[comp]` is a different graph contract.
- Batched observationwise dependencies must remain supported. Cross-observation
  mixing is not generic; scalar splitting can create exponential graph growth.
- Infinite-support Poisson latents are not finite enumeration. Do not silently
  truncate or substitute a different model.
- Normal-Normal conjugate and Laplace-approximate elimination are distinct from
  arbitrary exact continuous marginalization.
- Inspect warnings about dependent transform bounds; do not suppress them.

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
[extras source](https://github.com/pymc-devs/pymc-extras/tree/v0.14.0/pymc_extras/model/marginal),
[extras dependency metadata](https://github.com/pymc-devs/pymc-extras/blob/v0.14.0/pyproject.toml).
