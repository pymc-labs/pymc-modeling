# Hierarchical models: groups, information and parameterization

Use a hierarchy when groups plausibly share a population distribution and partial
pooling answers the question. State exchangeability and the estimand before adding
random effects. For group means, a basic model is

\[
y_i\mid\alpha_{j[i]}\sim N(\alpha_{j[i]},\sigma^2),\qquad
\alpha_j\mid\mu,\tau\sim N(\mu,\tau^2).
\]

Here tau is between-group **SD**, passed as `sigma=tau`; the `Normal(tau=...)`
keyword instead means precision. Estimate observation sigma unless it is genuinely
known. Put population-location and scale priors in the response's units and check
their implied group and observation variation before fitting.

## Labels are not indices

Preserve an explicit ordered group coordinate and label-to-code mapping. Validate
integer `group_idx` in `[0, J)` and the round trip
`groups[group_idx] == original_group_labels` for every row. A numeric group label
is not an array position; missing code `-1` silently selects the last group.
Require distinct group labels, unique observation IDs and aligned finite outcomes.
Check saved coordinates, observed data and draw-wise
`mu_obs == alpha[..., group_idx]` after inference.

Do not refactorize future labels independently. Known groups reuse their training
mapping. A new group requires a new effect drawn conditional on each posterior
`mu,tau`; substituting mu or another group's index drops between-group uncertainty.
Prior-only groups are valid when their prediction target is explicit.

## Centered and non-centered forms

Choose one branch inside a model with appropriate hyperpriors and coordinates:

```python
# Centered
alpha = pm.Normal("alpha", mu=mu, sigma=tau, dims="group")

# Non-centered alternative
z = pm.Normal("z", 0, 1, dims="group")
alpha = pm.Deterministic("alpha", mu + tau*z, dims="group")

# Shared observation mapping
mu_obs = pm.Deterministic("mu_obs", alpha[group_idx], dims="obs_id")
pm.Normal("response", mu=mu_obs, sigma=sigma, observed=y, dims="obs_id")
```

The non-centered generative construction already induces the intended Normal
prior: do not add a second density or manual Jacobian. To compare native compiled
densities, however, account for their different measures:

\[
\log p(\mu,\log\tau,z,y)-\log p(\mu,\log\tau,\alpha,y)=J\log\tau,
\qquad \alpha=\mu+\tau z.
\]

Both include the same log-transform Jacobian for tau. Equal scientific priors do
not imply identical native log densities or identical Monte Carlo draws.

For Gaussian group means, the information ratio
`I_j = n_j*tau**2/sigma**2` compares likelihood and hierarchical-prior precision.
With uncertain scales it is a posterior distribution, not a count-only rule.

- Weakly measured groups often favor non-centering: centered effects couple to a
  small population scale in a funnel.
- Strongly measured groups may favor centering: non-centering transfers tight
  likelihood constraints to combinations of mu, tau and z.
- Unequal information can favor mixed/partial non-centering; preserve the same
  model and explicit parameter mapping.
- For non-Gaussian models use actual likelihood curvature, predictors and support,
  not this ratio as a universal threshold.

Inspect divergences, rank R-hat, bulk/tail ESS, MCSE and plots for each fit. Plot
scales against effects/offsets; a small linear correlation does not rule out a
funnel. Diagnose scaling, support and identification before increasing tuning or
`target_accept`; non-centering cannot repair every pathology.

## Identification and population meaning

An intercept plus J unrestricted group dummies has J+1 coefficients but design
rank J: an intercept shift can be canceled by shifting all effects. Population
mu in the prior for group means is not a second likelihood intercept. Proper
priors can produce a posterior without supplying strong likelihood identification.
More rows within a few groups cannot replace more independent groups for every
population question; assess population-scale prior sensitivity.

For finite-population contrasts, use an explicit reference category or coherent
zero-sum prior. An arbitrary large Potential penalty is not exact identification.
A zero-sum construction changes the joint prior and intercept interpretation;
population mu in an exchangeable model need not equal the realized average alpha.

Varying slopes need within-group predictor variation and declared centering/units.
Correlated intercepts and slopes call for an appropriately scaled joint population
prior. Crossed/nested effects need separate indexing maps and connectivity/aliasing
checks, not a single group column reused indiscriminately.

## Quantify pooling

For known observation sigma and group mean ybar_j,

\[
w_j=\frac{n_j\tau^2}{\sigma^2+n_j\tau^2},\quad
E(\alpha_j\mid\mu,\tau,y)=w_j\bar y_j+(1-w_j)\mu,\quad
\operatorname{Var}(\alpha_j\mid\mu,\tau,y)=\frac{\tau^2}{1+n_j\tau^2/\sigma^2}.
\]

w is retained-data weight, 1-w is pooling weight. Smaller groups pool more for the
same hyperparameter draw and noise SD. Integrate the conditional mean over joint
posterior draws; inserting separate marginal means ignores dependence. Report
raw means, partially pooled means/intervals and conditional weights, distinguishing
likelihood-only reference estimates from additional Bayesian fits. Do not require
every estimate to lie between its raw mean and one empirical grand mean.

The residual `alpha - conditional_mean` has conditional mean zero and its squared
value divided by conditional variance has mean one. These provide implementation
checks with Monte Carlo uncertainty; they do not certify predictive calibration
or require intervals to contain every generating value.

Sources: [hierarchical HMC geometry](https://arxiv.org/abs/1312.0906),
[non-centering methodology](https://mc-stan.org/docs/2_37/stan-users-guide/efficiency-tuning.html#hierarchical-models-and-the-non-centered-parameterization),
[PyMC model implementation](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model/core.py),
[rank diagnostics](https://doi.org/10.1214/20-BA1221).
