# Elicitation, units and sensitivity

## Elicit statements without changing their meaning

Ask for a quantity's definition, units, time horizon, population, conditioning
information, support, quantiles and tail probabilities. Record who supplied each
statement and what information was available. A verbal “usually” is not permission
to invent a numerical probability. Label numerical teaching assumptions as such.

A practical expert-elicitation sequence is: define the quantity; discuss plausible
extremes; elicit median and quartiles; ask about tails; fit candidate families;
show the distributions back to the expert; then show simulated observations.
Preserve disagreement rather than averaging experts without an explicit pooling
method. A physical boundary differs from a soft probability bound: truncate only
for justified support restrictions, not to erase disagreeable predictions.

## Translate to the parameter scale

For a Normal with **central** mass `m` in `[L,U]`,

\[
\mu=(L+U)/2,\qquad s=(U-L)/(2\Phi^{-1}((1+m)/2)).
\]

- HalfNormal: `s=U/Phi_inverse((1+m)/2)` gives `P(X<U)=m`.
- Exponential: rate `lambda=-log(1-m)/U` gives `P(X<U)=m`.
- LogNormal: `log(X/u)` is Normal for declared unit reference `u`. Median is
  `u*exp(mu)`; `sigma` is log-scale SD, not outcome SD. Compatible quartiles obey
  `q2**2=q1*q3`; then `mu=log(q2/u)` and
  `sigma=(log(q3)-log(q1))/(2*Phi_inverse(.75))`. Incompatible quantiles require a
  more flexible family or an explicit compromise; inspect each achieved CDF.
- Gamma: shape/rate `alpha=mean**2/sd**2`, `beta=mean/sd**2`. SciPy's
  `scale=1/beta`. PyMC accepts `alpha,beta` or `mu,sigma`, not `alpha,mu`.
- Beta: mean `m` and concentration `k` imply `alpha=m*k`, `beta=(1-m)*k`,
  variance `m*(1-m)/(k+1)`. “50-50” does not determine concentration.
- Student-t: `sigma` is scale; for `nu>2`, SD is `sigma*sqrt(nu/(nu-2))`.
  Equal central mass does not imply equal variance or tail behavior.

For positive unit multiplier `c`, location/scale become `c*mu,c*sigma`.
LogNormal numeric parameters become `mu+log(c),sigma`. Differential entropy
changes by `log(c)`; nonlinear reparameterization can change the maximizing
family member. State the family and reference measure: there is no universal
“least informative” distribution from a single interval constraint.

For `mu=intercept+beta*x`, under `x_std=(x-cx)/sx`, `y_std=(y-cy)/sy`,

```
beta_original = beta_std * sy / sx
intercept_original = cy + sy * intercept_std - beta_original * cx
```

The standardized intercept conditions on `x=cx`. With log/logit links a contrast
`dx` gives rate/odds ratio `exp(beta*dx)`, not additive probability change.
Outcome-based scaling is an empirical modeling choice, not independent prior
information. No Normal(0,10), HalfCauchy or other family is universally “weak.”

## PreliZ: fit, inspect, translate

```python
import preliz as pz

# Illustrative mean change, mg/L: central 90% between -2 and 2.
fitted = pz.maxent(pz.Normal(mu=0.0),
                   lower=-2.0, upper=2.0, mass=0.90, plot=False)
print(fitted.opt)
print(fitted.cdf(2.0) - fitted.cdf(-2.0))

# Illustrative duration, hours: compatible lognormal quartiles.
duration = pz.quartile(pz.LogNormal(), q1=1.0, q2=2.0, q3=4.0, plot=False)
print([duration.cdf(q) for q in (1.0, 2.0, 4.0)])
```

In PreliZ 0.28.0, `plot=False` returns the fitted distribution; `plot=True`
returns a distribution/axes tuple. Objects are updated in place, and fully fixed
distributions have nothing to optimize. Specify mass and plotting explicitly.
Check `.opt.success`, warnings, achieved mass and each quantile; a plausible
number alone is insufficient. Pass supported parameters explicitly into PyMC,
e.g. `pm.Normal("theta", mu=fitted.mu, sigma=fitted.sigma)` inside a model.
The result is not a PyMC distribution or parameter dictionary.

Maxent optimizes within the selected family. In 0.28.0 the `fixed_stat` optimizer
folds its statistic residual into the mass constraint; independently check both
requirements rather than assuming separate guarantees. `pm.find_constrained_prior`
is deprecated in PyMC 6.3.1 but not absent: it requires scalar parameters/log-CDF
and minimizes lower-tail error subject to interval mass, not entropy.

### Interactive and predictive elicitation

PreliZ 0.28.0 provides `pz.Roulette(...)`, a notebook/widget class, not lowercase
`pz.roulette`. Its `.inputs` and `.dist` start as `None`; actual chip allocations
must precede interpreting fitted output. Preserve and independently check them.

`pz.predictive_explorer(fmodel, samples=..., engine="pymc",
group="prior_predictive", var_name=...)` explores prior inputs interactively.
Its `samples` spelling is unrelated to PyMC's `draws` argument.
`pz.predictive_finder` is not exported in 0.28.0.

`pz.ppe(model, target, engine="pymc", random_state=...)` is experimental
projective predictive elicitation. It proposes prior code as a string, not a
validated model. Elicit the predictive target independently; never substitute
held-out outcomes. Review the proposal rather than automatically executing the
string, and repeat prior predictive checks.

## Complete prior predictive example

Before inspecting outcomes, declare these **illustrative**, not expert-supplied,
assumptions for a regression in predetermined standardized units:

- The intercept has central 95% prior mass in [-1, 1].
- The change in latent mean from x=-1 to x=1 has central 95% prior mass in [-1, 1].
- Intercept and slope are independent Normals; residual SD is known to be 0.5.

The predictor contrast is `2*beta`, so the slope SD is half the intercept SD.
This script constructs its own 80-row synthetic dataset; it does not estimate
standardization or prior bounds from the realized outcomes. The generating slope
is deliberately larger than the stated prior contrast allows with high
probability, making it important to distinguish assumptions from the simulator.
NetCDF output needs a compatible backend, such as `h5netcdf` with `h5py`;
the saved figure uses Matplotlib.

```python
import arviz as az
import numpy as np
import pymc as pm
from scipy.stats import norm

seed = 20260911
rng = np.random.default_rng(seed)

# Translate the declared central masses before inspecting simulated outcomes.
z = norm.ppf(0.975)
alpha_sd = 1.0 / z
beta_sd = 1.0 / (2.0 * z)
noise_sd = 0.5
alpha_mass = norm.cdf(1.0, scale=alpha_sd) - norm.cdf(-1.0, scale=alpha_sd)
contrast_mass = (
    norm.cdf(1.0, scale=2.0 * beta_sd)
    - norm.cdf(-1.0, scale=2.0 * beta_sd)
)
print(f"Analytic P(-1 <= alpha <= 1): {alpha_mass:.12f}")
print(f"Analytic P(-1 <= 2*beta <= 1): {contrast_mass:.12f}")

x = np.linspace(-1.0, 1.0, 80)
y = 0.5 + 1.2 * x + rng.normal(0.0, noise_sd, 80)
coords = {"obs": np.arange(x.size)}

with pm.Model(coords=coords) as model:
    x_data = pm.Data("x", x, dims="obs")
    alpha = pm.Normal("alpha", mu=0.0, sigma=alpha_sd)
    beta = pm.Normal("beta", mu=0.0, sigma=beta_sd)
    mu = pm.Deterministic("mu", alpha + beta * x_data, dims="obs")
    pm.Normal("response", mu=mu, sigma=noise_sd, observed=y,
              shape=x_data.shape, dims="obs")
    prior = pm.sample_prior_predictive(draws=1000, random_seed=seed)

prior.to_netcdf("prior_predictive_raw.nc")  # Preserve simulation before analysis.
replicated = prior["prior_predictive"]["response"]
dataset_mean = replicated.mean("obs")
dataset_range = replicated.max("obs") - replicated.min("obs")
for label, statistic in (("Dataset mean", dataset_mean),
                         ("Dataset range", dataset_range)):
    interval = statistic.quantile([0.055, 0.5, 0.945], dim=("chain", "draw"))
    print(f"{label}: 5.5%, median, 94.5% = {interval.values}")

# One event per independent parameter/dataset draw, not per observation.
contrast = 2.0 * prior["prior"]["beta"]
inside = (contrast >= -1.0) & (contrast <= 1.0)
n_datasets = inside.size
probability = inside.mean().item()
probability_mcse = np.sqrt(probability * (1.0 - probability) / n_datasets)
print(f"P(-1 <= 2*beta <= 1): {probability:.4f}; "
      f"binomial MCSE={probability_mcse:.4f}; {n_datasets} prior datasets")

prior_plot = az.plot_ppc_dist(
    prior, group="prior_predictive", var_names=["response"], kind="ecdf",
    backend="matplotlib",
)
prior_plot.savefig("prior_predictive.png", bbox_inches="tight")
print("Inspect prior_predictive.png against the declared scale assumptions.")
```

The analytic central masses are 0.95. The simulation estimates the contrast
probability with binomial Monte Carlo uncertainty based on 1,000 independent
parameter draws, not 80,000 correlated-within-dataset observations. Choose the
simulation count for the tail precision needed; a plug-in binomial MCSE can be
misleading for rare events with no simulated occurrences.

Open the saved ECDF plot and inspect location, spread and tails alongside the
dataset summaries. A marginal ECDF can hide predictor-conditional disagreement,
so retain the explicit slope contrast. In an application, also check physical
support, dependence and group variation. The latent-mean contrast excludes
residual noise; replicated observations include its known SD of 0.5.

If these predictions are implausible, return to the declared intercept/contrast
bounds, independence, noise assumption or likelihood and record the scientific
reason for a revision. Avoid outcome-extrema tuning, seed search and clipping.
This is prior elicitation and plausibility checking, not posterior fitting.

## Sensitivity to reasonable alternatives

Hold likelihood, observation identities, transformations and estimand fixed.
Vary plausible scale **and** tail assumptions. For example, compare a baseline
Normal, a wider Normal and a Student-t matching the baseline central mass; these
are distinct scientific assumptions, not guaranteed interchangeable priors.

Diagnose every fit before comparing physical-scale means, 89% ETIs, meaningful
event probabilities such as `P(theta > threshold)`, and new-observation
predictions. Report MCSE of these quantities; for independent fits, the MCSE of
a difference uses the square root of summed MCSE squares. A prior-posterior plot
helps communicate changes but does not replace numerical sensitivity assessment.
Stable answers over the considered alternatives or strong posterior contraction
do not prove data dominance, global robustness, calibration or causality.

## Power sensitivity is local

Explicitly prepare log-density groups before using ArviZ power diagnostics:

```python
import arviz as az

pm.stats.compute_log_prior(idata, model=model)
pm.compute_log_likelihood(idata, model=model)
sensitivity = az.psense_summary(idata)
```

`psense`, `psense_summary`, `plot_psense_dist` and `plot_psense_quantities` have
PyMC facades under `pm.stats`/`pm.plots` as well. Check installed versions if
single-variable power plots fail during DataArray/group handling; preserve the
numeric result and report the plotting error rather than adding dummy variables.
Consult the [stats source](https://github.com/arviz-devs/arviz-stats/blob/main/src/arviz_stats/psense.py)
and [plot API](https://arviz-plots.readthedocs.io/en/latest/api/index.html).

For posterior draws from the reference target, raw log weights are
`(alpha-1)*log_prior` or `(alpha-1)*sum(pointwise_log_likelihood)`. Normalize
with log-sum-exp. Inspect Pareto k, weight concentration and importance-weight
ESS, which is not MCMC ESS. `power_scale_lw` supplies smoothed weights; `psislw`
can diagnose importance tails without implying the two smoothing procedures are
identical. Reweighting cannot recover unsupported regions or undiscovered modes.
The package's 0.05 sensitivity heuristic is not a scientific acceptance threshold.

A useful analytic check is known-SD Normal data with prior `N(m0,tau**2)`:

\[
V=(a_p/\tau^2+a_l n/\sigma^2)^{-1},\qquad
M=V(a_p m_0/\tau^2+a_l\textstyle\sum_i y_i/\sigma^2).
\]

The powered posterior is `Normal(M,sqrt(V))`. Positive-power refits using
`tau/sqrt(a_p)` and `sigma/sqrt(a_l)` have this posterior up to constants in
theta. They are diagnostic auxiliary targets, **not** newly justified observation
noise models. Predict a new observation with the original measurement variance:
mean `M`, variance `V+sigma**2`. For nonconjugate one-dimensional targets,
adaptive quadrature with error estimates can provide an independent check;
a finite grid is not exact integration. Compare substantive effects and actual
refits when reweighting is unreliable or a local result is consequential.

## Primary sources

- [PreliZ maxent](https://github.com/arviz-devs/preliz/blob/0.28.0/preliz/unidimensional/maxent.py),
  [optimization](https://github.com/arviz-devs/preliz/blob/0.28.0/preliz/internal/optimization.py)
  and [quartile](https://github.com/arviz-devs/preliz/blob/0.28.0/preliz/unidimensional/quartile.py).
- [PreliZ Roulette](https://github.com/arviz-devs/preliz/blob/0.28.0/preliz/unidimensional/roulette.py),
  [predictive explorer](https://github.com/arviz-devs/preliz/blob/0.28.0/preliz/predictive/predictive_explorer.py)
  and [projective predictive elicitation](https://github.com/arviz-devs/preliz/blob/0.28.0/preliz/predictive/ppe.py).
- [PyMC 6.3.1 constrained-prior implementation](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/func_utils.py)
  and [continuous distributions](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/continuous.py).
- [ArviZ sensitivity discussion](https://arviz-devs.github.io/EABM/Chapters/Sensitivity_checks.html)
  and [Kallioinen et al.](https://doi.org/10.1007/s11222-023-10366-5).
