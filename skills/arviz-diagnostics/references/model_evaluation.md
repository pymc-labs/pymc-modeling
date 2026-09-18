# Reliable predictive evaluation

This reference uses modern ArviZ DataTree workflows. API details below describe
the ArviZ 1.3 family; check installed signatures when adapting to another release.

## Define what is being predicted

Declare the response, units/transform, observed-variable name, observation
identity/order, training information, validation unit and utility before scoring.

- **New exchangeable row:** leave one row out; retain each conditional likelihood
  contribution across `chain`, `draw` and observation dimensions.
- **Entirely new group:** hold all group members out together; integrate unseen
  random effects instead of conditioning on their full-data posterior. A sum of
  marginal response log scores differs from one joint group log density.
- **Future outcomes:** train only on the past at each forecast origin. Specify
  horizon and whether forecasts update after each response. Random k-fold or
  fitting both past and future around an omitted time block is not forecasting.
- Fit transformations, feature selection and model selection inside training
  splits. Scores on different response measures need the transformation
  Jacobian; `az.loo(..., log_jacobian=...)` accepts pointwise log absolute
  forward-transform derivatives in the modern API.

## Prepare pointwise likelihoods and predictions

```python
import arviz as az
import pymc as pm

# dt contains actual posterior draws; model has observed response y.
pm.compute_log_likelihood(dt, model=model)
pp = pm.sample_posterior_predictive(dt, model=model, var_names=["y"])
dt["posterior_predictive"] = pp["posterior_predictive"]

loo = az.loo(dt, var_name="y", pointwise=True)
print(loo.elpd, loo.se, loo.p, loo.good_k, loo.warning)
```

Do not pass `compute_log_likelihood=True` to sampling. Check finite values where
the mathematical likelihood is finite, observation units and coordinate alignment
with observed/predictive responses. Do not silently sum observation axes or
concatenate unrelated likelihood variables. `log p(y_i | theta_s)` differs from
posterior predictive `log p(y_i | y)` and prior `log p(theta_s)`.
`pm.stats.compute_log_prior(dt, model=model)` optionally prepares prior densities;
it does not compute marginal likelihood.

ArviZ 1.3 `ELPDData` fields are `.elpd`, `.p`, `.elpd_i`, `.se`, `.good_k` and
`.pareto_k`, not historical `.elpd_loo`, `.p_loo` or `.loo_i`. ArviZ no longer
provides `waic`; use LOO rather than inventing an alias. A single-observation
`loo_i` selection accepts a positional index or coordinate mapping; its scalar
`se=0` is an undefined-SE sentinel, not perfect precision.

## Diagnose PSIS before interpreting scores

The sample-size-dependent threshold is

\[
k_{good}=\min\{1-1/\log_{10}(S),\;0.7\}.
\]

Use the result's `good_k`, all pointwise k values and warnings, not only a fixed
0.7 count. Let ArviZ estimate relative efficiency for trustworthy MCMC; `reff=1`
is appropriate for genuinely independent draws, not a generic speed option.

High Pareto k means the full posterior is a poor importance proposal for a
leave-one-out target. It does not alone establish bad MCMC, an erroneous row or
model misspecification. Inspect influential observations and the model without
deleting data or changing the threshold merely to obtain a clean result.

```python
az.plot_khat(loo)
flagged = loo.pareto_k > loo.good_k

# Optional: a supported PyMC model adapter and original pointwise LOO are required.
loo_mm = az.loo_moment_match(dt, loo_orig=loo, model=model)
```

Moment matching transforms/reweights existing draws; it **does not refit** and
need not fix high k. Callback implementations require unconstrained draws,
correct full posterior and pointwise likelihood densities, including transform
Jacobians. `az.loo(..., moment_match=True, model=model)` is a convenience route
when the installed adapter supports the model.

`az.reloo` requires an implemented `SamplingWrapper`, not just a DataTree and
`model=`. The wrapper must genuinely select data, refit, convert results and
evaluate excluded-data likelihoods. `az.loo_kfold(data, wrapper, k=...,
group_by=...)` likewise requires real refits; `folds=` and `stratify_by=` provide
alternative split controls. A stored full fit cannot perform cross-validation
on its own.

In the ArviZ 1.3 refit representation, corrected k is set to zero and stored log
weights to NaN. Zero is a **sentinel**, not measured PSIS success. Corrected
ELPD can be compared, but those weights cannot produce corrected PIT, moments
or CRPS. Generate leave-out predictive draws from the refits. Refitting removes
the full-posterior importance approximation, not finite Monte Carlo integration
error. Check predictive precision separately.

## Compare paired predictive performance

Compare the same response rows in the same order under the same target. Let
`d_i=elpd_i(A)-elpd_i(B)`; report `sum(d_i)` and paired uncertainty. A conventional
sample-variance SE is `sqrt(n*var(d_i, ddof=1))`, not the square root of two
independent model-SE squares. ArviZ 1.3 uses `ddof=0` for its nonsubsampled paired
variance; label independently calculated alternatives rather than treating this
finite-sample convention as a pairing error.

```python
comparison = az.compare(
    {"model_a": loo_a, "model_b": loo_b},
    method="stacking", round_to="none",
)
print(comparison)
az.plot_compare(comparison)
```

There is no `scale=` or `ic=` in this API. `elpd_diff` is model minus reference,
normally nonpositive. Current diagnostic columns include `diag_elpd` and
`diag_diff`, not a historical `warning` column. Inspect the returned table and
original pointwise diagnostics, not legacy column assumptions.

Normal or Bayesian-bootstrap intervals for ELPD differences have small-sample,
skewness and overlapping-training-set limitations; they are not posterior
parameter credible intervals. Small N, close predictions, clustered or serial
outcomes need particular care. There is no universal `|deltaELPD|<4` equivalence
rule. Report uncertainty and decision-relevant predictive differences, not only
rank or a threshold crossing.

## Learn from adaptive model development

Model comparison is also a way to understand assumptions. Compare scientifically
shared estimands, uncertainty and predictions across useful variants, even when
ELPD differences are small. Retain the motivation for consequential changes and
relevant earlier results; a model-development history is not the probabilistic
dependency graph of a single model.

Revising a prior, likelihood or structure after a PPC is legitimate exploration,
but the resulting sequence is data-dependent. Reliable CV estimates for individual
candidates do not make the maximum score across an adaptive search an unbiased
estimate of the winner's performance. Nested selection handles a reproducible
selection procedure; a held-out set repeatedly consulted for revisions is no
longer untouched confirmation.

Do not keep a bad model merely to avoid data reuse, or average known bugs into
predictions. Explain the scientific reasons for revisions. When many plausible
variants perform similarly, examine differences in the estimand or decision;
consider a richer encompassing model or predictive stacking where appropriate.
Use genuinely held-out or nested evaluation for claims about a selection procedure,
and do not present the selected model as uniquely confirmed.

## Stacking and predictive mixtures

Stacking maximizes cross-validated log score of a mixture over the candidate
set. Similar weights **do not establish equivalence**; weights are not posterior
model probabilities and zero weight does not prove a model false. Redundancy,
complementarity, model-set composition and finite data all affect weights.
`method="BB-pseudo-BMA"` is an alternative based on Bayesian-bootstrap ELPD
weighting, not posterior model evidence.

`az.weight_predictions` mixes predictive draws, not fitted parameters. Reorder
DataTrees to match comparison-table rows and their corresponding weights; check
observed data and coordinates yourself. ArviZ 1.3 implementations should not be
relied on to enforce observation equality. Integer allocation of draws means
finite mixtures need not have exactly the requested fractions; use the realized
sample count. Reset a resulting `sample` MultiIndex if required for NetCDF:

```python
mixed["posterior_predictive"] = (
    mixed["posterior_predictive"].to_dataset().reset_index("sample")
)
```

Full-data predictive mixtures with fitted CV weights are not independent
validation of the weight-selection procedure. Use held-out or nested evaluation
for that claim.

## Predictive quantities and calibration

Use the observation-generating variable, not just a latent mean, when the target
is a new observation. Match likelihood and predictive variable names and retain
function-specific importance diagnostics.

| Function | Interpretation and modern API cautions |
|---|---|
| `loo_expectations` | Returns `(expectation, function_specific_khat)`. Mean, median, variance, SD, quantiles and circular variants can have reliability different from the log score. |
| `loo_metrics` | RMSE, MAE, MSE, accuracy (`acc`) or balanced accuracy (`acc_balanced`); estimate/SE named tuple. Classification requires a binary/categorical target. |
| `loo_r2` | Requires `var_name`; summary named tuple by default, bootstrap values with `summary=False`. Set `ci_prob=.89, ci_kind="eti"` explicitly if needed. |
| `loo_score` | `kind="crps"` or `"scrps"`, not `score_func=`. CRPS is negated: larger is better. Log score is LOO's `.elpd`. `pointwise=True` preserves scores and k arrays. |
| `loo_pit` | Returns a Dataset. Discrete outcomes need randomized ties (`random_state=`); marginal uniformity is not conditional calibration. |
| `loo_influence` | Returns `(shift, function_khat)` for parameter/predictive summaries. Influence is not evidence that a row is erroneous. |
| `loo_subsample` | `observations=` accepts a count or indices; use `seed=` and retain subsampling SE. A difference estimator, not a sum over selected rows. |
| `update_subsample` | `observations=` is the number of additional observations, not the final total. |
| `loo_approximate_posterior` | `log_p` and `log_q` must be evaluated at the same proposal draws, in the same measure as the likelihood. Importance correction still needs reliable tails. |

`plot_loo_interval` uses `ci_probs=(.5,.89)` and `ci_kind="eti"`;
`plot_loo_pit` addresses PIT calibration. `plot_loo_pava` is for binary,
categorical or ordinal outcomes, not continuous Normal responses. Modern plots
return PlotCollection objects with `savefig(...)`.

Distribution comparisons answer other questions: KL is directional; Wasserstein
is unit/scale dependent. Neither is a predictive score or equivalence test.
`ci_in_rope` is the percentage of samples **inside the selected credible
interval** that also lie inside the ROPE, not total posterior ROPE probability
or interval-width overlap. Specify the interval and practical units explicitly.

## Grouped and temporal uncertainty

For grouped holdouts, retain whole groups and decide whether scores are marginal
per response or joint per group. Observation-level SE is not cluster-robust
uncertainty; a handful of groups cannot support precise generalization claims.
For new groups with latent effects, integrate over their distribution rather
than reusing full-data estimates.

For rolling/expanding forecasts, score only future rows at each origin. Marginal
horizon log scores differ from a joint multihorizon density, which includes
posterior-induced covariance. Serial outcomes and overlapping training windows
invalidate an independent-row normal interval. `loo_kfold` does not automatically
implement leave-future-out forecasting.

## From predictions to actions

A predictive score is not automatically the decision maker's utility. For an action
recommendation, establish feasible actions, consequences, constraints and loss or
utility, then integrate over posterior and future-outcome uncertainty. Assess
sensitivity to plausible models and preferences. Do not replace this with an ELPD
rank, a sign probability or whether a credible interval excludes zero.

When the analyst does not own the decision or values are unspecified, hand off
interpretable uncertainty summaries or draws. Do not invent costs or utilities,
and distinguish an inference report from an action recommendation.

## Marginal likelihood is a different question

`exp(ELPD_A-ELPD_B)` is not a Bayes factor. A Bayes factor compares marginal
likelihoods under proper priors and can be strongly prior-sensitive. ArviZ's
`bayes_factor` uses a Savage–Dickey KDE density-ratio estimate for appropriate
nested nulls, not bridge sampling. For `H0: theta=theta0`, `BF01` is posterior
density divided by prior density at `theta0` only when the identity's conditions
hold, including compatible conditional nuisance priors. Check support, proper
priors, density-estimation accuracy and numerator/denominator orientation.
An analytic conjugate calculation can expose KDE error in a simple model.

## Primary sources

- [ArviZ Stats API](https://python.arviz.org/projects/stats/en/latest/api/index.html)
  and [source](https://github.com/arviz-devs/arviz-stats) for version-specific
  result fields, refit wrappers, predictive metrics and weighting behavior.
- [Vehtari et al.: Pareto Smoothed Importance Sampling](https://jmlr.org/papers/v25/19-556.html).
- [Yao et al.: stacking predictive distributions](https://arxiv.org/abs/1704.02030).
- [Sivula et al.: uncertainty in LOO model comparison](https://arxiv.org/abs/2008.10296).
- [Paananen et al.: implicitly adaptive importance sampling](https://arxiv.org/abs/1906.08850).
- [Heck: Savage–Dickey density-ratio caveats (abstract)](https://pubmed.ncbi.nlm.nih.gov/30451277/).
  DOI: `10.1111/bmsp.12150`.
- [Bayesian Workflow](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf)
  (2026 corrected edition, §7.3 and §§9.2–9.6).
