# Bayesian model workflow

## Formulate before computing

State the population, observation unit, estimand or prediction target, units,
support, dependence, missingness and selection assumptions. Preserve original
row identities through transformations. Known measurement noise is appropriate
only when genuinely supplied by the measurement process; estimate unknown noise.
Separate a scientific model assumption from a convenience used for computation.

Use priors in interpretable units. Translate domain statements into parameter
and outcome implications, then run `pm.sample_prior_predictive(draws=...)`.
Inspect tails, impossible outcomes, group variation and decision-relevant
features, not only an aggregate min/max. Proper priors alone do not establish
identifiability; sensitivity to plausible priors matters when data are weak.

## Learn from related models

Start with a useful scaffold or an established subject-matter model. Sometimes
the best route is to simplify a larger target model or explore alternative
scientific explanations, rather than only add complexity. Change components
deliberately so their effects are understandable; do not remove known design,
measurement or dependence structure merely to obtain the smallest model.

Keep a compact model history alongside the analysis: what changed, why, its
computational status, and the effect on important estimands and predictions.
Retain relevant earlier model code/results to understand later variants, even
when predictive scores barely change. Relationships among model variants are
different from the probabilistic dependency graph within one model.

Revising assumptions after seeing data is legitimate model development, not a
pre-specified analysis. Explain the revision and preserve uncertainty across
scientifically viable alternatives. Fix bugs rather than averaging them into a
prediction; do not blindly select the maximum CV score from an adaptive search.

## Inspect the model

Check the registered likelihood, observation alignment and event/batch dimensions.
Use `model.initial_point()`, `model.point_logps()` and `model.debug()` to locate
invalid factors; compile log density and derivatives for precise numerical checks.
Independent small-case oracles are especially valuable for custom likelihoods.
Set numeric tolerances from dtype and conditioning, not after seeing errors.
Explicit float64 parameters may still leave lower-precision intermediate constants;
inspect the graph when testing normalization at tight tolerances.

## Fit and preserve the result

Use short preliminary fits or inexpensive approximations to expose construction,
geometry and gross-fit problems before paying for precise inference. Label their
results exploratory: incomplete diagnostics can guide the next investigation,
but do not support reliable posterior summaries or decisions. Do not spend a
final-analysis budget polishing an evidently unsuitable model.

Before a large fit, pilot the intended model size and hardware, including
compilation, sampling and concurrent-chain/predictive-array memory. For reportable
inference, adjust warmup from adaptation and geometry; around 1,000 iterations is
a starting convention, not a requirement or ceiling. Choose retained draws from
estimand-specific ESS and MCSE. Several independently initialized chains support
between-chain checks; many very short chains do not replace within-chain exploration.

Save before downstream calculations can fail, and keep the raw posterior separate
from enriched output. Reopen important saved results to check dimensions and
values. Saving after `pm.sample` returns cannot recover a process interrupted
during sampling; choose suitable incremental storage when that risk matters.

## Minimal complete regression

This standalone example generates 80 observations from an illustrative linear
Gaussian process. The observation noise SD is **known** to be 0.5 here; in real
work estimate unknown noise. Independent `Normal(0, 1)` priors for the intercept
and slope express illustrative assumptions in these units, not universal defaults.
The four-chain allocation demonstrates the workflow; adapt it using the budget
guidance above. It uses native PyMC so no optional sampler package is needed.

Run the block in the consuming environment with PyMC 6, ArviZ 1 and a Matplotlib
plotting backend. NetCDF output also needs a compatible backend, such as
`h5netcdf` with `h5py`. It writes separate raw, enriched and future-prediction
files in the working directory. Inspect the displayed or saved prior and PPC
plots for scale, tails and discrepancies.

```python
import numpy as np
import pymc as pm
import arviz as az

SEED = 20260911
rng = np.random.default_rng(SEED)
train_x = np.linspace(-1, 1, 80)
train_y = 0.5 + 1.2 * train_x + rng.normal(0, 0.5, 80)
train_ids = [f"t{i}" for i in range(80)]
known_noise_sd = 0.5

with pm.Model(coords={"obs_id": train_ids}) as model:
    x = pm.Data("x", train_x, dims="obs_id")
    alpha = pm.Normal("alpha", 0, 1)
    beta = pm.Normal("beta", 0, 1)
    mu = pm.Deterministic("mu", alpha + beta * x, dims="obs_id")
    pm.Normal(
        "response", mu, known_noise_sd, observed=train_y,
        shape=x.shape, dims="obs_id",
    )
    prior = pm.sample_prior_predictive(draws=1000, random_seed=SEED)

prior.to_netcdf("regression_prior.nc")
prior_mean = prior["prior_predictive"]["response"].mean("obs_id")
print("Prior-predictive dataset mean: 5.5%, median, 94.5%")
print(prior_mean.quantile([0.055, 0.5, 0.945], dim=("chain", "draw")))
prior_plot = az.plot_ppc_dist(
    prior, group="prior_predictive", var_names=["response"],
    kind="ecdf", backend="matplotlib",
)
prior_plot.savefig("regression_prior.png")
print("Inspect regression_prior.png against the stated assumptions before fitting.")

initial_point = model.initial_point()
initial_logps = model.point_logps(initial_point)
print("Initial factor log probabilities:", initial_logps)
if not np.isfinite(list(initial_logps.values())).all():
    model.debug(initial_point)
    raise ValueError("Inspect nonfinite initial factors before sampling.")

with model:
    idata = pm.sample(
        nuts_sampler="pymc", chains=4, cores=1, tune=1000, draws=1000,
        random_seed=SEED,
    )
idata.to_netcdf("regression_posterior.nc")  # Raw result before postprocessing.

print("Retained chains:", idata["posterior"].sizes["chain"])
print("Divergences by chain:")
print(idata["sample_stats"]["diverging"].sum("draw"))
summary = az.summary(
    idata, var_names=["alpha", "beta"], ci_prob=0.89, ci_kind="eti",
    round_to="none",
)
print("Posterior means, SDs, 89% ETIs, R-hat, ESS and MCSE:")
print(summary)

pm.compute_log_likelihood(idata, model=model)
idata.update(pm.sample_posterior_predictive(
    idata, model=model, var_names=["response"], random_seed=SEED,
))
idata.to_netcdf("regression_enriched.nc")
ppc_plot = az.plot_ppc_dist(
    idata, var_names=["response"], kind="ecdf", backend="matplotlib",
)
ppc_plot.savefig("regression_ppc.png")

new_x = np.array([-0.5, 0.0, 0.5])
new_ids = ["p0", "p1", "p2"]
try:
    pm.set_data({"x": new_x}, model=model, coords={"obs_id": new_ids})
    predictions = pm.sample_posterior_predictive(
        idata, model=model, predictions=True, var_names=["mu", "response"],
        freeze_vars=["alpha", "beta"], random_seed=SEED,
    )
    predictions.to_netcdf("regression_predictions.nc")
finally:
    pm.set_data({"x": train_x}, model=model, coords={"obs_id": train_ids})

print("Prediction identities:", predictions["predictions"].coords["obs_id"].values)
print("Latent mean mu versus noisy response: pointwise 89% ETIs")
print(az.summary(
    predictions, group="predictions", var_names=["mu", "response"],
    kind="stats", ci_prob=0.89, ci_kind="eti", round_to="none",
))
print(
    "Training data and identities restored:",
    np.array_equal(model["x"].get_value(), train_x)
    and list(model.coords["obs_id"]) == train_ids,
)
```

Read R-hat, bulk/tail ESS and MCSE for `alpha` and `beta` before interpreting
their uncertainty, and investigate every divergence. MCSE describes simulation
precision; posterior SD and the 89% ETI describe parameter uncertainty.
The ECDF PPC checks marginal distributional shape, not residual structure or
held-out prediction; add discrepancies matched to the scientific question.

The future `mu` draws recompute `alpha + beta*x_new` using the fitted coefficients.
Future `response` draws add observation noise, so their intervals include both
sources of uncertainty. The `try/finally` restores training predictors and IDs;
future means stay in a separate result rather than replacing training-sized
posterior deterministics. These intervals are pointwise, not simultaneous bands.

## Separate four questions

1. **Did we implement the intended density?** Compare log densities, gradients,
   identities and small-case analytic results. A sampler cannot fix a wrong graph.
2. **Did inference explore it accurately?** Inspect every divergence, rank R-hat,
   bulk/tail ESS, MCSE, tree depth/energy where applicable, and trace/rank plots.
   R-hat below 1.01 and ESS above 400 are useful screening targets, not guarantees.
   Tail ESS must correspond to the reported interval probabilities; set `prob`
   explicitly where the array-based API requires it. Precision must suit the
   decision. Do not excuse divergences by a small percentage.
3. **Does the model describe relevant data features?** Compare observed and
   replicated discrepancies, such as residual energy, extreme residuals,
   unexplained curvature, group imbalance or temporal dependence. A PPC tail
   probability is not a uniform frequentist p-value. A pointwise predictive
   envelope need not contain every observation.
4. **Does it predict the intended future use?** Use genuinely held-out information
   with appropriate grouped or temporal splits. In-sample fit, numerical agreement
   and healthy chains do not establish external predictive calibration.

Keep failures visible and diagnose their cause before revising the model,
parameterization, budget or decision criteria. More draws do not cure missing
identification, misspecification or approximation-family bias.

These distinctions apply throughout exploration; the required precision depends
on what is being claimed. A preliminary fit can motivate a revision without being
accepted as final inference. Keep its computational uncertainty visible.

## Experiment with simulated data

Probe new models with known-parameter simulations, changing sample size, design,
noise or identification to find where recovery and computation break down.
Repeated simulation-based calibration checks inference under the assumed joint
model, whereas PPCs criticize that model against observed data. Neither replaces
the other. See [simulated-data experiments and scoped SBC](model-testing.md#simulated-data-experiments)
for the procedure and its limits; scale the exercise to novelty, risk and cost.

## A small exact reference

For Gaussian regression `y | beta ~ Normal(X beta, sigma² I)` with genuinely known
sigma and prior `beta ~ Normal(m0, V0)`, define

\[
P=V_0^{-1}+X^TX/\sigma^2,\qquad
V=P^{-1},\qquad m=P^{-1}(V_0^{-1}m_0+X^Ty/\sigma^2).
\]

Compute with linear solves or factorizations rather than unnecessary matrix
inverses. For a new design row z, the latent mean has mean `z m` and variance
`z V z.T`; a new noisy observation adds `sigma²`. Compare MCMC estimates with
allowance for their MCSE. Agreement checks inference conditional on the observed
data; requiring every posterior interval to contain a generating parameter is
not a valid finite-sample correctness test.

## Communicate the result

State interval type and probability, computational limitations, prior sensitivity,
prediction conditioning and extrapolation. Distinguish pointwise intervals from
simultaneous bands. Compare predictive models using differences and uncertainty,
not stacking weights as an equivalence test. Causal interpretation additionally
requires a defensible structural model and identification assumptions.

When an action is requested, identify feasible alternatives, consequences and
constraints, and elicit the decision maker's loss or utility. Propagate posterior
and future-outcome uncertainty into expected loss or utility for each action;
choose according to that stated criterion, not whether an interval excludes zero
or a model tops an ELPD table. Examine sensitivity to plausible models and value
judgments. If costs, benefits or preferences are unavailable, provide interpretable
uncertainty summaries or draws for a decision-maker handoff rather than inventing
them. Not every analysis needs to prescribe an action.

Sources: [Bayesian Workflow](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf)
(2026 corrected edition, §§5.1, 7.3, 9.2–9.5, 11.4, 12.1 and Chapter 14),
[modern MCMC diagnostics](https://doi.org/10.1214/20-BA1221),
[PyMC sampling](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html),
[explicit log likelihood](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.stats.compute_log_likelihood.html).
