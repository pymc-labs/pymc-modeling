# Diagnostics and predictive checks

Examples use the modern ArviZ package family and DataTree APIs. Interval labels
and plotting details below describe ArviZ 1.3; check version-matched documentation
rather than mixing legacy InferenceData or matplotlib-Axes recipes.

## Complete diagnostic example

This script fits its own illustrative Gaussian regression: 80 synthetic rows,
independent `Normal(0,1)` intercept/slope priors and known observation-noise SD
0.5. The fixed seed and four-chain, 1,000-tune/1,000-draw allocation make the
example reproducible; choose real sampling budgets from exploration and
estimand-specific precision. Native PyMC sampling needs no optional sampler.
The figures use Matplotlib. NetCDF output needs a compatible backend, such as
`h5netcdf` with `h5py`.

```python
import arviz as az
import numpy as np
import pymc as pm

seed = 20260911
rng = np.random.default_rng(seed)
x = np.linspace(-1.0, 1.0, 80)
noise_sd = 0.5
y = 0.5 + 1.2 * x + rng.normal(0.0, noise_sd, 80)
coords = {"obs": np.arange(x.size)}

with pm.Model(coords=coords) as model:
    x_data = pm.Data("x", x, dims="obs")
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)
    mu = pm.Deterministic("mu", alpha + beta * x_data, dims="obs")
    pm.Normal("response", mu=mu, sigma=noise_sd, observed=y,
              shape=x_data.shape, dims="obs")
    factors = model.point_logps()
    print("Initial log-density factors:", factors)
    if not all(np.isfinite(value) for value in factors.values()):
        raise ValueError("Inspect model support and initial values before sampling.")
    idata = pm.sample(
        chains=4, cores=1, tune=1000, draws=1000,
        nuts_sampler="pymc", random_seed=seed,
    )

idata.to_netcdf("diagnostics_raw.nc")  # Save before any postprocessing.
posterior = idata["posterior"].to_dataset()
summary = az.summary(
    idata, var_names=["alpha", "beta"], ci_prob=0.89, ci_kind="eti",
    round_to="none",
)
print(summary[["mean", "sd", "eti89_lb", "eti89_ub", "r_hat",
               "ess_bulk", "ess_tail", "mcse_mean", "mcse_sd"]])
divergences_by_chain = idata["sample_stats"]["diverging"].sum("draw")
print("Divergences by chain:", divergences_by_chain)

# The estimand is the mean-response change from x=-1 to x=1, in response units.
contrast = (2.0 * posterior["beta"]).transpose("chain", "draw").values
contrast_eti = np.quantile(contrast, [0.055, 0.945])
contrast_mean_mcse = az.mcse(contrast, method="mean")
contrast_endpoint_mcse = [
    az.mcse(contrast, method="quantile", prob=q) for q in (0.055, 0.945)
]
print(f"Mean-response contrast: mean={contrast.mean():.6f}, "
      f"89% ETI={contrast_eti}, mean MCSE={contrast_mean_mcse:.6f}")
print("Contrast ETI endpoint MCSEs:", contrast_endpoint_mcse)
print("Contrast ESS at ETI tails:",
      az.ess(contrast, method="tail", prob=(0.055, 0.945)))

# Deliberately damaged diagnostic fixture; preserve the genuine fit.
original_alpha = posterior["alpha"].values.copy()
shifted = idata.copy(deep=True)
first_chain = shifted["posterior"]["chain"].values[0]
shifted["posterior"]["alpha"] = (
    shifted["posterior"]["alpha"]
    + 3.0 * (shifted["posterior"]["chain"] == first_chain)
)
shifted_values = shifted["posterior"]["alpha"].transpose("chain", "draw").values
original_rhat = az.rhat(
    posterior["alpha"].transpose("chain", "draw").values, method="rank"
)
shifted_rhat = az.rhat(shifted_values, method="rank")
print(f"Intercept rank R-hat: genuine={original_rhat:.6f}, "
      f"shifted-chain fixture={shifted_rhat:.6f}")
print("Original intercept unchanged:",
      np.array_equal(idata["posterior"]["alpha"].values, original_alpha))

# Enrich only the genuine fit; preserve observation identities in both groups.
pm.compute_log_likelihood(idata, model=model)
pp = pm.sample_posterior_predictive(
    idata, model=model, var_names=["response"], random_seed=seed,
)
idata["posterior_predictive"] = pp["posterior_predictive"]
idata.to_netcdf("diagnostics_enriched.nc")

trace = az.plot_trace_dist(
    idata, var_names=["alpha", "beta"], backend="matplotlib",
)
trace.savefig("diagnostics_trace.png", bbox_inches="tight")
rank = az.plot_rank(idata, var_names=["alpha", "beta"], backend="matplotlib")
rank.savefig("diagnostics_rank.png", bbox_inches="tight")
ppc = az.plot_ppc_dist(
    idata, var_names=["response"], kind="ecdf", backend="matplotlib",
)
ppc.savefig("diagnostics_ppc.png", bbox_inches="tight")
replicated = idata["posterior_predictive"]["response"]
print("Observed response mean and range:", y.mean(), np.ptp(y))
print("Replicated-dataset mean 89% ETI:",
      replicated.mean("obs").quantile([0.055, 0.945], dim=("chain", "draw")).values)
print("Replicated-dataset range 89% ETI:",
      (replicated.max("obs") - replicated.min("obs")).quantile(
          [0.055, 0.945], dim=("chain", "draw")).values)

# Prediction target: a new exchangeable row, not a new group or future time.
loo = az.loo(idata, var_name="response", pointwise=True)
print(f"LOO ELPD={loo.elpd:.3f}, SE={loo.se:.3f}, p={loo.p:.3f}")
print("Pointwise observation identities:", loo.elpd_i["obs"].values)
unreliable = (loo.pareto_k > loo.good_k) | ~np.isfinite(loo.pareto_k)
print(f"Pareto-k threshold={loo.good_k:.3f}, "
      f"max k={loo.pareto_k.max().item():.3f}, "
      f"flagged={unreliable.sum().item()}, warning={loo.warning}")
khat = az.plot_khat(loo, backend="matplotlib")
khat.savefig("diagnostics_khat.png", bbox_inches="tight")
print("Inspect diagnostics_trace.png, diagnostics_rank.png, "
      "diagnostics_ppc.png and diagnostics_khat.png.")
```

Inspect the saved figures, not just their filenames. For **computation**, use
the genuine fit's divergence counts, R-hat, ESS and MCSE together with trace/rank
behavior. Small MCSE is simulation precision; the 89% ETI describes posterior
uncertainty. The deliberately shifted copy should have a worse intercept R-hat
than the genuine fit and exceed 1.01. Its other variables and stored sampler
statistics are deliberately inconsistent with the shifted intercept: it is
only a diagnostic failure fixture, never a revised posterior or predictive fit.

For **in-sample adequacy**, compare the response ECDF, dataset mean and range;
then investigate conditional residual patterns or other scientific discrepancies
the marginal plot could hide. Agreement supports only the features inspected,
not held-out validation. For **predictive evaluation**, report pointwise LOO's
80 observation identities, threshold, warnings and influential rows before
using its ELPD for the declared exchangeable-row target. ELPD is a predictive
log score, not a Bayes factor. Preserve failed diagnostics and revisit their
causes rather than rerunning seeds to obtain a passing example.

## Preserve dimensions and identify the target

Save inference before postprocessing. Keep original chains and draw order; do
not thin or discard unfavorable chains to improve a diagnostic. Know whether
draws represent MCMC, independent simulation or an approximation. An observation-
free model samples a prior target even if its output group is called `posterior`.

```python
import arviz as az
import arviz_stats  # registers xarray .azstats accessors

idata = az.from_netcdf("posterior.nc")
posterior = idata["posterior"].to_dataset()
summary = az.summary(idata, var_names=["mu"], ci_prob=0.89,
                     ci_kind="eti", round_to="none")
print(summary[["mean", "sd", "eti89_lb", "eti89_ub",
               "ess_bulk", "ess_tail", "r_hat", "mcse_mean", "mcse_sd"]])
```

`eti89_lb`/`eti89_ub` are the 5.5th/94.5th posterior percentiles, not MCSE bounds.
Request `ci_kind="hdi"` explicitly for a highest-density interval; ETI and HDI can
differ substantially for skewed or multimodal targets. Keep full precision for
decisions. `fmt="xarray"` returns a Dataset rather than the default DataFrame.

Summary tail ESS need not use the interval's endpoint probabilities. For a
scalar variable, calculate the relevant endpoints explicitly:

```python
values = posterior["mu"].transpose("chain", "draw").values
rank_rhat = az.rhat(values, method="rank")
bulk = az.ess(values, method="bulk")
tail = az.ess(values, method="tail", prob=(0.055, 0.945))
mean_mcse = az.mcse(values, method="mean")
endpoint_mcse = [az.mcse(values, method="quantile", prob=q)
                 for q in (0.055, 0.945)]
```

Never collapse chains before R-hat/ESS. Preserve labels with xarray-aware
functions for vector parameters and monitor relevant latent coordinates, not
just a convenient population mean. Accessors such as `data.azstats.rhat(...)`
are an alternative to `az.rhat(data)`; do not instantiate accessor classes.
PyMC's `pm.stats`/`pm.plots` expose ArviZ facades; prefer one consistent namespace
rather than deprecated PyMC root aliases.

## Interpret health and precision separately

| Diagnostic | Investigate | Interpretation/action |
|---|---|---|
| HMC divergences by chain/location | Integration, gradients, scaling, constraints and difficult geometry | Investigate even small nonzero counts; locate problems before choosing a remedy |
| Rank-normalized split R-hat | Between/split-chain location and scale disagreement | Near-one values still require evidence of relevant-mode exploration |
| Bulk ESS | Effective information for central behavior | Use effective information rather than stored draw count |
| Tail ESS | Exploration at the stated tail probabilities | Check the quantiles needed for the claim separately from central behavior |
| Mean/SD/quantile MCSE | Precision for the actual estimand in meaningful units | Report separately from posterior uncertainty |
| Trace/rank graphics | Drifting, sticking, scale separation and rank imbalance | Locate affected chains/coordinates and compare with numerical diagnostics |
| Energy/BFMI | Chain-specific HMC energy exploration | Interpret for the actual sampler and model |

R-hat below 1.01 and bulk/tail ESS above 400 are useful screens, not sufficiency
proofs. Choose mean, quantile or event-probability MCSE requirements from the
scientific decision. Missing/nonfinite diagnostics or degenerate monitored
variables are unresolved. HMC energy/divergence statistics can be inapplicable
to another sampler: do not invent zeros or infer NUTS semantics for Metropolis.

Apply accuracy requirements to the claim being made. An explicitly exploratory
fit may have short chains and unresolved diagnostics yet reveal what to debug or
which model assumption to investigate next. Retain its unresolved checks and
refit to the required accuracy before reporting reliable final inference.

Use `az.diagnose` for an overview alongside estimand-specific checks.
More draws reduce MCSE only after trustworthy exploration. Funnels, missing
modes and non-identifiability need investigation first. Non-centering often
helps weakly informed scales; centering or partial parameterization may suit
strong data. Preserve the original result when assessing a revised fit.

## Inspect actual chain and geometry plots

```python
trace = az.plot_trace_dist(idata, var_names=["mu"], backend="matplotlib")
trace.savefig("trace.png", bbox_inches="tight")
rank = az.plot_rank(idata, var_names=["mu"], backend="matplotlib")
rank.savefig("rank.png", bbox_inches="tight")
```

Modern plotting returns `PlotCollection`/`PlotMatrix`, not an array of matplotlib
Axes. `plot_trace_dist` combines marginal distributions and traces; `plot_trace`
is trace-only. ArviZ 1.3 `plot_rank` uses rank ECDF diagnostics. Its `mtc_c` method
accounts for autocorrelation without thinning; rank highlighting is not an
automated convergence pass. `plot_rank_dist` also includes marginals.

For a hierarchical model with `v` and vector `x`, mark divergences explicitly:

```python
pair = az.plot_pair(idata, var_names=["v", "x"], coords={"component": [0]},
                    visuals={"divergence": True}, backend="matplotlib")
pair.savefig("pair.png", bbox_inches="tight")
```

Use `visuals={"divergence": True}`, not legacy `divergences=True`.
A neck-shaped divergence cluster is consistent with difficult geometry, not
proof of one remedy. `plot_autocorr`, `plot_ess`, `plot_ess_evolution` and
`plot_mcse` address dependence/precision; `plot_energy` needs recorded energy.

Distribution, forest, ridge, pair and parallel plots describe distributions,
not convergence. KDE can conceal modes. `mean`, `median`, `mode`, `std`, `var`,
`mad` (median absolute deviation), `iqr`, `eti` and `hdi` summarize different
features. `ecdf`, `histogram`, `kde`, `kde2d` and `qds` have reduction/smoothing
choices that must match the target. Plot composition uses `visuals`,
`aes_by_visuals`, `add_lines`, `add_bands` and `combine_plots` in the modern API.

## Posterior predictive criticism

Given a fitted PyMC model with observed response `y`:

```python
import pymc as pm

pp = pm.sample_posterior_predictive(idata, model=model, var_names=["y"])
idata["posterior_predictive"] = pp["posterior_predictive"]
az.plot_ppc_dist(idata, var_names=["y"], kind="ecdf")
```

Check task-relevant discrepancies: location, spread, tails, zeros, residual
patterns, dependence and group variation. `plot_ppc_dist`, `plot_ppc_dist_pit`,
`plot_ppc_pit`, `plot_ppc_interval` and `plot_ppc_tstat` provide distribution,
PIT, interval and statistic views. These reuse fitted outcomes: PPC tail
probabilities are not uniformly calibrated p-values, and good in-sample fit is
not external validation. A uniform-looking marginal PIT does not establish
conditional calibration. Use LOO or independent validation for that target.

`plot_dgof`/`plot_dgof_dist` check the fitted one-dimensional **density
representation**, not the observation model. Do not substitute them for PPCs.
For independent future-outcome PIT values, keep an honestly named Dataset and
use `plot_ecdf_pit` with the actual observation sample dimension (and `group=None`
for an ungrouped Dataset); do not label those values as simulation-based
calibration ranks. Finite predictive simulation and shared parameter uncertainty
limit uniformity interpretations.

SBC instead repeatedly draws parameters from a proper joint prior, simulates data,
refits the same model and checks ranks of generating test quantities among posterior
draws. It assesses computation under that model, not real-data adequacy. Its ranks
need tie handling, attention to MCMC dependence and finite-replication uncertainty;
do not turn a uniform-looking histogram into a universal correctness claim.

### Regression and counts

`plot_lm` should show predictors and observations with the intended posterior
mean/interval. A latent mean band omits observation noise and is not a
new-observation band. `bayesian_r2`, `residual_r2` and `metrics` assess different
summaries, not convergence or external accuracy. Match the documented variance
convention (sample variance for the ArviZ R² calculations); a returned
observation-level RMSE SE is not posterior MCSE. Known noise is a deterministic
constant, not an invented sampled parameter.

Use `plot_ppc_rootogram` for actual count predictions, not rounded continuous
draws. Respect exposure: a Gamma(shape `a`, rate `b`) prior and
`count_i ~ Poisson(rate*exposure_i)` imply posterior
`Gamma(a+sum(count), b+sum(exposure))`, a useful analytic check.
`plot_ppc_pava`/`plot_ppc_pava_residuals` address binary, categorical or ordinal
targets, not arbitrary continuous residuals. Transform a derived binary target
(e.g. count positive) identically for observed and every predictive draw,
preserving exposure/observation identities.

### Survival and censoring

Distinguish event time, recorded censored time and event status. In ArviZ's
survival workflow, `kaplan_meier` uses status 1=event, 0=censored stored under
the response name in `constant_data`. `generate_survival_curves` and
`plot_ppc_censored` need the corresponding prediction target; uncensored event-
time draws support survival curves, not an atom at the administrative cutoff.

For independent right censoring, Exponential event times and Gamma(a,b) rate
prior give `Gamma(a+events, b+sum(recorded_time))`. Censored observations contribute
survival probability, not a death at the cap. Informative censoring needs its
own model. In version-specific utility behavior, `extrapolation_factor=None`
avoids filtering/renormalizing long predicted times. Check tie handling: a
unique-time shortcut is not a general tied-event survival estimator. Compare
curves over an interpretable horizon; extrapolated tails can be prior-sensitive.

## Nested chains and model structure

Nested R-hat requires a genuinely nested design: independent superchains with
shared initial states for the chains inside each superchain. Do not relabel
ordinary chains after sampling. The sampler must support the required per-chain
initialization; check the backend rather than assuming it does. Preserve the
superchain mapping and use `rhat_nested` for that design alongside ordinary
precision and exploration checks. It is not an automatic many-short-chains fix.

Inspect model structure when missing observations, wrong plates or unexpected
dependencies could explain suspicious output. `str(model)`/`model.table()`
describe roles; `pm.model_to_mermaid`, `pm.model_to_networkx` and
`pm.model_to_graphviz` expose dependency graphs. NetworkX/Graphviz need their
optional dependencies, and DOT rendering also needs the Graphviz executable.
Graph structure does not calculate convergence. Sampler `stats_dtypes_shapes`
declares possible statistics; actual availability is backend/run-specific.
Do not mutate metadata to make absent diagnostics appear present.

## Primary sources

- [ArviZ summary](https://arviz-stats.readthedocs.io/en/latest/api/generated/arviz_stats.summary.html),
  [R-hat](https://arviz-stats.readthedocs.io/en/latest/api/generated/arviz_stats.rhat.html),
  [ESS](https://arviz-stats.readthedocs.io/en/latest/api/generated/arviz_stats.ess.html)
  and [MCSE](https://arviz-stats.readthedocs.io/en/latest/api/generated/arviz_stats.mcse.html).
- [Rank-normalized R-hat and ESS methodology](https://arxiv.org/abs/1903.08008).
- [ArviZ Stats API](https://python.arviz.org/projects/stats/en/latest/api/index.html)
  and [ArviZ Plots API](https://arviz-plots.readthedocs.io/en/latest/api/index.html)
  for predictive, survival and nested-design signatures.
- [PyMC model graph source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/model_graph.py)
  and [sampler statistics protocol](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/step_methods/compound.py).
