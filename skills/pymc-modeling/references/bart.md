# BART: fitting, prediction state and interpretation

BART models a function as a sum of trees, not an observation process. Specify the
likelihood, link, conditional independence and noise model separately. Flexible
means do not remove heteroscedasticity, measurement or extrapolation assumptions.

The API cautions here refer to pymc-bart 0.13.1/bartrs 0.4.0. Inspect the installed
release before assuming they persist. That BART release requires PyMC >=6.3,<7,
Python >=3.12 and bartrs >=0.4; extras 0.14's older core requirements conflict.
Use compatible project dependencies instead of overriding bounds.

## Outcome-dependent regularization and prior prediction

BART's Y argument is not another observed likelihood. In 0.13.1 it supplies
initialization and empirical prior/proposal scaling: initial mean Y.mean(), leaf
scale Y.std()/sqrt(m), except exactly binary {0,1} uses 3/sqrt(m). This is
outcome-dependent scaling, not an outcome-independent weak prior. Place it inside
training folds; never include future outcomes. Constant Y yields zero empirical
leaf scale and needs explicit investigation.

Depth probability is `alpha*(1+depth)**(-beta)`. Tree count, depth, split weights
and particle/batch settings change regularization/exploration. Assess sensitivity
with the same scientific target.

Before tree history exists, this version's `BARTRV.rng_fn` returns a constant
training-response mean. A prior-predictive call therefore does **not** simulate a
full generative function prior. A noise-only plausibility check is narrower; do not
label it a full BART prior check. After fitting, stored posterior trees are not
fresh prior draws either.

Binary classification can use sigmoid(BART) with Bernoulli and its binary scaling.
Counts need a positive link and a defensible latent-scale Y setup; blindly log(Y)
fails at zero. Audit response support and induced priors for each likelihood.

## Sampling the correct tree posterior

```python
from bartrs import PGBART
import pymc_bart as pmb

with pm.Model() as model:
    mu = pmb.BART("mu", X, y, m=tree_count, response="constant")
    sigma = pm.HalfNormal("sigma", sigma=noise_prior_scale)
    pm.Normal("response", mu=mu, sigma=sigma, observed=y)
    step = [PGBART(vars=[mu]), pm.NUTS(vars=[sigma])]
    idata = pm.sample(step=step, nuts_sampler="pymc", tune=tune,
                      draws=draws, chains=chains, random_seed=42)
idata.to_netcdf("posterior.nc")
```

This is an interface example, not a claim that a chosen release supplies independent
tree streams; check the version caution below before choosing chain-based precision.
PGBART comes from bartrs in this version and updates trees with particle Gibbs.
Generic NUTS/nutpie/NumPyro/BlackJAX over the training function vector does not
implement that posterior: the BART RV's symbolic logp is zero while its actual
prior/update lives in the tree sampler. NUTS energy/divergences concern only its
continuous block, not tree exploration.

bartrs compiles a Numba likelihood callback for Rust CPU execution. Configuring
another continuous-step linker does not change that callback. Preserve ctypes or
object-mode fallback warnings and check dtype at graph construction; float64 input
arrays do not guarantee every intermediate uses float64.

**Tree RNG caution:** in bartrs 0.4.0 PGBART does not forward its PyMC RNG argument
to native settings, whose seed defaults to zero. Different pm.sample seeds or
fresh model instances do not establish independent native tree streams. Do not
manufacture independence by reshaping or relabelling output. Inspect the current
[PGBART implementation](https://github.com/pymc-devs/bartrs/blob/58764868f0e1b0984434fa41ea98b948001dd7bb/python/bartrs/pgbart.py)
and supported seed controls before interpreting R-hat/ESS as independent-chain
precision. More chains alone cannot repair missing seed propagation.

## Prediction needs tree state and draw pairing

A posterior DataTree holds training mu, continuous parameters and statistics, not
the native tree histories needed for new X. A reconstructed model plus that file
is insufficient. In this version completed fits append `(baseline_forest,batches)`
to the BART Op's `all_trees`; history is completed only at the last expected retained
iteration. Interrupted traces can lack usable new-input tree state.

For version-compatible native persistence:

1. Save the raw DataTree immediately.
2. Save baseline/batches using supported TreeArrays state hooks, with package
   versions, tree/output counts and chronological draw identity. Do not serialize
   live managers, compiled callbacks or the whole live model.
3. Reconstruct `PosteriorSampler.from_history(batches,baseline_forest,m,n_outputs)`.
   Use explicit chronological indices in `sample_posterior(x_float64,draw_indices,None)`;
   one-output results have shape `(draw,1,row)`.
4. Pair each function prediction with its corresponding sigma/other likelihood
   parameter draw before adding new observation noise.
5. Check reloaded training predictions against saved mu, and new-input predictions
   against their pre-serialization values without reordering to hide mismatch.

Native state is version-specific and not a resume checkpoint for RNG/adaptation.
Pickle can execute code: load only state you created and trust, never downloaded
examples or untrusted artifacts.

A live generic BART predictive call can resample trees randomly from pooled history,
not in the current trace's chain/draw order. That can lose joint dependence with
sigma. Audit the intended conditioning rather than assume `set_data` preserves
pairing. Training mu is enough for an in-sample PPC, not new-X function predictions.

## Diagnose and interpret

Inspect function-space summaries and sigma, not arbitrary tree labels alone.
Use trace/rank plots, bulk/tail precision, continuous-step diagnostics and supported
tree-stream independence checks. A few averaged function traces can hide poor
mixing at particular inputs. Save predictions, validate row/draw identities, separate
mean/observation intervals and mark extrapolation. Residual dispersion/tail PPCs are
in-sample criticism; held-out RMSE/coverage or predictive scores require leakage-safe
splits and do not establish repeated-refit calibration from one batch.

| Utility | Meaning and limits in 0.13.1 |
|---|---|
| get_variable_inclusion | Normalized split counts, not probabilities of nonzero effects. Undefined if total splits zero. DataFrame labels follow sorted values; custom labels may not be reordered. |
| compute_variable_importance | VI/backward/backward_VI compare reduced and full function predictions using squared Pearson correlation, not held-out outcome R² or predictive likelihood. |
| plot_variable_inclusion / importance / scatter_submodels | Views of those quantities. Library r2_hdi denotes HDIs, not response ETIs. |
| plot_pdp / plot_ice | Altered-input function summaries, not interventions. Correlated predictors can create unsupported combinations; smoothing changes display, not leaves. |
| vi_to_kulprit | Candidate feature paths, not projected/refitted models. Inspect whether the full feature set is included. |
| plot_convergence | Deprecated warning-only route in this release; use actual ArviZ diagnostics. |

Split counts depend on predictor type, split opportunities, correlation and
regularization. Redundant predictors can substitute; low inclusion is not no effect.
Reduced/full functions can be sampled independently, so all-variable agreement
need not be one. Feature selection requires separate predictive assessment.
`response="linear"`/`"mix"` are experimental; categorical split rules need their
own encoding/source checks.

Sources: [BART API](https://www.pymc.io/projects/bart/en/latest/api_reference.html),
[BART distribution](https://github.com/pymc-devs/pymc-bart/blob/39792e8fd410b333d8e2fb1791f6359a84e3f7f4/pymc_bart/bart.py),
[utilities](https://github.com/pymc-devs/pymc-bart/blob/39792e8fd410b333d8e2fb1791f6359a84e3f7f4/pymc_bart/utils.py),
[native sampler](https://github.com/pymc-devs/bartrs/blob/58764868f0e1b0984434fa41ea98b948001dd7bb/src/lib.rs),
[tree persistence](https://github.com/pymc-devs/bartrs/blob/58764868f0e1b0984434fa41ea98b948001dd7bb/src/tree.rs),
[BART formulation](https://doi.org/10.1214/09-AOAS285).
