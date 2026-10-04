# Predictive criticism and out-of-sample use

## Declare the target

Separate posterior uncertainty about a latent conditional mean from a new
observation, which also includes observation noise. State the observation unit,
future design and information available before its outcome. An offset must be
known then, not a fitted residual or target encoding. Keep training transformations
for future inputs. Existing-group and new-group predictions have different
conditioning; temporal predictions must respect the forecast origin.

Use pointwise interval probabilities explicitly, for example `(0.055, 0.945)` for
an 89% ETI. These are not simultaneous bands. In Gaussian regression with known
noise variance, new-observation variance adds that noise variance to latent-mean
variance. Distinguish both in plots and reported uncertainty.

## Build a resizable prediction graph

```python
# All arrays and prior/noise scales are supplied by the project's model.
with pm.Model(coords={"obs_id": train_ids}) as model:
    x = pm.Data("x", train_x, dims="obs_id")
    offset = pm.Data("offset", train_offset, dims="obs_id")
    alpha = pm.Normal("alpha", 0, intercept_scale)
    beta = pm.Normal("beta", 0, slope_scale)
    mu = pm.Deterministic("mu", alpha + beta*x + offset, dims="obs_id")
    pm.Normal("response", mu, known_noise_sd, observed=train_y,
              shape=x.shape, dims="obs_id")
```

Only fix noise when genuinely known. Validate ordered unique IDs, finite values,
lengths and complete feature availability before mutating the model. Equal lengths
are not identity checks; do not sort or relabel to conceal mismatches.

`set_data` can resize a dimension, not change tensor rank. Update every feature on
the shared axis, including offsets, exposures and group indices, with new coords.
Updates are not transactional: validate first and restore training inputs in
`finally`. Never fit against original outcomes paired with changed predictors.
There is no need to insert fake future responses into a data container.

## Preserve conditioning

In [PyMC 6.3.1 forward sampling](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/sampling/forward.py):

- `var_names` selects returned variables; it does not itself request resampling.
- `sample_vars` explicitly regenerates trace variables. RVs absent from the trace,
  including observed responses, are regenerated automatically.
- `freeze_vars` explicitly reuses trace values; it cannot overlap `sample_vars`.

Trace variables match by name with compatible shapes/coordinates. Changed data
make dependent deterministics volatile: recompute new-design `mu`, not a frozen
training-sized array. Freeze coefficients only when their posterior conditioning
remains appropriate; changed prior ancestors may alter the question.

```python
try:
    pm.set_data({"x": new_x, "offset": new_offset}, model=model,
                coords={"obs_id": new_ids})
    predictions = pm.sample_posterior_predictive(
        idata, model=model, predictions=True,
        var_names=["mu", "response"], freeze_vars=["alpha", "beta"],
        random_seed=42,
    )
    predictions.to_netcdf("predictions.nc")
finally:
    pm.set_data({"x": train_x, "offset": train_offset}, model=model,
                coords={"obs_id": train_ids})
```

`predictions=True` selects a storage group; it is not a data split, leakage check
or calibration procedure. Changing inputs does not update posterior parameter
knowledge. New outcomes require a deliberate refit or inferential update.
Check draw-wise deterministic identities and observation IDs, not only plausible
averages. Do not merge differently sized future means into training posterior data.

## Criticize the observation model

Choose discrepancies that could reveal plausible failures: standardized residual
energy for scale, maximum residual magnitude for tails, residual projection on a
quadratic predictor for omitted curvature, or group/time summaries for dependence.
If a statistic depends on parameters, compare observed and replicated versions
at the same posterior draw. Save discrepancy distributions and uncertainty in
estimated tail probabilities; account for autocorrelation. A constant indicator
provides no useful estimated MCSE, rather than proof of perfect precision.

Posterior predictive probabilities are not uniform frequentist p-values. Reusing
observations for fitting and checking makes PPCs in-sample criticism, not external
calibration. Neither every point lying inside a band nor agreement on a few
features establishes the unique mechanism. Check computational health separately;
poor mixing leaves scientific interpretation unresolved even with attractive PPCs.
Inspect diagnostic and predictive plots, not just their existence.

## Evaluate future use separately

Keep held-out outcomes unavailable to fitting, preprocessing and hyperparameter
selection. Report latent-mean error only when its target is available; evaluate
new-observation coverage and predictive scores against genuinely future outcomes.
Mark extrapolation. A binomial interval for coverage is only a conditional-design
approximation: common fitted-model error can correlate cases. Repeated-refit,
grouped and temporal calibration address different questions from one fixed batch.
Compute pointwise log likelihood explicitly before LOO; choose a deletion unit
consistent with dependence and the scientific prediction target.

## Lower-level prediction tools

| Tool | Responsibility |
|---|---|
| `compute_deterministics` | Evaluate with current data over posterior samples; use `extend_dataset` only for deliberate mutation. Do not use deprecated `merge_dataset`. |
| `vectorize_over_posterior` | Replace free RVs with posterior arrays using full sample dimensions. Use `allow_rvs_in_graph=False` for deterministic evaluation. |
| `draw` | Evaluate/generate a graph, not infer a posterior. Unsubstituted RVs draw from the generative graph. |
| `compile_forward_sampling_function` | Returns a callable and RVs regenerated; manage volatility, freeze sets and trace-time `constant_data` deliberately. |
| `predictions_to_inference_data` | Convert existing `(chain, draw, *shape)` arrays to a labelled DataTree; neither generates predictions nor repairs alignment. |

Select the intended compiler explicitly when low-level backend behavior matters;
`FAST_RUN` alone need not imply the C linker. Preserve warnings and precision
failures rather than silently switching implementations.

Sources: [posterior predictive API](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html),
[deterministics](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/sampling/deterministic.py),
[conversion](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/backends/arviz.py),
[Bayesian workflow](https://arxiv.org/abs/2011.01808).
