---
name: model-evaluation
description: >
  Model comparison and evaluation for Bayesian models. Use when comparing models,
  computing LOO-CV, interpreting ELPD, model stacking/averaging, or computing
  Bayes factors. Covers ArviZ 1.0 API exclusively. Triggers on: model comparison,
  LOO, ELPD, stacking, Bayes factor, cross-validation, model averaging, Pareto k,
  predictive accuracy, information criterion.
---

# Model Evaluation and Comparison (ArviZ 1.0)

CRITICAL: ArviZ 1.0 replaces InferenceData with xarray.DataTree. `az.waic` is removed entirely — use PSIS-LOO-CV exclusively. Default credible interval is 0.89 ETI (not 0.94 HDI), controlled via `ci_prob=` and `ci_kind=` (replaces the old `hdi_prob=`).

For model building context, prior selection, and convergence diagnostics, see the [pymc-modeling skill](../pymc-modeling/SKILL.md).

## LOO-CV with ArviZ 1.0

Leave-one-out cross-validation via Pareto-smoothed importance sampling (PSIS).

```python
import arviz as az
import arviz_stats  # registers the .azstats accessor

# dt is a DataTree from pm.sample()
loo_result = az.loo(dt)
print(loo_result)
# Returns: ELPDData with elpd_loo, se, p_loo, n_data_points, pareto_k

# Equivalent via the xarray accessor (DataArray / Dataset / DataTree all supported):
loo_result = dt.azstats.loo()
```

### Pareto k Diagnostics

Pareto k values indicate reliability of PSIS approximation for each observation:

| k value | Interpretation | Action |
|---|---|---|
| k < 0.5 | Good | LOO estimate reliable |
| 0.5 < k < 0.7 | Marginal | Results usable but less accurate |
| 0.7 < k < 1.0 | Bad | Estimate unreliable — use moment matching or k-fold |
| k > 1.0 | Very bad | PSIS fails entirely — must use k-fold CV |

```python
# Check Pareto k values
print(loo_result.pareto_k)

# Plot Pareto k diagnostics
az.plot_khat(loo_result)

# Count problematic observations
import numpy as np
k_values = loo_result.pareto_k.values
print(f"k > 0.7: {np.sum(k_values > 0.7)} observations")
```

### What to Do When k > 0.7

1. Try moment matching first (fast, automatic)
2. If still bad, use k-fold cross-validation
3. Check if problematic observations are outliers — consider robust likelihood
4. Re-examine the model — high k often signals model misspecification

## Moment Matching

Automatically refit problematic observations using moment matching:

```python
# Requires log_likelihood in the DataTree
loo_mm = az.loo_moment_match(dt)
```

This importance-weights the posterior for each problematic observation, improving the PSIS approximation without refitting the model. Much faster than k-fold.

## K-Fold Cross-Validation

When LOO is unreliable for many observations, use exact k-fold CV:

```python
# Perform 10-fold cross-validation
kfold_result = az.loo_kfold(dt, K=10)
print(kfold_result)
```

This refits the model K times, so it is K times slower than LOO. Use only when LOO diagnostics indicate problems.

## az.compare() — Full Workflow

Compare multiple models on predictive accuracy:

```python
# dt1, dt2, dt3 are DataTree objects from pm.sample()
comparison = az.compare(
    {"linear": dt1, "quadratic": dt2, "spline": dt3},
    scale="log",        # log scale (default) or deviance
)
print(comparison)
```

Note: `az.compare` in ArviZ 1.0 only supports LOO, so the `ic=` argument has been dropped.

### Interpreting the Comparison Table

| Column | Meaning |
|---|---|
| `rank` | Model rank (0 = best) |
| `elpd_loo` | Expected log pointwise predictive density |
| `p_loo` | Effective number of parameters |
| `d_loo` | Difference in ELPD from best model |
| `weight` | Stacking weight (sums to 1) |
| `se` | Standard error of ELPD |
| `dse` | Standard error of the ELPD difference |
| `warning` | True if any Pareto k > 0.7 |

### Decision Rules

- `d_loo` = 0: best model
- `|d_loo| < 4`: models are practically indistinguishable — prefer simpler one
- `|d_loo| > 4` and `|d_loo/dse| > 2`: meaningful difference in predictive accuracy
- `warning = True`: LOO unreliable for this model — investigate Pareto k values

```python
# Visualize comparison
az.plot_compare(comparison)

# Detailed forest plot of ELPD differences
az.plot_elpd({"linear": dt1, "quadratic": dt2, "spline": dt3})
```

See `references/model_comparison.md` for detailed usage.

## Model Averaging

### Stacking Weights (Default)

Stacking minimizes KL divergence from the true predictive distribution to the weighted mixture. This is the recommended default.

```python
comparison = az.compare({"m1": dt1, "m2": dt2, "m3": dt3})
# Stacking weights are in the "weight" column by default
print(comparison["weight"])
```

### Pseudo-BMA+ Weights

Alternative weighting based on Bayesian bootstrap of ELPD:

```python
comparison = az.compare(
    {"m1": dt1, "m2": dt2, "m3": dt3},
    method="BB-pseudo-BMA",
)
```

### When to Use Which

| Method | Use When |
|---|---|
| Stacking | Default. Best for prediction when true model is not in the set |
| Pseudo-BMA+ | Want Bayesian uncertainty over weights |
| Equal weights | Models represent different scientific hypotheses to average over |

### Generating Averaged Predictions

```python
weights = comparison["weight"].values
# Manually mix posterior predictive samples
# weighted by stacking weights
```

See `references/stacking.md` for detailed averaging workflows.

## Bayes Factors via Bridge Sampling

Bayes factors compare marginal likelihoods. Conceptually different from LOO (predictive accuracy vs. evidence).

```python
# Bayes factors are difficult to compute reliably
# Bridge sampling is the most reliable method but requires specialized setup
# For most applied work, LOO-CV is preferred

# Approximate Bayes factor from LOO (rough):
# BF ~ exp(elpd_loo_m1 - elpd_loo_m2)
# This is a very rough approximation — use with caution
```

### Limitations of Bayes Factors

- Highly sensitive to prior specification (unlike LOO)
- Numerically unstable for complex models
- Penalize model complexity differently than LOO
- Not recommended for routine model comparison — prefer LOO

## LOO-PIT Calibration

LOO probability integral transform checks if the model is calibrated:

```python
az.plot_loo_pit(dt, y="observed_data_name")
```

### Interpretation

- **Uniform histogram**: model is well-calibrated
- **U-shaped**: underdispersed predictions (too narrow)
- **Inverted U**: overdispersed predictions (too wide)
- **Skewed**: systematic bias in predictions

This is a powerful diagnostic that LOO uniquely provides — it checks calibration without held-out data.

## New ArviZ 1.0 Functions

### loo_expectations()

Compute LOO-weighted posterior expectations (mean, variance, quantile) for each observation. Requires both `posterior_predictive` and `log_likelihood` groups on the DataTree:

```python
# LOO-weighted posterior predictive mean for each observation
loo_mean = az.loo_expectations(dt, kind="mean")
loo_var = az.loo_expectations(dt, kind="var")
loo_q = az.loo_expectations(dt, kind="quantile", probs=[0.055, 0.945])
```

### loo_metrics()

Compute common LOO-based predictive metrics (RMSE, MAE, etc.) from `posterior_predictive` and `log_likelihood`:

```python
metrics = az.loo_metrics(dt, kind="rmse")
```

### .azstats xarray accessor

`import arviz_stats` registers an `.azstats` accessor on `DataArray`, `Dataset`, and `DataTree`. This gives a fluent xarray-native interface alongside the top-level `az.*` functions (which remain available after `import arviz as az`):

```python
import arviz_stats  # registers accessor; required even if you already imported arviz

dt.azstats.loo()                       # same as az.loo(dt)
dt["posterior"].azstats.rhat()         # on a Dataset
dt["posterior"].azstats.ess()
dt["posterior"].azstats.summary()
dt["posterior"].azstats.hdi()
dt["posterior"].azstats.eti()
```

### loo_r2()

Bayesian R-squared via LOO:

```python
r2 = az.loo_r2(dt)
print(f"LOO-R2: {r2.mean():.3f} [{r2.quantile(0.055):.3f}, {r2.quantile(0.945):.3f}]")
```

### loo_score()

Compute LOO-based scoring rules (CRPS, log score):

```python
score = az.loo_score(dt, score_func="crps")
```

### loo_subsample()

LOO with subsampling for large datasets:

```python
# When n > 10000, subsample for speed
loo_sub = az.loo_subsample(dt, observations=1000)
```

### reloo()

Exact refit LOO for observations with high Pareto k:

```python
# Refits the model for problematic observations
loo_exact = az.reloo(dt, loo_result, model=model)
```

## Standard Evaluation Workflow

```python
import arviz as az

# 1. Compute LOO
loo = az.loo(dt)
print(loo)

# 2. Check Pareto k
az.plot_khat(loo)

# 3. If k > 0.7, try moment matching
if (loo.pareto_k > 0.7).any():
    loo = az.loo_moment_match(dt)

# 4. LOO-PIT calibration
az.plot_loo_pit(dt, y="y")

# 5. Compare models
comparison = az.compare({"model_a": dt_a, "model_b": dt_b})
az.plot_compare(comparison)
print(comparison)

# 6. Predictive R2
r2 = az.loo_r2(dt)
```
