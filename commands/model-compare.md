---
name: model-compare
description: Compare multiple Bayesian models using LOO-CV
---

Compare two or more Bayesian models using PSIS-LOO-CV (Pareto-Smoothed Importance Sampling Leave-One-Out Cross-Validation).

1. Find model result files. Look for `.nc` (NetCDF) files in the current directory, or ask the user to specify 2 or more result files:

```
Glob for **/*.nc files in the working directory.
```

2. Load each result file using ArviZ 1.0 DataTree API:

```python
import arviz as az
import pymc as pm

# Load results
model_results = {}
for name, path in MODEL_FILES.items():
    dt = az.from_netcdf(path)
    model_results[name] = dt
    print(f"Loaded {name}: groups = {list(dt.children.keys())}")
```

3. Ensure log-likelihood is available for each model. If missing, compute it:

```python
for name, dt in model_results.items():
    if "log_likelihood" not in dt.children:
        print(f"WARNING: {name} is missing log_likelihood group.")
        print("Rerun with pm.compute_log_likelihood(dt, model=model) after pm.sample()")
```

4. Run model comparison using LOO-CV:

```python
# ArviZ 1.0: LOO is the only information criterion (WAIC is removed)
comparison = az.compare(model_results)
print(comparison)
```

5. Generate comparison plot:

```python
az.plot_compare(comparison)
```

6. Interpret results:

```python
print("\n--- Model Comparison Interpretation ---")
print(f"\nBest model: {comparison.index[0]}")
print(f"ELPD difference from best:")
for model_name in comparison.index[1:]:
    row = comparison.loc[model_name]
    elpd_diff = row["elpd_loo"] - comparison.iloc[0]["elpd_loo"]
    se_diff = row["dse"]
    print(f"  {model_name}: {elpd_diff:.1f} +/- {se_diff:.1f}")
    if abs(elpd_diff) < 2 * se_diff:
        print(f"    -> Not meaningfully different from best model")
    else:
        print(f"    -> Meaningfully worse than best model")

# Stacking weights
print(f"\nStacking weights (for model averaging):")
for model_name in comparison.index:
    w = comparison.loc[model_name, "weight"]
    print(f"  {model_name}: {w:.3f}")

# Pareto k warnings
for name, dt in model_results.items():
    loo_result = az.loo(dt)
    n_bad = (loo_result.pareto_k > 0.7).sum().item()
    if n_bad > 0:
        print(f"\nWARNING: {name} has {n_bad} observations with Pareto k > 0.7")
        print("  LOO estimates may be unreliable for this model")
```

7. Provide a summary:
   - Rank models by ELPD (expected log pointwise predictive density)
   - Note whether differences are meaningful (ELPD difference > 2*SE)
   - Report stacking weights for model averaging
   - Flag any Pareto k warnings that make LOO estimates unreliable
   - Note: WAIC is not available in ArviZ 1.0; use PSIS-LOO-CV exclusively
