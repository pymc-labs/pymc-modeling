---
name: prior-check
description: Generate and analyze prior predictive checks for the current model
---

Generate and analyze prior predictive checks for a PyMC model.

1. Find the PyMC model definition. Look in the current file or ask the user to specify which file contains the model:

```
Search for files containing "pm.Model" or "pymc.Model" in the working directory.
```

2. Read the model file and identify the model context block (`with pm.Model(...) as model:`).

3. Generate code to run prior predictive sampling:

```python
import pymc as pm
import arviz as az
import numpy as np

# [Insert the model definition code here]

with model:
    prior_pred = pm.sample_prior_predictive(draws=500, random_seed=42)
```

4. Create prior predictive plots:

```python
# Prior predictive check plot
az.plot_ppc(prior_pred, group="prior", kind="cumulative")

# Plot prior distributions for key parameters
az.plot_dist(prior_pred["prior"]["PARAM_NAME"].values.flatten())
```

5. Analyze the prior predictions:

```python
# Get the observed variable name from the model
obs_var = [v.name for v in model.observed_RVs][0]

prior_y = prior_pred["prior_predictive"][obs_var].values
print(f"Prior predictive range: [{np.min(prior_y):.4f}, {np.max(prior_y):.4f}]")
print(f"Prior predictive mean: {np.mean(prior_y):.4f}")
print(f"Prior predictive std: {np.std(prior_y):.4f}")
print(f"Prior predictive median: {np.median(prior_y):.4f}")

# Check for extreme values
q01, q99 = np.quantile(prior_y, [0.01, 0.99])
print(f"Prior predictive 1%-99% range: [{q01:.4f}, {q99:.4f}]")
```

6. Report findings:
   - Are prior predictions in a plausible range for the observed data?
   - Are there extreme or impossible values (negative counts, probabilities > 1, etc.)?
   - Do the prior predictions span the range of observed data?
   - Suggest specific prior adjustments if issues are found (tighten wide priors, change distribution family, adjust hyperparameters).
