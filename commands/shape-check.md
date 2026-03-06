---
name: shape-check
description: Validate model shapes and dimensions before sampling
---

Validate PyMC model shapes and dimensions before committing to a full sampling run.

1. Find the PyMC model definition. Look in the current file or ask the user to specify:

```
Search for files containing "pm.Model" or "pymc.Model" in the working directory.
```

2. Read the model file and identify the model context block.

3. Run `model.debug()` to check for shape and value issues:

```python
import pymc as pm
import numpy as np

# [Insert the model definition code here]

# Debug check: reports shape mismatches, invalid parameter values, etc.
model.debug()
```

4. Run a fast prior predictive check with minimal draws to catch broadcasting errors:

```python
with model:
    # Single draw is enough to catch shape/broadcasting errors
    try:
        prior_test = pm.sample_prior_predictive(samples=1, random_seed=42)
        print("Prior predictive sampling: OK")
    except Exception as e:
        print(f"Shape/broadcasting error detected: {e}")
```

5. Check coords and dims consistency:

```python
# Verify coords match data dimensions
print("Model coords:")
for dim_name, coord_vals in model.coords.items():
    print(f"  {dim_name}: {len(coord_vals)} values")

# Check each variable's dims
print("\nVariable dimensions:")
for rv in model.free_RVs + model.observed_RVs:
    dims = model.named_vars_to_dims.get(rv.name, "none")
    print(f"  {rv.name}: dims={dims}")
```

6. Check for common shape issues:

- **Mismatched coord lengths**: Do coordinate arrays match the corresponding data dimensions?
- **Index vector bounds**: Are index vectors within the valid range for their target parameter?
- **Design matrix shape**: Is X shape (N, K) and not (K, N)?
- **Broadcasting traps**: Will (J,) and (N,) shapes broadcast to (N, J) unexpectedly?
- **MutableData compatibility**: If using `pm.set_data()`, are shapes compatible?

7. Report results:
   - List any shape mismatches with the specific variable, expected shape, and actual shape.
   - For each issue, provide the fix (corrected dims, proper indexing, transpose, etc.).
   - If no issues found, confirm that shapes and dimensions are consistent.
