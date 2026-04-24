---
name: pymc-diagnose
description: Run full MCMC diagnostics on a DataTree file
---

Run a comprehensive MCMC diagnostic analysis on sampling results.

1. First, find a results file. Look for `.nc` (NetCDF) files in the current directory or ask the user to specify one:

```
Glob for **/*.nc or **/results*.nc files in the working directory.
```

2. Load the results using ArviZ 1.0 DataTree API:

```python
import arviz as az
import numpy as np

dt = az.from_netcdf("RESULTS_FILE_PATH")
print(f"Groups: {list(dt.children.keys())}")
```

3. Run the full diagnostic workflow:

**Divergences:**
```python
if "sample_stats" in dt.children:
    divergences = dt["sample_stats"]["diverging"].values
    n_div = divergences.sum()
    total = divergences.size
    print(f"Divergences: {n_div} / {total} ({100*n_div/total:.1f}%)")
```

**Summary statistics (R-hat, ESS):**
```python
# ArviZ 1.0 uses 0.89 ETI by default (not 0.94 HDI)
summary = az.summary(dt, var_names=["~log_likelihood"])
print(summary)

# Flag problematic parameters
rhat_bad = summary[summary["r_hat"] > 1.01]
ess_bulk_bad = summary[summary["ess_bulk"] < 400]
ess_tail_bad = summary[summary["ess_tail"] < 400]

if len(rhat_bad) > 0:
    print(f"\nR-hat > 1.01 for: {list(rhat_bad.index)}")
if len(ess_bulk_bad) > 0:
    print(f"\nLow ESS bulk (<400) for: {list(ess_bulk_bad.index)}")
if len(ess_tail_bad) > 0:
    print(f"\nLow ESS tail (<400) for: {list(ess_tail_bad.index)}")
```

**Trace and rank plots:**
```python
az.plot_trace(dt, kind="rank_vlines")
```

**Energy diagnostic:**
```python
if "sample_stats" in dt.children:
    energy = dt["sample_stats"]["energy"].values.flatten()
    e_bfmi = np.var(np.diff(energy)) / np.var(energy)
    print(f"E-BFMI: {e_bfmi:.3f} ({'OK' if e_bfmi > 0.3 else 'LOW - poor exploration'})")
```

**LOO-CV / Pareto k diagnostics (if log_likelihood exists):**
```python
if "log_likelihood" in dt.children:
    loo_result = az.loo(dt)
    print(loo_result)
    pareto_k = loo_result.pareto_k
    n_bad = (pareto_k > 0.7).sum().item()
    if n_bad > 0:
        print(f"\n{n_bad} observations with Pareto k > 0.7 (unreliable LOO estimate)")
```

Note: WAIC is not available in ArviZ 1.0. Use PSIS-LOO-CV exclusively.

4. Interpret all results together and provide:
   - A summary of overall sampling quality (good / acceptable / problematic)
   - Specific issues found, ordered by severity
   - Concrete remediation steps for each issue:
     - Divergences: suggest non-centered parameterization, increased target_accept, or prior adjustments
     - High R-hat: suggest more samples, check for multimodality
     - Low ESS: suggest reparameterization, more samples, or thinning
     - Bad Pareto k: suggest model revision for influential observations
