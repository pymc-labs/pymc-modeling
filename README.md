# pymc-modeling

A Claude Code plugin for PyMC Bayesian modeling. Provides skills, agents, hooks, slash commands, and an MCP server to assist with probabilistic programming using PyMC, ArviZ, and related tools.

## Installation

This plugin is distributed through the `python-analytics-skills` marketplace.

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add pymc-labs/python-analytics-skills

# Install the plugin
claude plugin install pymc-modeling@python-analytics-skills
```

To update to the latest version:

```bash
claude plugin update pymc-modeling
```

To verify the plugin structure during development:

```bash
bash scripts/validate-plugin.sh
```

## What's Included

### Skills

| Skill | Trigger | Description |
|-------|---------|-------------|
| **pymc-modeling** | Bayesian modeling, PyMC code, MCMC | Core PyMC modeling patterns, ArviZ 1.0 API, coords/dims, sampler configuration |
| **pymc-testing** | Testing PyMC models, pytest + PyMC | Mock sampling, fixture patterns, CI-friendly model tests |
| **prior-elicitation** | Prior selection, PreliZ, `find_constrained_prior` | Prior elicitation workflows, constrained priors, prior predictive checks |
| **model-evaluation** | Model comparison, LOO, ELPD, stacking | LOO-CV, WAIC, `compare()`, model stacking/averaging, Bayes factors |
| **pymc-extras** | `pymc_extras`, splines, R2D2, `fit_laplace` | Splines, distributional regression, horseshoe/R2D2 priors, marginalization, Laplace approximation |

### Agents

| Agent | Purpose |
|-------|---------|
| **diagnostics-agent** | Reads ArviZ output, identifies convergence issues (R-hat, ESS, divergences), suggests remediation |
| **model-review-agent** | Reviews PyMC model code for shape errors, broadcasting issues, identifiability problems, prior scale mismatches |
| **prior-elicitation-agent** | Interactive prior selection: takes parameter descriptions, suggests priors with justification |

### Slash Commands

| Command | Description |
|---------|-------------|
| `/pymc-diagnose` | Generate full ArviZ diagnostic report with interpretation |
| `/prior-check` | Scaffold prior predictive check code for a model |
| `/shape-check` | Run shape validation on a model file |
| `/model-compare` | Scaffold LOO comparison code for multiple models |

### Hooks

- **UserPromptSubmit**: Detects PyMC-related keywords and suggests relevant skills
- **PostToolUse (Write/Edit)**: Checks `.py` files for common PyMC mistakes (flat priors, missing `observed=`, deprecated ArviZ patterns)
- **PostToolUse (Write/Edit)**: Reminds to add convergence diagnostics after `pm.sample()`
- **PreCompact**: Preserves modeling context (model spec, convergence issues, decisions) before context compaction

### MCP Server

A PyMC documentation search tool that provides live access to PyMC, ArviZ, and PyMC-extras API docs during modeling sessions.

## Target stack

This plugin targets **PyMC 6+, PyTensor 3+, ArviZ 1.0+** and teaches these APIs exclusively (no dual-version content).

**Required versions**

| Library | Minimum |
|---|---|
| `arviz` (+ `arviz-base`, `arviz-stats`) | `1.0` (released 2026-03-02) |
| `pymc` | `6.0` |
| `pytensor` | `3.0` |

ArviZ 1.0 is on PyPI. PyMC 6 and PyTensor 3 are unreleased at the time of writing — install from git while waiting for releases:

```bash
pip install "pymc @ git+https://github.com/pymc-devs/pymc@v6"
pip install "pytensor @ git+https://github.com/pymc-devs/pytensor@v3"
```

**Highlights of the API shift**

- `xarray.DataTree` replaces `arviz.InferenceData`; bracket access: `dt["posterior"]`, not `idata.posterior`
- `az.summary(dt)` or the `ds.azstats.summary()` accessor (after `import arviz_stats`)
- `az.waic` is gone — use `az.loo`, `az.loo_expectations`, `az.loo_metrics`
- Default interval is **0.89 ETI** (equal-tailed), not 0.94 HDI
- `pm.compile` replaces `pm.compile_pymc`; `pm.sample_prior_predictive(draws=N)` replaces `samples=N`
- `pm.compute_log_likelihood(idata, model=model)` must be called explicitly after sampling
- PyTensor `tag.test_value` is removed — use `.eval` on a symbolic variable with a point dict
- `Op.L_op` / `Op.R_op` renamed to `Op.pull_back` / `Op.push_forward`

The `pymc-modeling` skill and the other four skills cover the full surface.

## Contributing

Contributions are welcome. To add a new skill:

1. Create a new directory under `skills/` with a `SKILL.md` file
2. Add keyword triggers to `hooks/scripts/suggest-skill.sh`
3. Update `README.md` with the new skill's description
4. Run `bash scripts/validate-plugin.sh` to verify the plugin structure

## License

MIT - see [LICENSE](LICENSE) for details.
