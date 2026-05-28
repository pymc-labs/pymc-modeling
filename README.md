# pymc-modeling

A PyMC Bayesian modeling assistant. Provides skills, tools, commands, and hooks for probabilistic programming with PyMC 6+, PyTensor 3+, and ArviZ 1.0+.

Works with Claude Code and pi.

## Features

- **5 skills** covering core modeling, testing, prior elicitation, model evaluation, and pymc-extras
- **Custom tools** for API lookup, code pattern search, and error diagnosis
- **4 commands**: `/pymc-diagnose`, `/prior-check`, `/shape-check`, `/model-compare`
- **Post-write linting** for deprecated PyMC/ArviZ patterns
- **PyMC import detection** when reading files

## Installation

### Claude Code

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

Or install from source:

```bash
git clone https://github.com/pymc-labs/pymc-modeling
cd pymc-modeling
bash install.sh
```

### pi

Install from source using the included install script:

```bash
git clone https://github.com/pymc-labs/pymc-modeling
cd pymc-modeling
bash install.sh
```

This symlinks the pi extension to `~/.pi/agent/extensions/pymc-modeling`, where pi auto-discovers it. The extension provides built-in tools (no MCP server needed), commands, and event handlers.

### Validate installation

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
| **model-evaluation** | Model comparison, LOO, ELPD, stacking | LOO-CV, `compare()`, model stacking/averaging, Bayes factors |
| **pymc-extras** | `pymc_extras`, splines, R2D2, `fit_laplace` | Splines, distributional regression, horseshoe/R2D2 priors, marginalization, Laplace approximation |

### Tools

**Claude Code** uses an MCP server (`pymc-docs`) that provides:

| Tool | Description |
|------|-------------|
| `pymc_api_lookup` | Look up PyMC/ArviZ function signatures, descriptions, and gotchas |
| `pymc_example_search` | Search for PyMC code examples and patterns |
| `pymc_error_lookup` | Look up common PyMC errors and their fixes |

**pi** provides the same three tools as built-in extension tools (no MCP server required).

### Commands

| Command | Description |
|---------|-------------|
| `/pymc-diagnose` | Generate full ArviZ diagnostic report with interpretation |
| `/prior-check` | Scaffold prior predictive check code for a model |
| `/shape-check` | Run shape validation on a model file |
| `/model-compare` | Scaffold LOO comparison code for multiple models |

### Hooks / Event Handlers

| Hook | Claude Code | pi |
|------|-------------|-----|
| **Keyword detection** | `suggest-skill.sh` (UserPromptSubmit) | `before_agent_start` event |
| **Post-write lint** | `pymc-post-write.sh` (PostToolUse Write/Edit) | `tool_result` event handler |
| **Import detection** | `detect-pymc-stack.sh` (PostToolUse Read) | `tool_result` event handler |
| **Context preservation** | `hooks.json` PreCompact prompt | `before_agent_start` event |

### Agents (Claude Code only)

| Agent | Purpose |
|-------|---------|
| **diagnostics-agent** | Reads ArviZ output, identifies convergence issues (R-hat, ESS, divergences), suggests remediation |
| **model-review-agent** | Reviews PyMC model code for shape errors, broadcasting issues, identifiability problems, prior scale mismatches |
| **prior-elicitation-agent** | Interactive prior selection: takes parameter descriptions, suggests priors with justification |

pi provides equivalent functionality through the extension's `before_agent_start` context injection and custom tools.

## Target stack

This plugin targets **PyMC 6+, PyTensor 3+, ArviZ 1.0+** and teaches these APIs exclusively (no dual-version content).

**Required versions**

| Library | Minimum |
|---|---|
| `arviz` (+ `arviz-base`, `arviz-stats`) | `1.0` (released 2026-03-02) |
| `pymc` | `6.0` |
| `pytensor` | `3.0` |

ArviZ 1.0 is on PyPI. PyMC 6 and PyTensor 3 are unreleased at the time of writing. Install from git while waiting for releases:

```bash
pip install "pymc @ git+https://github.com/pymc-devs/pymc@v6"
pip install "pytensor @ git+https://github.com/pymc-devs/pytensor@v3"
```

**Highlights of the API shift**

- `xarray.DataTree` replaces `arviz.InferenceData`; bracket access: `dt["posterior"]`, not `idata.posterior`
- `az.summary(dt)` or the `ds.azstats.summary()` accessor (after `import arviz_stats`)
- `az.waic` is gone. Use `az.loo`, `az.loo_expectations`, `az.loo_metrics`
- Default interval is **0.89 ETI** (equal-tailed), not 0.94 HDI
- `pm.compile` replaces `pm.compile_pymc`; `pm.sample_prior_predictive(draws=N)` replaces `samples=N`
- `pm.compute_log_likelihood(idata, model=model)` must be called explicitly after sampling
- PyTensor `tag.test_value` is removed. Use `.eval` on a symbolic variable with a point dict
- `Op.L_op` / `Op.R_op` renamed to `Op.pull_back` / `Op.push_forward`

## Contributing

Contributions are welcome. To add a new skill:

1. Create a new directory under `skills/` with a `SKILL.md` file
2. Add keyword triggers to `hooks/scripts/suggest-skill.sh`
3. Update `README.md` with the new skill's description
4. Run `bash scripts/validate-plugin.sh` to verify the plugin structure

## License

MIT. See [LICENSE](LICENSE) for details.
