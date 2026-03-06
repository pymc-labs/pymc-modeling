# pymc-modeling

A Claude Code plugin for PyMC Bayesian modeling. Provides skills, agents, hooks, slash commands, and an MCP server to assist with probabilistic programming using PyMC, ArviZ, and related tools.

## Installation

```bash
git clone https://github.com/pymc-labs/pymc-modeling.git
cd pymc-modeling
bash install.sh
```

The install script detects your OS (Linux/macOS), creates symlinks in `~/.claude/plugins/pymc-modeling/`, and makes hook scripts executable.

To verify the installation:

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

## ArviZ 1.0 Note

This plugin targets **ArviZ 1.0+**, which uses a new `datatree`-based API. Key differences from ArviZ 0.x:

- Access groups via `dt["posterior"]` instead of `idata.posterior`
- Use `dt.dt.summarize()` instead of `az.summary()`
- Use `convert_to_datatree()` instead of `convert_to_inference_data()`

The pymc-modeling skill includes comprehensive ArviZ 1.0 guidance.

## Contributing

Contributions are welcome. To add a new skill:

1. Create a new directory under `skills/` with a `SKILL.md` file
2. Add keyword triggers to `hooks/scripts/suggest-skill.sh`
3. Update `README.md` with the new skill's description
4. Run `bash scripts/validate-plugin.sh` to verify the plugin structure

## License

MIT - see [LICENSE](LICENSE) for details.
