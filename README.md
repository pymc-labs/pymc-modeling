# pymc-modeling

A PyMC Bayesian modeling assistant. Provides skills, tools, commands, and hooks for probabilistic programming with PyMC 6+, PyTensor 3+, and ArviZ 1.0+.

Works with Claude Code, Oh My Pi / pi-compatible harnesses, Codex, Gemini, OpenCode, and generic Agent Skills consumers.

## Features

- **5 skills** covering core modeling, testing, prior elicitation, model evaluation, and pymc-extras
- **Custom tools** for API lookup, code pattern search, and error diagnosis
- **4 commands**: `/pymc-diagnose`, `/prior-check`, `/shape-check`, `/model-compare`
- **Post-write linting** for deprecated PyMC/ArviZ patterns
- **PyMC import detection** when reading files

## Installation

### Claude Code marketplace

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

### Source install for one harness

Use the install script when you want local symlinked resources. The script installs **only** the target you name; run it again with another target if you use multiple harnesses.

If the target already has `pymc-modeling` resources, the installer first removes the existing plugin-owned paths and then recreates them from this checkout.

```bash
git clone https://github.com/pymc-labs/pymc-modeling
cd pymc-modeling
bash install.sh <target>
```

Targets:

| Target | Harness | Installed resources |
|--------|---------|---------------------|
| `claude-code` | Claude Code | Full plugin at `~/.claude/plugins/pymc-modeling` |
| `omp` | Oh My Pi / pi-compatible | Extension, skills, agents, and commands at `${PI_CODING_AGENT_DIR:-~/.omp/agent}` |
| `pi` | Legacy pi | Extension at `~/.pi/agent/extensions/pymc-modeling`; skills at `~/.pi/agent/skills` |
| `codex` | Codex | Skills, agents, and commands at `~/.codex/` |
| `gemini` | Gemini | Skills, agents, and commands at `~/.gemini/` |
| `opencode` | OpenCode | Skills and commands at `~/.config/opencode/` |
| `agents` | Generic Agent Skills consumers | Skills at `~/.agents/skills` |

Examples:

```bash
bash install.sh claude-code   # Claude Code only
bash install.sh omp           # Oh My Pi / pi-compatible only
bash install.sh codex         # Codex only
```

Run `bash install.sh --help` to list targets. Set `PI_CODING_AGENT_DIR` before running `bash install.sh omp` if your Oh My Pi-compatible harness uses a non-default agent directory.

### Validate installation

```bash
bash scripts/validate-plugin.sh
```

## Usage

After installation, restart your agent harness so it re-discovers skills, commands, tools, hooks, and extensions.

### Ask normally

The assistant should pick up the PyMC skills automatically when your prompt or files mention PyMC, PyTensor, ArviZ, MCMC diagnostics, priors, model comparison, or related Bayesian modeling tasks.

Example prompts:

```text
Review this PyMC model for shape and identifiability problems.
Help me choose priors for this hierarchical logistic regression.
Diagnose these divergences and low ESS values.
Compare these two models with PSIS-LOO.
```

### Invoke a skill explicitly

If your harness supports skill commands, call the relevant skill directly:

```text
/skill:pymc-modeling build a non-centered hierarchical model
/skill:pymc-testing write pytest tests for this model
/skill:prior-elicitation choose priors for a positive scale parameter
/skill:model-evaluation compare these models with LOO
/skill:pymc-extras use B-splines for a smooth age effect
```

### Use slash commands

The bundled commands expand into task-specific instructions:

```text
/pymc-diagnose
/prior-check
/shape-check
/model-compare
```

### Use tools when available

Claude Code exposes the tools through the `pymc-docs` MCP server. Oh My Pi / pi-compatible harnesses expose the same tools through the TypeScript extension:

```text
pymc_api_lookup("pm.sample")
pymc_example_search("hierarchical non-centered")
pymc_error_lookup("divergences")
```

### Automatic checks

When hooks or extension events are supported:

- prompts mentioning PyMC get extra PyMC 6 / ArviZ 1.0 context
- reading `.py` or `.ipynb` files with PyMC/PyTensor/ArviZ imports surfaces a PyMC guidance reminder
- writing or editing Python files warns about deprecated PyMC/ArviZ patterns

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

**Oh My Pi / pi-compatible harnesses** provide the same three tools as built-in extension tools (no MCP server required). Other harnesses still get the skills and slash-command prompts when they support those resource types.

### Commands

| Command | Description |
|---------|-------------|
| `/pymc-diagnose` | Generate full ArviZ diagnostic report with interpretation |
| `/prior-check` | Scaffold prior predictive check code for a model |
| `/shape-check` | Run shape validation on a model file |
| `/model-compare` | Scaffold LOO comparison code for multiple models |

### Hooks / Event Handlers

| Behavior | Claude Code | Oh My Pi / pi-compatible |
|----------|-------------|--------------------------|
| **Keyword detection** | `suggest-skill.sh` (UserPromptSubmit) | `before_agent_start` event |
| **Post-write lint** | `pymc-post-write.sh` (PostToolUse Write/Edit) | `tool_result` event handler |
| **Import detection** | `detect-pymc-stack.sh` (PostToolUse Read) | `tool_result` event handler |
| **Context preservation** | `hooks.json` PreCompact prompt | `before_agent_start` event |

### Agents

| Agent | Purpose |
|-------|---------|
| **diagnostics-agent** | Reads ArviZ output, identifies convergence issues (R-hat, ESS, divergences), suggests remediation |
| **model-review-agent** | Reviews PyMC model code for shape errors, broadcasting issues, identifiability problems, prior scale mismatches |
| **prior-elicitation-agent** | Interactive prior selection: takes parameter descriptions, suggests priors with justification |

Oh My Pi, Codex, Gemini, and other harnesses use these agent files when they support user-level agent discovery. Oh My Pi / pi-compatible harnesses also provide equivalent context injection and custom tools through the extension.

## Target stack

This plugin targets **PyMC 6+, PyTensor 3+, ArviZ 1.0+** and teaches these APIs exclusively (no dual-version content).

**Required versions**

| Library | Minimum |
|---|---|
| `arviz` (+ `arviz-base`, `arviz-stats`, `arviz-plots`) | `1.0` (released 2026-03-02) |
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
- ArviZ plotting moved to the ArviZverse API: `az.plot_dist` replaces `plot_posterior`/`plot_density`, `az.plot_ppc_dist` replaces `plot_ppc`, and `az.plot_trace_dist` replaces density+trace `plot_trace`
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
