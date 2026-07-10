# pymc-modeling

An agent plugin for Bayesian modeling with **PyMC 6+**, **PyTensor 3+**, and **ArviZ 1.0+**. It packages modeling guidance, task prompts, agents, and integrations for Claude Code, Oh My Pi / pi-compatible harnesses, Codex, Gemini, OpenCode, and generic Agent Skills consumers.

## Features

- **5 skills** for PyMC modeling, testing, prior elicitation, model evaluation, and `pymc-extras`
- **3 agents** for model review, convergence diagnostics, and interactive prior elicitation
- **4 task commands** for diagnostics, prior predictive checks, shape validation, and LOO comparison
- **3 lookup tools** for API documentation, modeling patterns, and common errors
- Claude Code hooks and Oh My Pi / legacy pi extension events that surface current PyMC guidance

## Installation

### Source installation

This repository is installed directly; use its source installer for every supported harness.
For every target, install `bash` and Git. The `claude-code` target additionally requires `pixi` to start its MCP server and `jq` for its hooks and checkout validation.


```bash
git clone https://github.com/pymc-labs/pymc-modeling.git
cd pymc-modeling
bash install.sh <target>
```

The installer supports Linux and macOS. It installs only the target you name; run it again for each additional harness. For the selected target, it removes existing `pymc-modeling`-owned paths and recreates the supported resources from this checkout.

| Target | Harness | Installed resources |
|--------|---------|---------------------|
| `claude-code` | Claude Code | Complete plugin at `~/.claude/plugins/pymc-modeling`, including skills, agents, commands, hooks, and the `pymc-docs` MCP server |
| `omp` | Oh My Pi / pi-compatible | TypeScript extension, skills, agents, and commands at `${PI_CODING_AGENT_DIR:-~/.omp/agent}` |
| `pi` | Legacy pi | TypeScript extension and skills at `~/.pi/agent` |
| `codex` | Codex | Skills, agents, and commands at `~/.codex` |
| `gemini` | Gemini | Skills, agents, and commands at `~/.gemini` |
| `opencode` | OpenCode | Skills and commands at `~/.config/opencode` |
| `agents` | Generic Agent Skills consumers | Skills at `~/.agents/skills` |

Examples:

```bash
bash install.sh claude-code
bash install.sh omp
bash install.sh codex
```

Run `bash install.sh --help` to list the canonical targets. Set `PI_CODING_AGENT_DIR` before `bash install.sh omp` when an Oh My Pi-compatible harness uses a non-default agent directory.

### Validate the checkout

```bash
bash scripts/validate-plugin.sh
```

Restart the target harness after installation so it discovers the installed skills, commands, hooks, and extension.

## Usage

### Ask normally

Skills activate when the prompt or a read Python/notebook file concerns PyMC, PyTensor, ArviZ, MCMC diagnostics, priors, model comparison, or related Bayesian modeling work.

```text
Review this PyMC model for shape and identifiability problems.
Help me choose priors for this hierarchical logistic regression.
Diagnose these divergences and low ESS values.
Compare these two models with PSIS-LOO.
```

### Invoke a skill explicitly

When the harness supports explicit skill commands:

```text
/skill:pymc-modeling build a non-centered hierarchical model
/skill:pymc-testing write pytest tests for this model
/skill:prior-elicitation choose priors for a positive scale parameter
/skill:model-evaluation compare these models with LOO
/skill:pymc-extras use B-splines for a smooth age effect
```

### Use task commands

Claude Code receives the command files through the plugin. Oh My Pi and legacy pi register the same commands natively:

```text
/pymc-diagnose
/prior-check
/shape-check
/model-compare
```

### Use lookup tools

Claude Code exposes the tools through the `pymc-docs` FastMCP server. Oh My Pi and legacy pi expose the same tools through the TypeScript extension; other targets receive only the resource types their harness supports.

```text
pymc_api_lookup("pm.sample")
pymc_example_search("hierarchical non-centered")
pymc_error_lookup("divergences")
```

## Included resources

### Skills

| Skill | Use it for |
|-------|------------|
| **pymc-modeling** | Core PyMC modeling, ArviZ 1.0 APIs, coordinates/dimensions, sampler configuration, diagnostics, and model criticism |
| **pymc-testing** | Fast, deterministic pytest coverage for PyMC models |
| **prior-elicitation** | Prior selection, constrained priors, PreliZ workflows, and prior predictive checks |
| **model-evaluation** | LOO-CV, ELPD, Pareto-$k$ diagnostics, stacking, and model averaging |
| **pymc-extras** | Splines, distributional regression, R2D2 and horseshoe priors, marginalization, and Laplace approximation |

### Tools

| Tool | Description |
|------|-------------|
| `pymc_api_lookup` | Look up PyMC or ArviZ function signatures, descriptions, and gotchas |
| `pymc_example_search` | Search bundled PyMC code patterns and examples |
| `pymc_error_lookup` | Look up common PyMC errors, diagnostics, and fixes |

### Commands

| Command | Description |
|---------|-------------|
| `/pymc-diagnose` | Produce an ArviZ diagnostic report for sampling results |
| `/prior-check` | Guide a model's prior predictive check |
| `/shape-check` | Validate model shapes and dimensions before sampling |
| `/model-compare` | Compare models with PSIS-LOO-CV |

### Agents

| Agent | Purpose |
|-------|---------|
| **diagnostics-agent** | Interprets ArviZ output, including R-hat, ESS, divergences, and remediation options |
| **model-review-agent** | Reviews PyMC code for shape, broadcasting, identifiability, and prior-scale problems |
| **prior-elicitation-agent** | Guides interactive prior selection from parameter descriptions and domain constraints |

## Integrations

### Claude Code

The Claude Code plugin includes the `pymc-docs` FastMCP server and these hooks:

| Event | Behavior |
|-------|----------|
| `UserPromptSubmit` | `suggest-skill.sh` identifies relevant PyMC topics and suggests the appropriate skill |
| `PostToolUse` on `Read` | `detect-pymc-stack.sh` notices PyMC, PyTensor, or ArviZ imports in `.py` and `.ipynb` files |
| `PostToolUse` on `Write` or `Edit` | `pymc-post-write.sh` warns about deprecated PyMC and ArviZ patterns in Python code |
| `PreCompact` | Preserves the active model specification, diagnostics, and next steps in compacted context |

### Oh My Pi and legacy pi

The TypeScript extension:

- contributes non-conflicting project-local skills through `resources_discover`;
- adds PyMC 6 / PyTensor 3 / ArviZ 1.0 context during `before_agent_start`;
- detects PyMC-stack imports after `read` tool results;
- lints Python files after `write` and `edit` tool results; and
- registers the three lookup tools and four task commands natively.

## Target stack

This project teaches the **PyMC 6+ / PyTensor 3+ / ArviZ 1.0+** API cutover exclusively. It does not provide compatibility guidance for earlier releases.

Key differences from the older APIs:

- `pm.sample()` returns an `xarray.DataTree`; access groups with brackets such as `idata["posterior"]`.
- Use `az.summary(dt)` or `ds.azstats.summary()` after importing `arviz_stats`.
- `az.waic` is removed; use `az.loo`, `az.loo_expectations`, and `az.loo_metrics`.
- Use `pm.compile`, not `pm.compile_pymc`.
- Use `pm.sample_prior_predictive(draws=N)`, not `samples=N`.
- Call `pm.compute_log_likelihood(idata, model=model)` explicitly after sampling when log likelihood is required.
- PyTensor 3 removes `tag.test_value`; evaluate symbolic variables with a point dictionary instead.
- `Op.pull_back` and `Op.push_forward` replace `Op.L_op` and `Op.R_op`.

The project does not install PyMC itself. Install a compatible modeling stack in the environment where the agent will work.

## Benchmark

[`benchmark/`](benchmark/README.md) measures whether the `pymc-modeling` skill improves Bayesian models across five targeted tasks. It uses its own Pixi environment:

```bash
cd benchmark
pixi install
pixi run validate
pixi run test
```

See the [benchmark guide](benchmark/README.md) for data preparation, full runs, scoring, analysis, and platform requirements.

## Contributing

To add a skill:

1. Create `skills/<name>/SKILL.md`.
2. Add the appropriate keyword triggers to `hooks/scripts/suggest-skill.sh`.
3. Update this README with the skill's purpose.
4. Run `bash scripts/validate-plugin.sh`.

## License

MIT. See [LICENSE](LICENSE) for details.
