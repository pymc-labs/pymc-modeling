# AGENTS.md

This file provides guidance to AI assistants (Claude Code, pi, and compatible tools) when working with code in this repository.

## Project Overview

A PyMC Bayesian modeling assistant. Provides skills, agents, hooks, commands, and tools that assist with probabilistic programming using PyMC 6+, PyTensor 3+, ArviZ 1.0+, and related tools. Works with both Claude Code (as a plugin) and pi (as an extension).

This is **not** a Python package. It is a Claude Code plugin distributed via the `python-analytics-skills` marketplace and a pi extension located in `.pi/extensions/pymc-modeling/`.

## Key Commands

```bash
# Validate plugin structure (checks dirs, files, JSON, executables)
bash scripts/validate-plugin.sh
```

### Claude Code

```bash
# Install via marketplace
claude plugin marketplace add pymc-labs/python-analytics-skills
claude plugin install pymc-modeling@python-analytics-skills
```

### Benchmark Suite (run from `benchmark/` directory)

```bash
pixi install                              # Install dependencies
pixi run prepare-data                     # Clean/subsample datasets
pixi run test                             # Run unit tests (pytest)
pixi run validate                         # Validation gate (T1 only)
pixi run python -m src.cli run --all      # Full suite (30 runs)
pixi run python -m src.cli score --all    # Score all completed runs
pixi run python -m src.cli analyze        # Generate analysis report
```

### MCP Server (run from `mcp-server/` directory)

Uses pixi with `mcp` PyPI dependency. Server provides `pymc_api_lookup`, `pymc_example_search`, and `pymc_error_lookup` tools.

## Architecture

```
├── skills/           # SKILL.md files with reference docs (5 skills)
│   ├── pymc-modeling/    # Core: PyMC patterns, ArviZ 1.0, nutpie, coords/dims
│   ├── pymc-testing/     # pytest patterns for PyMC models
│   ├── prior-elicitation/# PreliZ, constrained priors
│   ├── model-evaluation/ # LOO-CV, model comparison, stacking
│   └── pymc-extras/      # Splines, R2D2, distributional regression
├── .pi/extensions/   # pi extension
│   └── pymc-modeling/    # index.ts, package.json, data/ (JSON lookup tables)
├── agents/           # Subagent definitions (.md): diagnostics, model review, prior elicitation
├── commands/         # Slash commands (.md): /pymc-diagnose, /prior-check, /model-compare, /shape-check
├── hooks/            # hooks.json + shell scripts
│   ├── hooks.json        # UserPromptSubmit, PostToolUse (Write|Edit), PostToolUse (Read), PreCompact
│   └── scripts/          # suggest-skill.sh (prompt-keyword skill suggestion),
│                         #   detect-pymc-stack.sh (loads skill when a read .py/.ipynb imports pymc/pytensor/arviz),
│                         #   pymc-post-write.sh (lint checks on edited files)
├── mcp-server/       # FastMCP server for PyMC/ArviZ doc search (JSON data files)
├── benchmark/        # Skill effectiveness benchmark (pixi-managed, separate environment)
│   ├── src/              # runner.py, scorer.py, analysis.py, cli.py
│   ├── tasks.yaml        # 5 task definitions with prompts and rubrics
│   └── tests/            # pytest tests for runner, scorer, analysis
└── scripts/          # validate-plugin.sh
```

## Stack versions (critical)

This plugin targets **PyMC 6+, PyTensor 3+, ArviZ 1.0+** (clean break — no PyMC 5 / ArviZ 0.x back-compat content). When editing skill content or benchmark code:

- Bracket access `dt["posterior"]`, not attribute `idata.posterior`
- `az.summary(dt)` or the accessor `ds.azstats.summary()` (requires `import arviz_stats`). `.dt.summarize()` is not a real accessor — xarray reserves `.dt` for datetime.
- `az.convert_to_datatree()` not `convert_to_inference_data()`
- `az.waic` is removed — use `az.loo` and `az.loo_metrics` / `az.loo_expectations`
- `pm.compile` not `pm.compile_pymc`
- `pm.sample_prior_predictive(draws=N)` not `samples=N`
- After `pm.sample(...)`, call `pm.compute_log_likelihood(idata, model=model)` explicitly — passing `compute_log_likelihood=True` to `pm.sample` now emits a `FutureWarning`
- No `tag.test_value` / `config.compute_test_value` in custom models (removed in PyTensor 3) — for debugging, invoke the `.eval` method on a symbolic variable with a point-dict of inputs
- `Op.pull_back` / `Op.push_forward` not `Op.L_op` / `Op.R_op`

**Benchmark note:** The benchmark uses ArviZ 1.0 but keeps PyMC pinned to 5 until PyMC 6 ships (acceptable transitional state; NetCDF files from PyMC 5 load fine under ArviZ 1.0).

## Development Notes

- Skills are Markdown files (`SKILL.md`) with YAML frontmatter (name, description) and reference subdirectories
- Hooks are defined in `hooks/hooks.json` and delegate to bash scripts in `hooks/scripts/`; every script must exit 0 even on error (a failing hook must never block the tool call)
- Skill auto-loading fires from two angles: `suggest-skill.sh` matches keywords in the user's prompt, and `detect-pymc-stack.sh` inspects files Claude reads and loads the skill when they import pymc/pytensor/arviz
- The pi extension lives in `.pi/extensions/pymc-modeling/index.ts`. When adding a tool or command, update both the MCP server (for Claude Code) and the extension (for pi). Both consume the JSON data files in `.pi/extensions/pymc-modeling/data/`
- The benchmark uses pixi for environment management — always use `pixi run` from `benchmark/`
- Benchmark isolates conditions by running Claude in `/tmp/benchmark/` to prevent plugin hook contamination
- Adding a new skill: create `skills/<name>/SKILL.md`, add triggers to `hooks/scripts/suggest-skill.sh`
