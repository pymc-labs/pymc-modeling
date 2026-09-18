# pymc-modeling

Agent skills for Bayesian modeling with PyMC, PyTensor and ArviZ. Each skill is
a folder of instructions and focused references in the
[Agent Skills format](https://agentskills.io/specification).

## Skills

| Skill | Use it for |
|---|---|
| [pymc-modeling](skills/pymc-modeling/SKILL.md) | Model specification, inference, predictions and specialized model families |
| [prior-elicitation](skills/prior-elicitation/SKILL.md) | Prior selection, elicitation, predictive checks and shrinkage |
| [arviz-diagnostics](skills/arviz-diagnostics/SKILL.md) | MCMC diagnostics, predictive checks, LOO and model comparison |
| [pytensor-workflows](skills/pytensor-workflows/SKILL.md) | Symbolic graphs, shapes, compilation, gradients and custom Ops |

## Install

### Local checkout

From the repository root, use the [Skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add .
```

Choose the skills and agents you want.

For a local Claude Code managed bundle, run these commands from the repository
root inside Claude Code:

```text
/plugin marketplace add ./
/plugin install pymc-modeling@pymc-modeling
```

This local installation does not require the renamed marketplace to be published.
Use a clean checkout for this installation: Claude Code can copy untracked
files too, including a maintainer's `.pixi/` environments.

Or copy individual directories from `skills/` into your agent's supported skills
location. Keep the whole directory, including references, any scripts and `LICENSE`.
No custom installer or running service is required.

### Published repository (publication required)

Use the following commands only after the matching skills and marketplace
manifest have been published to `pymc-labs/pymc-modeling`. The published manifest
must name both the marketplace and its plugin `pymc-modeling`. Until then, use
the local-checkout instructions above. Local validation does not establish remote
availability or successful remote plugin installation.

For the Skills CLI:

```bash
npx skills add pymc-labs/pymc-modeling
```

For a Claude Code managed bundle:

```text
/plugin marketplace add pymc-labs/pymc-modeling
/plugin install pymc-modeling@pymc-modeling
```

Choose either the plugin or the Skills CLI/manual installation, not both, to
avoid duplicate skills.

## Use

Ask your agent to perform a relevant task, or invoke the skill by name using
your agent's skill mechanism. For example:

- “Build a hierarchical model for these grouped observations.”
- “Check whether these priors imply plausible outcomes.”
- “Diagnose this posterior and assess whether LOO is reliable.”
- “Find the shape or gradient error in this PyTensor graph.”

The guidance targets PyMC 6+, PyTensor 3+ and ArviZ's DataTree API. Code examples
use the consuming project's compatible Python environment. Optional packages
such as PreliZ, pymc-extras, BART and alternative backends are needed only for
the workflows that use them. Consult version-matched documentation; do not
replace a working environment just to install these instructions.

## Maintainer checks

These checks exercise the educational custom Op and its CPU backends, not the
full set of modeling examples or accelerator hardware. They do not install
anything into a consuming project's environment.

The Linux test environments require glibc 2.35 or newer. From this repository:

```bash
pixi run --locked test
pixi run --locked -e jax test -m jax
pixi run --locked -e pytorch test -m pytorch
pixi run --locked -e mlx test -m mlx
```

The default environment checks the base Op and Numba; other backend tests skip
when their optional packages are absent. The named environments install the
corresponding backend, with CPU-only PyTorch and MLX packages. `pixi.lock` records
exact resolved versions, and each pytest session prints the versions actually
exercised. Tests compare values to SciPy and weighted derivatives and curvature
to analytic references, and check support boundaries and input errors.

The GitHub Actions workflow runs all four environments with the lockfile.
Use `pixi run check` and `pixi run format` for regression-test code quality.
These maintainer tools are not needed to install or use the skills.

## License

Adapted from [pymc-labs/pymc-modeling](https://github.com/pymc-labs/pymc-modeling/tree/b41e66104685ba06fce89091cf0706e9bc21872e).
Licensed under [MIT](LICENSE). Each skill includes the same license for
standalone distribution, including the original PyMC Labs copyright notice.
