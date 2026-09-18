# pymc-agent-skills

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

Or copy individual directories from `skills/` into your agent's supported skills
location. Keep the whole directory, including references, any scripts and `LICENSE`.
No custom installer or running service is required.

### Published repository (publication required)

Use the following commands only after the matching skills and marketplace
manifest have been published to `fonnesbeck/pymc-agent-skills`. A local checkout,
installer discovery or manifest validation does not establish remote availability
or successful plugin installation.

For the Skills CLI:

```bash
npx skills add fonnesbeck/pymc-agent-skills
```

For a Claude Code managed bundle:

```text
/plugin marketplace add fonnesbeck/pymc-agent-skills
/plugin install pymc-agent-skills@pymc-agent-skills
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

## License

Adapted from [pymc-labs/pymc-modeling](https://github.com/pymc-labs/pymc-modeling/tree/b41e66104685ba06fce89091cf0706e9bc21872e).
Licensed under [MIT](LICENSE). Each skill includes the same license for
standalone distribution, including the original PyMC Labs copyright notice.
