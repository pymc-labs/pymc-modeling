# PyMC Extras: select the extension for the model

Use `pymc-extras` when it supplies the distribution, elimination algorithm,
state-space implementation or model-building interface the task needs. Prefer
its maintained implementation to writing a second Kalman filter, Markov-chain
log density or optimizer wrapper. Core PyMC remains sufficient for ordinary
regression, mixtures with `pm.Mixture`, manual non-centering and most likelihoods.
Installing extras does not justify changing a scientifically appropriate model.

## Resolve compatibility before choosing an API

This guidance targets PyMC 6.4.0, pymc-extras 0.16.0, PyTensor 3.3.7 and ArviZ 1.3.0 with the DataTree API.
PyMC and extras both need PyTensor `>=3.3.6,<3.4`.
Extras needs Python >=3.12, PyMC `>=6.4.0,<6.5`, ArviZ `>=1.2,<2`, and PreliZ `>=0.27,<0.29`.

These bounds define compatible versions, not the checked stack.
The manifest keeps the upstream PyTensor floor of 3.3.6.
Both solve groups in `pixi.lock` resolve to 3.3.7, including the default and JAX environments.

Let the consuming project's package manager resolve dependencies. Never bypass dependency
pins. Do not install into or replace a working environment merely to load a skill.
For an older project, inspect that release's signatures and source instead.

Record the actual environment before adapting an example:

```python
from importlib.metadata import version
import inspect
import pymc_extras as pmx

for package in ("pymc-extras", "pymc", "pytensor", "arviz", "preliz"):
    print(package, version(package))
print(inspect.signature(pmx.marginalize))
```

The distribution is named `pymc-extras`; the import is `pymc_extras`, conventionally
`pmx`. Do not use the old `pymc_experimental` namespace. Root exports are not
identical to submodule APIs: use the explicit imports in the focused references.
Stable documentation may advance beyond an installed release. Check tagged source
and installed `help()` before relying on defaults or optional backend support.

## Choose by the operation, not the package name

| Need | Extension and selection criterion | Read |
|---|---|---|
| Remove finite latent states | Exact marginalization; enables continuous inference without discarding latent uncertainty | [Mixtures and marginalization](mixtures.md) |
| Remove conjugate continuous variables | Exact supported conjugacy; check the actual dependent graph, not just the prior's family | [Mixtures and marginalization](mixtures.md) |
| Integrate a nonconjugate latent block approximately | Laplace marginalization; the remaining posterior is approximate too | [Mixtures and marginalization](mixtures.md) |
| Optimize, initialize or approximate a posterior | Extras MAP, Laplace, Pathfinder, DADVI or specialized INLA; select by geometry and required uncertainty accuracy | [Approximate inference](approximate-inference.md) |
| Experimental continuous inference with MLX | Unadjusted MCLMC retains discretization bias. Check float32 graphs and approximation sensitivity. | [MLX MCLMC](#experimental-mlx-mclmc) |
| Linear-Gaussian temporal latent process | State-space models and structural components, with a separate ETS innovations recursion | [Extras state-space workflows](extras-statespace.md) |
| Latent discrete sequence or joint finite state | `DiscreteMarkovChain` / `JointCategorical`; respect event axes and state ordering | [Extras distributions](extras-distributions.md) |
| Counts, differences, block maxima or threshold exceedances | Specialized distributions with a matching observation process | [Extras distributions](extras-distributions.md) |
| Compress a large independent likelihood | `histogram_approximation`; binning changes the likelihood and requires an approximation check | [Extras distributions](extras-distributions.md) |
| Configure and serialize hierarchical priors | `Prior`, `Censored`, `Scaled` and deserialization | [Extras model tools](extras-model-tools.md) |
| Package model construction or deployment | `as_model` for a factory; `ModelBuilder` only when fit/predict/persistence justify a class | [Extras model tools](extras-model-tools.md) |
| Tune hierarchical parameterization | VIP learns interpolation between centered and non-centered forms; refit and diagnose afterward | [Extras model tools](extras-model-tools.md) |
| Transfer an earlier posterior to a new model | `prior_from_idata` is a joint Gaussian approximation, not exact sequential updating | [Extras model tools](extras-model-tools.md) |
| R-squared shrinkage or spline bases | `R2D2M2CP` with declared scales; spline utilities with identifiable penalties | [Shrinkage](../../prior-elicitation/references/shrinkage.md), [splines](splines-distributional.md) |

## Validate the changed computation

1. Check support, event/batch shapes, initial log density and predictive draws.
   A successful import or finite objective is not a validation of the algorithm.
2. Compare a small case against an independent oracle: finite enumeration for
   discrete elimination, conjugate posterior moments for Gaussian inference,
   SciPy with matched parameter conventions for distributions, or an independent
   Kalman recursion for state-space predictions.
3. State which operation is exact, which is approximate and which outputs are
   conditioned on observations. Approximation draws and optimization paths are
   not converged MCMC chains. A MAP estimate supplies no posterior uncertainty.
4. For supported elimination, recover marginalized quantities before you interpret their posterior or generate same-unit predictions.
   Extras 0.16.0 also supports Laplace conditional recovery with a Gaussian approximation at the conditional mode.
   Recovery does not make an approximation exact.
5. Save parameter inference separately from downstream simulations. Specify
   training identities, forecast origins, future covariates and random seeds.
   Do not tune priors, approximation checks or seeds to obtain favorable results.

Optional packages are method-specific. BlackJAX-based inference requires a
compatible JAX/BlackJAX installation; histogram workflows can need xhistogram or
Dask. The package's `complete` extra is not a promise of every inference backend
or accelerator. Install only the dependencies of the chosen operation and inspect
backend restrictions before compiling the model.

## Experimental MLX MCLMC

Import `fit_mlx_mclmc` from `pymc_extras.inference.mlx_mclmc`, not the `pmx.fit` dispatcher.
The sampler targets float32 continuous models.
Its unadjusted dynamics retain discretization bias.

The upstream `mlx_mclmc` extra installs MLX only on macOS arm64.
The sampler source checks model dtype and discrete variables, not the operating system.
Its momentum kernel uses Metal when available and MLX operations otherwise.

Do not confuse that installation marker with a runtime platform check.
The repository's Linux `mlx[cpu]` environment checks custom Ops, not this sampler.
Check approximation sensitivity and posterior agreement with an independent reference before scientific use.

Sources: [sampler interface](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/inference/mlx_mclmc/mlx_mclmc.py),
[model checks](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/inference/mlx_mclmc/logp.py),
and [CPU/Metal momentum selection](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/inference/mlx_mclmc/kernel.py).

## Primary sources

- [Official API inventory](https://www.pymc.io/projects/extras/en/stable/api_reference.html).
- [0.16.0 dependency metadata](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pyproject.toml).
- [0.16.0 package exports](https://github.com/pymc-devs/pymc-extras/blob/v0.16.0/pymc_extras/__init__.py).
- [Published release metadata](https://pypi.org/project/pymc-extras/0.16.0/).
