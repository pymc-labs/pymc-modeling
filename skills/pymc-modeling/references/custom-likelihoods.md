# Custom distributions, external densities and simulators

Use built-in distributions when they express the model. A custom likelihood needs
independent correctness checks, not just an executable graph.

| Scientific information | Interface | Boundary |
|---|---|---|
| Composition of existing random variables | `CustomDist(...,dist=...)` | Automatic density derivation only for supported measurable graphs. |
| Normalized density and numerical generator | `CustomDist(...,logp=...,random=...)` | Names/shapes alone do not show the generator matches the density. |
| Density only | `CustomDist(...,logp=...)` | No predictive generator is implied. |
| Extra factor or constraint | `Potential` | No observed RV, automatic pointwise likelihood or predictive generator. |
| Simulator and discrepancy only | `Simulator` with SMC | Summary/epsilon-dependent ABC pseudo-likelihood, not exact data likelihood. |

Use CustomDist rather than deprecated DensityDist. Construction internals are not
alternative modeling APIs.

## Callback and observation contracts

- `logp(value,*parameters)` returns symbolic pointwise log density.
- `random(*parameters,rng=None,size=None)` returns numerical draws using the
  supplied generator and requested size; never reseed inside it.
- `dist(*parameters,size)` returns a symbolic random graph, as an alternative to
  `random`, not in addition to it.
- Optional `logcdf(value,*parameters)` and `support_point(rv,size,*parameters)`
  add capabilities; a support point is an initializer, not a posterior estimate.

For a scalar-event binomial illustration using the project's counts and log odds:

```python
import pymc as pm
from scipy.special import expit

def logp(value, eta, trials):
    return pm.logp(pm.Binomial.dist(n=trials, logit_p=eta), value)

def random(eta, trials, rng=None, size=None):
    return rng.binomial(trials, expit(eta), size=size)

with model:
    pm.CustomDist("successes", eta, trials, logp=logp, random=random,
                  signature="(),()->()", dtype="int64",
                  observed=successes, dims="batch")
```

This demonstrates the interface; use `pm.Binomial` directly for ordinary work.
A genuine external density replaces logp with a validated PyTensor Op. The optional
`pytensor-workflows` skill includes a `BinomialLogpOp` example; this skill does not
require it. That example takes equal-length float64 log-odds and integer count
vectors and returns one log probability per batch, without scalar broadcasting.
An intercept wrapper must broadcast explicitly if adopting that contract.

Validate observations **before** dtype casting: reject fractional/nonfinite counts,
negative trials, successes outside 0..trials and incompatible shapes. Integer
derivatives are undefined, not zeros. A vector parameter batch is not a vector
event: keep scalar-event logp unreduced for pointwise scoring. A Potential for the
same complete likelihood intentionally sums those factors.

## Check density, derivatives and generation independently

Compare normalized pointwise density to an independent library/formula, full joint
density to likelihood plus priors at multiple points, and gradients to independent
finite differences. Test nonuniform incoming cotangents for vector-output Ops,
not only the gradient of a sum. For small models use quadrature or analytic
posterior moments; a second implementation sharing the same Op is not independent.

Verify support, event/batch shape, dtype, parameter boundaries and generator moments.
Check the exact backend lowering used: Numba density compilation does not imply
a numerical random callback is nopython, and neither proves JAX/GPU support.
Preserve fallback warnings. Save real inference before diagnostics; compute
pointwise log likelihood explicitly and check observation alignment. Prior/PPC
discrepancies assess the model separately from numerical implementation.

## Potentials are factors, not generators

`model.potentials` lists registered factors; `potentiallogp` totals them. A soft
penalty is not an exact constraint or normalized probability law. Named-dimension
Potential accepts an xtensor or labelled nonscalar inputs and reduces contributions;
check it does not repeat a broadcasted factor.

Prior predictive generation ignores Potentials and warns. Empty predictive groups
can exist, so inspect generated variables rather than group names. A separately
specified observed generator can be used with a genuine posterior, but a Potential
alone neither supplies nor normalizes it. HMC `QuadPotential` objects instead
represent kinetic energy/mass matrices; they are unrelated to likelihood Potentials.

## ABC with Simulator

Simulator callback order is `(rng,*parameters,size)`. Gaussian/Laplace/custom
distances and identity/sorted/mean/median summaries define different approximations.
Choose epsilon on the summary's units. A summary can discard information; a small
epsilon does not itself guarantee adequacy or computational feasibility.

For n Normal observations with known SD s, a sample-mean summary and Gaussian
kernel width epsilon add variance: integrated ABC summary variance is
`s²/n + epsilon²`, not exact-data `s²/n`. Compare against this target rather than
claim exact posterior recovery. Black-box simulation is not automatically
differentiable; use its supported SMC route instead of inventing gradients for NUTS.
In [PyMC 6.3.1 Simulator source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/simulator.py),
the advertised kullback_leibler distance is unimplemented; inspect the current
release before choosing it.

SMC stage histories can be ragged/object-valued. Preserve them in suitable storage
instead of dropping history to force NetCDF. Particle order is not MCMC time;
use independent-population and approximation checks, not manufactured R-hat/ESS.

## Log-probability graphs and transforms

`conditional_logp({rv:value,...})` returns factors keyed by value variables.
Replace all stochastic parents; finite evaluation of a graph with an unresolved
RV does not make a deterministic objective. `transformed_conditional_logp` needs
complete RV/value/transform mappings and handles Jacobians. Prefer model compilation
for routine use. CDF/survival/quantile dispatch is not supplied for every custom law.

Forward maps constrained values to sampling coordinates; backward reverses it.
Add the backward-map log Jacobian exactly once. Simplex checks use K-1 independent
coordinates; circular wrapping is not globally invertible and Chain order matters.
Float64 parameters may leave lower-precision intermediate constants; investigate
construction-time constant policy for tight survival/quantile checks rather than
casting outputs or relaxing a failed tolerance.

Sources: [CustomDist](https://www.pymc.io/projects/docs/en/stable/api/distributions/custom.html),
[Simulator](https://www.pymc.io/projects/docs/en/stable/api/distributions/simulator.html),
[Potential](https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model.core.Potential.html),
[log probability](https://github.com/pymc-devs/pymc/tree/v6.3.1/pymc/logprob),
[transforms](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/distributions/transforms.py).
