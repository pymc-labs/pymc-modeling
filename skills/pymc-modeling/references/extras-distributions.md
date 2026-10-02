# Specialized distributions and transforms in pymc-extras

Use these families when the observation process or prior structure calls for
something beyond core PyMC—not simply because the distribution is available.
This reference targets **pymc-extras 0.15.1**. Import families from
`pymc_extras.distributions`; use `.dist(...)` for unnamed distributions and
`Family("y", ..., observed=data)` for likelihoods. Follow the environment contract
in [pymc-extras](pymc-extras.md). The [official distribution index](https://www.pymc.io/projects/extras/en/stable/api/distributions.html)
is useful for discovery; the release sources linked below determine exact APIs.

Validate data support before casting. In particular, several custom discrete
log densities do not reject fractional observations explicitly. Broadcasting
creates tensor shapes, not a scientific independence assumption. A density,
random generator, CDF, gradient and backend implementation are separate
capabilities; an exported class does not establish all of them.

## Discrete states: chains versus joint configurations

Choose `DiscreteMarkovChain` for a finite-state process with dependence on recent
states. Choose `JointCategorical` for a probability table over a small vector of
states, including correlated initial states of a higher-order chain. Neither is
a continuous parameter that NUTS can sample directly.

### Chain shape and ordering

For `n_lags=L`, `K` states and `steps=S`:

- The chain's final support axis has **`L + S` states**. `steps` counts transitions,
  not the total length. `shape=(T,)` instead sets total length, giving `T-L`
  transitions; batch axes precede the final time axis.
- Homogeneous `P` has shape `(*batch, K, ..., K)`, with `L+1` state axes.
  `P[..., x[t-L], ..., x[t-1], x[t]]` is the next-state probability: oldest
  lag first, next state last. Entries must be nonnegative and each last-axis
  row must sum to one. All state axes have equal length.
- With `time_varying_P=True`, shape is `(*batch, S, K, ..., K)`; transition
  slice `s` produces the state at index `L+s`. The time axis must have length
  `S`, and can supply `steps` when omitted. Without the flag it is a batch axis.
- Specify exactly one of `P` and **`logit_P`**, the actual release argument name;
  softmax normalizes `logit_P` along the last axis. Do not copy the inconsistent
  `P_logits`/`P_logit` names in the release docstring.
- Supply an unregistered `init_dist` via `.dist()`. It is cloned, not a shared
  realization. Scalar-support `pm.Categorical.dist(...)` supplies independent
  initial states; vector-support `JointCategorical.dist(..., n_lags=L)` supplies
  their joint law. Its support length must equal `L`. Explicit initial
  probabilities are preferable to silently accepting a uniform assumption.

Observed and latent state codes must be integers in `0..K-1`. The chain logp
uses tensor indexing; do not expect every bad code to produce a clean
`-inf` rather than an indexing error or negative-index alias. Direct chain
logp sums transitions and initial-state density, yielding one logp per batch
chain—not one independent observation per time point.

[Chain implementation and examples](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/timeseries.py).

### JointCategorical is not a row-normalized categorical matrix

`JointCategorical.dist(p=p, n_lags=L)` consumes **all trailing `L` axes** of
`p`, each of length `K`, and returns a vector of length `L`. For `L=2`,
`p[i,j]` is `P(x[0]=i, x[1]=j)`. Normalize the whole `K**L` table within
each batch, not its rows. Alternatively use `logit_p` of the same shape;
the implementation flattens the joint table into one categorical distribution.
`p` and `logit_p` are mutually exclusive. Zero entries make configurations
impossible. Leading axes are batch axes, and table storage grows exponentially
with `L`.

The release logp encodes a vector into a single base-`K` integer. Validate **each**
entry and the vector length beforehand: an invalid entry can alias another
flattened state instead of being rejected. Sampling and logp are implemented;
do not assume generic discrete marginalization or a specialized step method for
an arbitrary standalone joint RV merely from the export. Its use as a chain's
joint initial distribution is implemented explicitly.

[Joint implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/multivariate/joint_categorical.py).

### Small HMM with exact state marginalization

The following constructs a two-state model and integrates its latent path out.
The separated emission priors are illustrative scientific assumptions that also
anchor labels, not a universal identification strategy.

```python
import numpy as np
import pymc as pm
from pymc_extras.distributions import DiscreteMarkovChain
from pymc_extras.marginal import marginalize

y = np.array([-1.1, -0.8, 0.9, 1.2, 0.7, -0.6])
with pm.Model(coords={"time": np.arange(y.size), "state": [0, 1]}) as hmm:
    P = pm.Dirichlet("P", a=np.array([[4.0, 1.0], [1.0, 4.0]]))
    mean = pm.Normal("mean", mu=[-1.0, 1.0], sigma=0.25, dims="state")
    sigma = pm.HalfNormal("sigma", sigma=0.5)
    states = DiscreteMarkovChain(
        "states", P=P, init_dist=pm.Categorical.dist(p=[0.5, 0.5]),
        steps=y.size - 1, dims="time",
    )
    pm.Normal("y", mu=mean[states], sigma=sigma, observed=y, dims="time")

collapsed_hmm = marginalize(hmm, ["states"])
# Fit collapsed_hmm with the ordinary continuous-sampling workflow.
```

This is the log-space HMM forward algorithm, not a point estimate of the path.
Release implementations cover higher-order chains, time-varying transitions
and joint initial states; cost still grows rapidly with lag order/state count.
Emission likelihoods must remain separable over time conditional on states.
Do not insert a state-dependent `pm.Deterministic` or `pm.Potential` before
marginalization: the public transformation rejects these dependents. Cross-time
emission reductions/dependencies require a different likelihood, not blind
reuse of the forward algorithm. Recover states conditionally after fitting;
see [mixtures and marginalization](mixtures.md) for marginalization/recovery APIs.
For forecasting distinguish posterior-smoothed training states from a new path
propagated from the final state. Marginal likelihood contributions are whole
sequences, not iid time-point terms for naive LOO.

[HMM implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/marginal/distributions/discrete_markov_chain.py),
[public transformation restrictions](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/model/marginal/marginalize.py),
[release examples/tests](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/tests/model/marginal/test_discrete_markov_chain.py).

## Counts and signed differences

| Family and selection | Parameters and boundaries | Important failure mode |
|---|---|---|
| `GeneralizedPoisson`: under- or overdispersed counts when its dispersion mechanism is defensible | `mu>0`; release logp checks `max(-1,-mu/4) <= lam <= 1`. `lam=0` is Poisson; negative/positive gives under-/overdispersion. Despite the argument name, expectation is `mu/(1-lam)`, variance `mu/(1-lam)**3`, not `mu`. | Use `lam<1` for finite moments and practical prediction. At `lam=1` these formulas diverge and the branching RNG can produce extremely large, slow draws. For `lam<0` support ends at the largest integer with `mu+lam*x>0`; moving that endpoint across data creates hard likelihood walls. Avoid equality `mu+lam*x=0`, where the log formula is delicate. |
| `BetaNegativeBinomial`: heterogeneity in a negative-binomial success probability, not bounded trial counts | `alpha,beta,r>0`; nonnegative integer failures before `r` successes, with positive real `r` also admitted. Intended law is `p~Beta(alpha,beta)`, `x~NegativeBinomial(p=p,n=r)`. Mean `r*beta/(alpha-1)` requires `alpha>1`; variance requires `alpha>2`. | **Release predictive mismatch:** the custom logp describes that law, but its generator calls `NegativeBinomial.dist(p,r)` positionally, interpreted as `mu=p,alpha=r` in PyMC 6.3. Do not trust native prior/posterior predictive draws from this helper in 0.15.1. |
| `Skellam`: the signed difference of two conditionally independent Poisson counts | Integers on the entire real line; `mu1,mu2>=0`; mean `mu1-mu2`, variance `mu1+mu2`. Independence is substantive; unequal exposures belong in their respective rates. | The release logp uses `log(mu1)-log(mu2)` and an unscaled Bessel `iv`. Keep rates strictly positive for this implementation; permitted zero-rate boundaries can be undefined numerically. Extreme arguments can overflow/underflow. A correlated pair does not generally have this difference law. |

If a regression predictor represents the generalized-Poisson mean `m`, pass
`mu=m*(1-lam)`. For negative dispersion also satisfy the `-mu/4` bound and the
observed-count support. Core NegativeBinomial is usually simpler for ordinary
overdispersion; a Beta mixture should represent additional probability
heterogeneity, not merely add a parameter.

The BetaNegativeBinomial mismatch is visible in the
[release discrete source](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/discrete.py)
and [PyMC 6.3 NegativeBinomial signature](https://github.com/pymc-devs/pymc/blob/v6.3.0/pymc/distributions/discrete.py).
With `alpha=3, beta=2, r=2`, the intended mean is 2. A 50,000-draw check
gave mean 0.60512 from the released helper versus 1.99514 from the explicit
Beta/NegativeBinomial generator. This is not ordinary Monte Carlo variation.
Use an explicit generative model when prediction is required:

```python
import numpy as np
import pymc as pm

# Illustrative fixed hyperparameters; each observation has its own probability.
y_count = np.array([0, 2, 1, 4], dtype=np.int64)
with pm.Model(coords={"observation": np.arange(y_count.size)}) as beta_nb_model:
    probability = pm.Beta("probability", alpha=3.0, beta=2.0, dims="observation")
    pm.NegativeBinomial(
        "y", p=probability, n=2.0, observed=y_count, dims="observation",
    )
```

This adds latent probabilities but uses matching density and RNG. For a **new**
observation draw a fresh Beta probability; reusing a fitted observation's
posterior probability instead predicts that observation's latent propensity.
The release discrete families implement logp and random generation, but no
family-specific logcdf/icdf methods; do not assume censoring/truncation wrappers
will work without checking required numerical capabilities.

## Extremes: block maxima, exceedances and body flexibility

Choose the sampling unit before choosing the family:

- **`GenExtreme(mu,sigma,xi)`** models block maxima under an extreme-value
  approximation, not all raw measurements or threshold excesses. `sigma>0`;
  Coles shape `xi` has the **opposite sign** from SciPy `genextreme(c=...)`:
  use `c=-xi`, or pass the SciPy sign with `scipy=True`. Release logp/logcdf
  additionally restrict **`-1<xi<1`**, narrower than the mathematical family.
- **`GenPareto(mu,sigma,xi)`** models values above a chosen threshold `mu`;
  alternatively subtract the threshold and use `mu=0`. `sigma>0`, and
  `xi` has the **same sign** as SciPy `genpareto(c=xi)`. No release logp bound
  on `xi` beyond its support/scale restrictions.
- **`ExtGenPareto(mu,sigma,xi,kappa)`** uses `F(x)=GPD_CDF(x)**kappa`, `kappa>0`.
  `kappa=1` recovers GPD; it changes the body and tail-survival multiplier,
  **not the tail index**. Use it when body flexibility is scientifically
  warranted; tail-only data can confound small `sigma` with large `kappa`.
  It is not a cure for an unjustified threshold or dependent extremes.

For GEV, write `z=(x-mu)/sigma`: density requires `1+xi*z>0`.
Positive `xi` gives a lower endpoint and a heavy upper tail; negative `xi` gives
an upper endpoint. At `xi=0` the law is Gumbel on the real line. For both Pareto
families, support is `x>=mu` when `xi>=0` and **`mu<=x<mu-sigma/xi`** when
`xi<0`. Their upper wall is open in release logp, including `xi=-1`, unlike
SciPy's endpoint limiting-density convention. At the lower endpoint ExtGPD
density is infinite for `kappa<1`, finite for `kappa=1`, zero for `kappa>1`:
none of these creates an atom for exact zeros/threshold recordings.

All three have finite mean only for `xi<1`, finite variance only for
`xi<1/2`; the Pareto families have finite order-`r` moments iff `xi<1/r`.
Choose moment limits from the inferential target, not as an automatic default.
An initialization support point does not make divergent moments finite.

Keep a peaks-over-threshold location fixed at a scientifically chosen threshold.
Changing the threshold changes the data selection. A conditional exceedance
model alone does **not** estimate the exceedance occurrence rate; return levels
need that rate/exposure and dependence assumptions too. Rounded threshold values,
censoring and dry-day zeros need their own observation mechanism.

### Complete conditional tail model

Here the illustrative prior assumes a heavy tail with finite variance, so
`0<xi<0.45` keeps observations inside support without a data-dependent prior
bound. It deliberately excludes exponential and bounded-tail alternatives;
include those in sensitivity analysis when plausible.

```python
import numpy as np
import pymc as pm
from pymc_extras.distributions import GenPareto

threshold = 10.0
exceedances = np.array([10.2, 10.8, 11.4, 12.0, 13.1, 15.0])
with pm.Model(coords={"exceedance": np.arange(exceedances.size)}) as tail_model:
    sigma = pm.LogNormal("sigma", mu=np.log(2.0), sigma=0.5)
    xi_fraction = pm.Beta("xi_fraction", alpha=2.0, beta=3.0)
    xi = pm.Deterministic("xi", 0.45 * xi_fraction)
    GenPareto(
        "y", mu=threshold, sigma=sigma, xi=xi,
        observed=exceedances, dims="exceedance",
    )
```

If negative `xi` is scientifically plausible, every observation must obey
`xi > -sigma/(max(data)-mu)`. This is a **likelihood support constraint**.
Putting the observed maximum into a normalized truncated *prior* changes the
prior and hence the posterior; do not present it as data-independent elicitation.
The release docs recommend such bounds, plus a floor at `-1` to avoid an
unbounded density approaching the upper wall for `xi<-1`. State those modeling
choices explicitly; study initialization/geometry and threshold sensitivity.
For ExtGPD, freeing `mu` with `kappa<1` additionally permits an unbounded density
as `mu` approaches the smallest observation. Near a finite upper wall,
finite-precision gradients can degrade even when logp is accurate.

### Tail numerical interfaces and transforms

GenPareto and ExtGenPareto implement random generation, logp, logcdf,
logccdf and icdf. Their default latent-variable transform is a parameter-aware
**logit-CDF** map, taking either bounded or half-infinite support to the real
line and giving a standard Logistic transformed density. Keep the default
rather than using a simple log transform when the upper wall can move with
`xi`. The transform handles latent values; it does not constrain `sigma,xi`
to keep **observations** in support. Internal transform classes are not public
objects to import or subclass for ordinary modeling.

GenExtreme implements RNG, logp and logcdf, but its release `logcdf` returns
`-inf` whenever `1+xi*z<=0`. For negative `xi`, the mathematically correct CDF
at/above the upper endpoint is one (`logcdf=0`); therefore do not rely on that
release method for upper-tail censoring or bounds extending beyond the wall.
Its density also treats endpoints as out of support. Check actual wrapper
requirements rather than assuming all three families are interchangeable.

[GEV source](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/continuous.py),
[GPD source, recommendations and transform](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/pymc_genpareto.py),
[ExtGPD source and transform](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/pymc_extgenpareto.py),
[ExtGPD endpoint/tail formulas](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/pytensor_extgenpareto.py).

## Chi and Maxwell: radial magnitudes

`Chi(nu)` is the square root of `ChiSquared(nu)`, `nu>0`, with nonnegative
support. It is a norm of independent standard-Normal components when `nu` is
the corresponding integer dimension, not a generic positive-response family
or a chi-square statistic. It has no scale argument: introduce scale explicitly
if needed. `Maxwell(a)` is `a*Chi(3)`, `a>0`, appropriate to a three-dimensional
isotropic Gaussian speed/magnitude model. `a` is the component SD, **not** the
speed SD: mean is `2*a*sqrt(2/pi)`, variance `a**2*(3-8/pi)`.

Both are symbolic `CustomDist` constructions whose density is inferred from
the transformed base law and whose draws follow that construction. `Chi` sets
a log transform for unobserved named variables explicitly; the Maxwell
constructor does not explicitly set one. Check latent initialization/transform
behavior rather than assuming every positive custom distribution chooses it.
Exact zero has no probability atom; do not use these laws as zero-inflated
models. Do not assume a CDF implementation from the symbolic density alone.

[Construction and API source](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/continuous.py).

## Histogram compression: when it changes the likelihood

`histogram_approximation(name, dist, observed=data, **h_kwargs)` returns a
**`pm.Potential`**, computed over data axis zero. It requires `xhistogram`;
Dask inputs are also supported when its dependencies are installed. Dispatch is
by **observed integer dtype**, not by the distribution's discrete/continuous
class. Preserve counts as integers; do not cast continuous measurements to
integers to choose the faster branch.

- With integer data and `min_count=None`, unique-value counts can reproduce the
  original independent log likelihood exactly, provided all observations
  compressed together share parameters (possibly distinct parameters per
  trailing column). This is compression, not a new sampling mechanism.
- `min_count` filters histogram edges based on global occurrence counts. It is
  **not** harmless speed tuning: bins then span omitted values, which can be
  assigned the retained lower-edge value, while values outside the retained
  edge range are lost. It can change both representation and effective likelihood.
- With floating-point data, `n_quantiles` creates quantile edges and contributes
  `sum(count*logpdf(midpoint))`. It is a **midpoint approximation** to raw-data
  density, not `sum(count*log(CDF(upper)-CDF(lower)))` for genuinely binned data.
  Compare posterior targets and tail predictions against raw data and finer
  resolution. Repeated edges, constant data, sharp density curvature and
  support boundaries need particular scrutiny.
- Compression removes row identities. It is inappropriate when conditional
  parameters vary by row, exposure/covariates must remain aligned, or the data
  likelihood is joint/dependent (for example an HMM).
- `zero_inflation=True` makes a separate zero bin; it does **not** add an atom
  or an inflation probability to `dist`. The implementation selects positive
  data for nonzero quantiles, so it is intended for nonnegative measurements,
  not generic signed outcomes. Use a properly specified hurdle/atom likelihood
  for structural zeros.

Small exact-compression example (optional `xhistogram` must be installed):

```python
import numpy as np
import pymc as pm
from pymc_extras.distributions import histogram_approximation

counts = np.array([0, 1, 1, 2, 3, 1, 0, 2], dtype=np.int64)
with pm.Model() as histogram_model:
    rate = pm.Exponential("rate", lam=0.5)
    histogram_approximation("counts_logp", pm.Poisson.dist(mu=rate), observed=counts)
```

Do not also register the same counts as an observed Poisson likelihood: that
would double-count them. A Potential is not an observed RV with a predictive
callback; prior/posterior predictive sampling does not use it as a generative
observation law. Generate new counts from a separate Poisson RV with the fitted
rate. There is no automatic row-level log-likelihood group for LOO; reconstruct
log likelihood at the intended observational unit from original data.

[Histogram implementation](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/histogram_utils.py).

## PartialOrder: strict inequalities on selected entries

Use `PartialOrder` when scientific constraints form a directed acyclic graph,
not necessarily a total sort. For a NumPy adjacency array, `adj[i,j]=1` means
**value `i` is strictly smaller than value `j`**. The final square axes describe
the graph; corresponding RV entries lie on the final axis, with matching graph
batch axes immediately before it. Transitive closure is computed automatically;
cycles/equalities are rejected. Use a binary graph with at least one edge (the
release input check also rejects an all-zero adjacency matrix), and obtain
valid initial values with `transform.initvals(shape=...)`.

```python
import numpy as np
import pymc as pm
from pymc_extras.distributions.transforms import PartialOrder

# 0 < 1 and 0 < 2; no ordering assumption between 1 and 2.
adjacency = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
order = PartialOrder(adjacency)
with pm.Model() as ordered_model:
    values = pm.Normal(
        "values", mu=0.0, sigma=1.0, shape=(3,), transform=order,
        initval=order.initvals(shape=(3,)),
    )
```

Backward transformation uses positive exponential gaps above the largest
parent; the Jacobian is included by PyMC. This is a constrained sampling
parameterization, **not sorting of random draws**. Prior-predictive generation
from the base Normal ignores its sampling transform and can violate the order;
for a generatively ordered prior, construct roots and positive gaps explicitly.
Restricting a base density to an order region also requires attention to its
normalizing constant, especially if that constant depends on unknown
parameters or a model-comparison calculation. A max over multiple parents
introduces nondifferentiable ties; diagnose geometry, not just initial order.

[Transform implementation and examples](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/transforms/partial_order.py).

## R2D2M2CP: follow the shrinkage workflow

Use [prior elicitation: shrinkage](../../prior-elicitation/references/shrinkage.md)
for variance-budget interpretation, predictor scaling, regularization and
prior-predictive checks. Consult the
[0.15.1 helper source](https://github.com/pymc-devs/pymc-extras/blob/v0.15.1/pymc_extras/distributions/multivariate/r2d2m2cp.py)
for the consuming environment. `R2D2M2CP` is a prior-construction helper, not an
observational likelihood. It returns named fields **`eps`** (residual SD) and
**`beta`** (coefficients); `output_sigma` is total outcome SD, while positive
`input_sigma` gives predictor SDs. Center predictors; use ones for already
unit-SD predictors and a separate intercept prior. Supply named `dims` and an
explicit allocation: `variables_importance` provides Dirichlet concentrations,
or normalized `variance_explained` supplies fixed shares. The release still
returns ones when all allocation arguments are omitted, not normalized `1/p`.
Neither this variance budget nor continuous shrinkage is a variable-inclusion
probability or guaranteed realized sample R-squared for correlated predictors.
