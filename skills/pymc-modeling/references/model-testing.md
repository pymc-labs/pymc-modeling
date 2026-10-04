# Meaningful PyMC model testing

A useful test rejects a plausible wrong model. Keep three layers separate:

| Layer | Checks | Cannot establish |
|---|---|---|
| Fast model/consumer behavior | Real graph evaluation, data-update response, support/dimension boundaries, scoped mock sampling | Posterior recovery or convergence. |
| Numerical extension correctness | Independent normalized densities, analytic/finite-difference derivatives, nonuniform cotangents | Global correctness, geometry or calibration. |
| Actual inference | Real saved chains, independent posterior references, diagnostics/PPCs | Validity for a different model, data or backend. |

Prefer a small number of behavioral tests to a large API/wiring suite. A variable
name or successful import cannot show that the likelihood uses the right data.

## Scope mock sampling tightly

In [PyMC 6.3.1 testing source](https://github.com/pymc-devs/pymc/blob/v6.3.1/pymc/testing.py),
`mock_sample` draws prior predictions, renames prior to posterior, repeats the same
array across chain labels, removes prior/prior_predictive and supplies no
sample_stats unless callbacks fabricate them. These identical chain copies are
not inference, regardless of group names or attractive diagnostics. Posterior
predictive observations still require an explicit separate operation.

```python
import pytest
import pymc as pm
from pymc.testing import mock_sample

@pytest.fixture
def mock_sampling(monkeypatch):
    monkeypatch.setattr(pm, "sample", mock_sample)
```

Patch the call site used by the project and let the fixture restore it. Never leave
a global sampling replacement active for real inference. The broader official
setup/teardown helper also replaces Flat/HalfFlat with Normal/HalfNormal; those
prior changes cannot support scientific conclusions. Do not manufacture sampler
statistics merely to make a diagnostic consumer pass. Warmup/stat groups vary
across real samplers too; do not pin incidental universal group lists.

## Test mathematical behavior

**Full model density.** For Gaussian regression, independently sum the coefficient
prior log densities and observation Normal log densities at alpha+beta*x. Check
multiple parameter points so a reversed predictor or omitted likelihood fails.
A finite synthetic generating coefficient need not lie in every posterior interval;
the exact conditional Gaussian posterior is a stronger recovery reference.

**Data updates.** Use mock coefficient values only as inputs to the real prediction
graph. Change x by a nonconstant delta and check that mu changes by beta*delta with
preserved row identities. This catches a dropped slope, stale data or broadcasting
error without asserting a mock echoes its configuration. Restore inputs in finally.

**Pullbacks.** A gradient of sum(logp) uses all-one cotangents and can miss a broken
chain rule. For vector output use nonuniform signed weights and compare the weighted
objective derivative with central differences of an independent density. Count
derivatives stay undefined; returning zero is not a valid repair.

**Support and deterministic objectives.** Validate fractional/overflow/out-of-range
counts before casts. `pymc.testing.assert_no_rvs` checks that derived objectives
contain no unresolved measurable randomness. One finite evaluation does not prove
objective determinism. Graph equality helpers (`assert_equivalent_model`,
`equal_computations_up_to_root`) check structure/roots, not mathematical equivalence
or scientific adequacy; equivalent algebra can have different graphs.

## Numerical distribution helpers

`check_logp`, `check_logcdf`, `check_logccdf` and `check_icdf` accept finite Domain
objects and independent oracles such as SciPy logpdf/logcdf/logsf/ppf. Keep invalid
parameter checks and meaningful support boundaries. Declare dtype/constant policy
before building tight numerical oracles; do not relax tolerances to conceal a
lower-precision intermediate.

Domain's first/last supplied values are excluded edges unless `edges=` says
otherwise. Supply explicit edges or duplicate endpoints when testing the endpoint
itself. A test domain is not a scientific prior. `n_samples=-1` can cover the
finite parameter grid; quantile helpers have their own probability-grid contract.

Discrete CDF-versus-summed-PMF and CDF/quantile self-consistency are useful secondary
invariants, but shared errors can cancel. `assert_support_point_is_expected` checks
an initializer, not a mean. Avoid copying the library's internal random-distribution
harness into an applied project: exact seeded draws mostly check wiring, and
repeated KS/chi-square tests need false-positive/discreteness analysis.

## Simulated-data experiments

For a new or substantially changed model, choose scientifically plausible known
parameters and generate data with the relevant design, observation process and
missingness/censoring mechanism. Fit the intended model and compare important
parameters, contrasts and predictions with the generating values. Use an
independent generator or analytic reference where practical so shared coding
errors are less likely to pass unnoticed.

Vary sample size, noise, group imbalance and identification to find failure
boundaries. Deliberately try cases that should expose missing information or
break the implementation. Distinguish computation failure from data that cannot
identify a parameter. A fixed-parameter experiment diagnoses behavior at that
scenario; it is not prior-averaged calibration, and one interval missing the
truth is not evidence of a bug.

### Scoped simulation-based calibration

Use SBC when validating a reusable model/inference implementation or investigating
suspected computational bias. It is not a mandatory certification step for every
routine fit. Define the model, design, test quantities and computational scope:

1. Draw a parameter vector from the **proper joint prior**, including hierarchical
   dependencies, then generate observations conditional on it and the design.
2. Fit that simulated dataset with the same model and prior. Use real inference;
   retain failed fits and warnings rather than retrying seeds or silently excluding
   difficult replications.
3. For each chosen scalar test quantity `T(theta, y)`, rank its generating value
   among `M` posterior values evaluated on that same simulated dataset. Ranks
   range from `0` to `M`. Randomize ties and account for MCMC dependence before
   treating draws as exchangeable or applying iid rank-uniformity bands.
4. Repeat over independently generated datasets. Inspect rank histograms or ECDFs
   with finite-replication uncertainty, alongside fit diagnostics. Choose quantities
   sensitive to plausible errors, including data-dependent quantities when needed;
   parameter-only ranks can miss an implementation that simply returns the prior.

Calibration is averaged over the declared generating distribution, not guaranteed
at every fixed parameter value. Gross failures can appear in a few repetitions;
small biases need more power. Limited test quantities and cancellation across
regions can conceal errors, so uniform-looking ranks are not universal proof.
Broad priors may generate irrelevant or unrealistic datasets: assess prior
predictive realism and state the scope rather than silently filtering hard cases.
SBC checks computation under the assumed model, not its adequacy for real data.
Predictive PIT values and posterior predictive p-values are not SBC ranks.

## Keep real inference checks separate

Reopen actual saved DataTrees and recompute rank R-hat, bulk/tail ESS, mean MCSE
and divergences. Check independent analytic/quadrature posterior targets and
pointwise likelihood values where available. Never use mock output for LOO,
convergence or parameter-recovery claims.

For a diagnostic consumer, change one **copy** of a real chain enough to create
incompatibility and confirm rejection, preserving original output. This exercises
a scientific failure in the draws rather than forwarding a dictionary containing
a large R-hat. No generic DataTree heuristic can certify that arbitrary arrays
came from MCMC; retain the consuming project's record of how they were generated.
Rechecking diagnostics does not create new calibration or held-out performance.

Sources: [official testing reference](https://www.pymc.io/projects/docs/en/stable/api/testing.html)
and [Bayesian Workflow](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf)
(2026 corrected edition, §§4.4–4.5, 6.3 and Chapter 14).
